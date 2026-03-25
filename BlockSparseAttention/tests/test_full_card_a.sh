#!/bin/bash
# test_full_card_a.sh
# Card A: baseline + sparsity=0.5 + sparsity=0.6，共 19 个用例
# 用法: DEVICE=2 LOG_DIR=./logs/full_0410 ./test_full_card_a.sh
set -uo pipefail

# ========== 可配置项 ==========
DEVICE=${DEVICE:-2}
MAX_RETRY=2
SLEEP_BETWEEN=60
TODAY=$(date +%Y%m%d)
LOG_DIR=${LOG_DIR:-"./logs/full_${TODAY}"}

# ========== 环境变量 ==========
export ASCEND_RT_VISIBLE_DEVICES=$DEVICE
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=${ALGO:-0}
export OVERLAP=${OVERLAP:-0}
export FAST_LAYERNORM=1
export ROPE_OPT=1

mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary_card_a.log"
DIT_CSV="$LOG_DIR/dit_times_card_a.csv"
echo "case_name,status,dit_time_s,sparsity,sparse_start_step,mask_refresh_interval" > "$DIT_CSV"

# ========== 公共推理参数 ==========
COMMON_ARGS=(
    --task i2v-A14B
    --ckpt_dir /home/weights/Wan2.2-I2V-A14B
    --size "1280*720"
    --frame_num 81
    --sample_steps 10
    --image examples/i2v_input.JPG
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    --base_seed 0
    --quant_dit_path /home/weights/Wan2.2-I2V-A14B-w8a8c8-self-attn-bf16-rot
)

# ========== 工具函数 ==========
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUMMARY_LOG"; }

extract_dit_time() {
    local log_file="$1"
    grep "Dit used time" "$log_file" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?=s)' || echo "N/A"
}

# run_case <case_name> <sparsity_tag> <start_tag> <refresh_tag> [额外推理参数...]
# sparsity_tag/start_tag/refresh_tag 仅用于 CSV（baseline 传 "-"）
run_case() {
    local case_name="$1"
    local sp_tag="$2"
    local start_tag="$3"
    local refresh_tag="$4"
    shift 4
    local extra_args=("$@")
    local log_file="$LOG_DIR/${case_name}.log"
    local video_out="$LOG_DIR/${case_name}.mp4"

    log "------------------------------------------------------------"
    log "开始: $case_name  [sp=$sp_tag start=$start_tag refresh=$refresh_tag]"
    log "日志: $log_file"

    local attempt=1 success=0
    while [ $attempt -le $((MAX_RETRY + 1)) ]; do
        [ $attempt -gt 1 ] && log ">>> 第 $((attempt-1)) 次重试: $case_name"

        {
            echo "===== $case_name  attempt=$attempt  $(date) ====="
            torchrun --nproc_per_node=1 generate.py \
                "${COMMON_ARGS[@]}" \
                --save_file "$video_out" \
                "${extra_args[@]}"
            echo "===== exit=$?  $(date) ====="
        } 2>&1 | tee -a "$log_file"

        local exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            local dit
            dit=$(extract_dit_time "$log_file")
            log "✓ 成功: $case_name  DiT耗时=${dit}s"
            echo "${case_name},SUCCESS,${dit},${sp_tag},${start_tag},${refresh_tag}" >> "$DIT_CSV"
            success=1
            break
        else
            log "✗ 失败 exit=$exit_code  尝试 $attempt/$((MAX_RETRY+1)): $case_name"
            [ $attempt -le $MAX_RETRY ] && { log "  等待 30s 重试..."; sleep 30; }
        fi
        attempt=$((attempt+1))
    done

    if [ $success -eq 0 ]; then
        log "✗✗ 放弃: $case_name"
        echo "${case_name},FAILED,N/A,${sp_tag},${start_tag},${refresh_tag}" >> "$DIT_CSV"
        echo "FAILED: $case_name" >> "$LOG_DIR/failed_cases.txt"
    fi
}

# ========== Card A 测试用例（共 19 个）==========
TOTAL=19; CASE_NUM=0
log "============================================================"
log "Card A 测试开始，共 ${TOTAL} 个用例，DEVICE=${DEVICE}"
log "日志目录: ${LOG_DIR}"
log "============================================================"

sleep_between() {
    CASE_NUM=$((CASE_NUM+1))
    log "进度 ${CASE_NUM}/${TOTAL} 完成，sleep ${SLEEP_BETWEEN}s..."
    [ $CASE_NUM -lt $TOTAL ] && sleep $SLEEP_BETWEEN
}

# ---- case00: 基线（不使能 BSA）----
run_case "case00_baseline" "-" "-" "-"
sleep_between

# ---- sparsity=0.5（9个，稀疏度最低的一组）----
run_case "case01_sp0.5_start4_refresh1" "0.5" "4" "1" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 1
sleep_between

run_case "case02_sp0.5_start4_refresh2" "0.5" "4" "2" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 2
sleep_between

run_case "case03_sp0.5_start4_refresh0" "0.5" "4" "0" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 0
sleep_between

run_case "case04_sp0.5_start2_refresh1" "0.5" "2" "1" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 1
sleep_between

run_case "case05_sp0.5_start2_refresh2" "0.5" "2" "2" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 2
sleep_between

run_case "case06_sp0.5_start2_refresh0" "0.5" "2" "0" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 0
sleep_between

run_case "case07_sp0.5_start0_refresh1" "0.5" "0" "1" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 1
sleep_between

run_case "case08_sp0.5_start0_refresh2" "0.5" "0" "2" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 2
sleep_between

run_case "case09_sp0.5_start0_refresh0" "0.5" "0" "0" \
    --use_rainfusion --sparsity 0.5 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 0
sleep_between

# ---- sparsity=0.6（9个）----
run_case "case10_sp0.6_start4_refresh1" "0.6" "4" "1" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 1
sleep_between

run_case "case11_sp0.6_start4_refresh2" "0.6" "4" "2" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 2
sleep_between

run_case "case12_sp0.6_start4_refresh0" "0.6" "4" "0" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 0
sleep_between

run_case "case13_sp0.6_start2_refresh1" "0.6" "2" "1" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 1
sleep_between

run_case "case14_sp0.6_start2_refresh2" "0.6" "2" "2" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 2
sleep_between

run_case "case15_sp0.6_start2_refresh0" "0.6" "2" "0" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 0
sleep_between

run_case "case16_sp0.6_start0_refresh1" "0.6" "0" "1" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 1
sleep_between

run_case "case17_sp0.6_start0_refresh2" "0.6" "0" "2" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 2
sleep_between

run_case "case18_sp0.6_start0_refresh0" "0.6" "0" "0" \
    --use_rainfusion --sparsity 0.6 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 0
sleep_between

# ========== 汇总 ==========
log "============================================================"
log "Card A 测试结束: $(date)"
log "DiT耗时汇总 → ${DIT_CSV}"
log "------------------------------------------------------------"
# 打印 CSV 表格到 summary
column -t -s',' "$DIT_CSV" 2>/dev/null | tee -a "$SUMMARY_LOG" || cat "$DIT_CSV" | tee -a "$SUMMARY_LOG"
if [ -f "$LOG_DIR/failed_cases.txt" ]; then
    log "失败用例:"
    cat "$LOG_DIR/failed_cases.txt" | tee -a "$SUMMARY_LOG"
fi
log "============================================================"
