#!/bin/bash
# test_full_card_b.sh
# Card B: sparsity=0.7 + sparsity=0.8，共 18 个用例
# 用法: DEVICE=5 LOG_DIR=./logs/full_0410 ./test_full_card_b.sh
set -uo pipefail

# ========== 可配置项 ==========
DEVICE=${DEVICE:-5}
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
SUMMARY_LOG="$LOG_DIR/summary_card_b.log"
DIT_CSV="$LOG_DIR/dit_times_card_b.csv"
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

# ========== Card B 测试用例（共 18 个）==========
TOTAL=18; CASE_NUM=0
log "============================================================"
log "Card B 测试开始，共 ${TOTAL} 个用例，DEVICE=${DEVICE}"
log "日志目录: ${LOG_DIR}"
log "============================================================"

sleep_between() {
    CASE_NUM=$((CASE_NUM+1))
    log "进度 ${CASE_NUM}/${TOTAL} 完成，sleep ${SLEEP_BETWEEN}s..."
    [ $CASE_NUM -lt $TOTAL ] && sleep $SLEEP_BETWEEN
}

# ---- sparsity=0.7（9个）----
run_case "case19_sp0.7_start4_refresh1" "0.7" "4" "1" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 1
sleep_between

run_case "case20_sp0.7_start4_refresh2" "0.7" "4" "2" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 2
sleep_between

run_case "case21_sp0.7_start4_refresh0" "0.7" "4" "0" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 0
sleep_between

run_case "case22_sp0.7_start2_refresh1" "0.7" "2" "1" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 1
sleep_between

run_case "case23_sp0.7_start2_refresh2" "0.7" "2" "2" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 2
sleep_between

run_case "case24_sp0.7_start2_refresh0" "0.7" "2" "0" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 0
sleep_between

run_case "case25_sp0.7_start0_refresh1" "0.7" "0" "1" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 1
sleep_between

run_case "case26_sp0.7_start0_refresh2" "0.7" "0" "2" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 2
sleep_between

run_case "case27_sp0.7_start0_refresh0" "0.7" "0" "0" \
    --use_rainfusion --sparsity 0.7 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 0
sleep_between

# ---- sparsity=0.8（9个，稀疏度最高的一组）----
run_case "case28_sp0.8_start4_refresh1" "0.8" "4" "1" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 1
sleep_between

run_case "case29_sp0.8_start4_refresh2" "0.8" "4" "2" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 2
sleep_between

run_case "case30_sp0.8_start4_refresh0" "0.8" "4" "0" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 4 --mask_refresh_interval 0
sleep_between

run_case "case31_sp0.8_start2_refresh1" "0.8" "2" "1" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 1
sleep_between

run_case "case32_sp0.8_start2_refresh2" "0.8" "2" "2" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 2
sleep_between

run_case "case33_sp0.8_start2_refresh0" "0.8" "2" "0" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 2 --mask_refresh_interval 0
sleep_between

run_case "case34_sp0.8_start0_refresh1" "0.8" "0" "1" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 1
sleep_between

run_case "case35_sp0.8_start0_refresh2" "0.8" "0" "2" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 2
sleep_between

run_case "case36_sp0.8_start0_refresh0" "0.8" "0" "0" \
    --use_rainfusion --sparsity 0.8 --rainfusion_type v3 --sparse_start_step 0 --mask_refresh_interval 0
sleep_between

# ========== 汇总 ==========
log "============================================================"
log "Card B 测试结束: $(date)"
log "DiT耗时汇总 → ${DIT_CSV}"
log "------------------------------------------------------------"
column -t -s',' "$DIT_CSV" 2>/dev/null | tee -a "$SUMMARY_LOG" || cat "$DIT_CSV" | tee -a "$SUMMARY_LOG"
if [ -f "$LOG_DIR/failed_cases.txt" ]; then
    log "失败用例:"
    cat "$LOG_DIR/failed_cases.txt" | tee -a "$SUMMARY_LOG"
fi
log "============================================================"
