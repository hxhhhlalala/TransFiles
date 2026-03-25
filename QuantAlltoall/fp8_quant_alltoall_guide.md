# FP8 量化通信接入指南 (Ascend NPU · Ulysses 序列并行)

> 参考实现：`Wan2.2/wan/modules/attn_layer.py`（`QuantAllToAllAttention`，继承自 `xFuserLongContextAttention`）
> 调用点：`Wan2.2/wan/distributed/sequence_parallel.py`（`sp_attn_forward`，由 `QUANT_ALLTOALL=1` 选路）
> 已验证：Wan2.2（SP=4，视频生成模型）
> 最终方案：**BSND 布局 FP8 量化通信**（代码分析见第 3 节，BSND 内部拷贝次数更少）
>
> **Profiler 数据说明**：Before 组（基线）= BF16 BSND alltoall；After 组（对比）= FP8 BSND alltoall（与 Before 方向相同，代码已通过 `scatter_idx=2, gather_idx=1` 确认）。

---

## 1. 背景与动机

Ulysses 序列并行的前向注意力中，每张卡对 Q/K/V 做 All-to-All（scatter 头维度、gather 序列维度），通信量随 SP 规模线性增长。
将 Q/K/V 在 All-to-All **前**量化为 FP8，通信量减半（BF16 → FP8 = 2B → 1B/element），同时注意力计算使用专用 FP8 FA kernel，整体 end-to-end 开销下降。

---

## 2. Profiler 对比分析

### 2.1 实验设置

| 项目 | 值 |
|---|---|
| 模型 | Wan2.2 |
| SP（序列并行度） | 4 |
| 单卡 S_local | ~18900（BF16 方案）/ 18944（FP8 方案，补齐对齐） |
| N（注意力头数） | 40 |
| D（head_dim） | 128 |
| S_full（All-to-All 后全序列） | ~75600 / 75776 |

### 2.2 关键算子对比表

下表截取同一 Transformer 层中，**从 RoPE 结束到 output projection 前**的关键算子。

> "Before" = BF16 All-to-All + alltoall 后 per-head FP8 量化
> "After" = FP8 All-to-All（BSND 方向，`scatter_idx=2, gather_idx=1`）+ alltoall 前整体 FP8 量化

#### 2.2.1 旋转位置编码（RoPE）段

| Op | Before Duration (µs) | After Duration (µs) | 说明 |
|---|---|---|---|
| Cast BF16→FLOAT (Q) | 183 | 183 | RoPE 前类型转换 |
| RotaryPositionEmbedding (Q) | 265 | 275 | |
| Cast FLOAT→BF16 (Q) | 173 | 182 | |
| Cast BF16→FLOAT (K) | 189 | 187 | |
| RotaryPositionEmbedding (K) | 264 | 271 | |
| Cast FLOAT→BF16 (K) | 175 | 181 | |
| **小计** | **~1249** | **~1279** | 基本相同 |

#### 2.2.2 量化预处理段（差异最大）

| 算子流程 | Before | Duration (µs) | After | Duration (µs) |
|---|---|---|---|---|
| 旋转矩阵 | per-head Slice `[75600,1280]→[75600,128]` ×2 | ~16×2=32 | 全局 rot_T `[128,128]→[128,128]` | 6 |
| 旋转 | per-head MatMulV3 `[75600,128]×[128,128]` ×2 | ~12×2=24 | 全局 MatMulV3 `[757760,128]×[128,128]` (Q) | 112 |
| | per-head 同上 (K) | ×10 heads | 全局 MatMulV3 (K) | 110 |
| BSND→BNSD | 无（per-head 时已是 BSND） | — | Transpose `[18944,40,128]→[40,18944,128]` ×3 | 142 / 139 / 119 |
| 补齐 pad | PadV3 ×3 per head | ~10×3=30 | 无（量化时直接 BNSD 补齐） | — |
| FP8 量化 | DynamicQuantV2 `[1,1,591,16384]` (Q) ×1 per head | ~15 | DynamicQuantV2 `[1,40,148,16384]` (Q, 全头) | 168 |
| | DynamicQuantV2 `[1,1,296,32768]` (K/V) ×2 per head | ~15×2=30 | DynamicQuantV2 `[1,40,74,32768]` (K,V) | 129 / 133 |
| **每头循环代价（×10）** | ~(32+24+30+45)×10 = **1310µs** | ← 在 FA 循环内累积 | **0** | ← 已提出循环外 |
| **量化总代价（一次性）** | — | — | 6+112+110+142+139+119+168+129+133 = **~1158µs** | |

> **结论**：量化开销从「×10 循环内」挪到「循环外一次性处理」，循环体变轻，调度更规整。

#### 2.2.3 All-to-All 通信段

| 算子 | Before | Duration (µs) | Wait (µs) | After | Duration (µs) | Wait (µs) |
|---|---|---|---|---|---|---|
| 内部 Transpose（pre-A2A, Q） | `[18900,4,10,128]→[4,18900,10,128]` BF16 | 115 | **408** | `[4,10,18944,128]→[4,18944,10,128]` FP8 | 88 | 165 |
| **hcom_alltoall_ Q** | BF16 | **1044** | 143 | FP8 | **491** | 180 |
| 内部 Transpose（post-A2A, Q） | BF16 | 121 | 1 | FP8 | 62 | 1 |
| **hcom_alltoall_ K** | BF16 | **978** | 121 | FP8 | **524** | 133 |
| 内部 Transpose（post-A2A, K） | BF16 | 122 | 1 | FP8 | 60 | 1 |
| **hcom_alltoall_ V** | BF16 | **988** | 154 | FP8 | **492** | 139 |
| Scale pre-A2A Transpose（Q） | — | — | — | `[4,10,148]→[4,148,10]` | 5 | 140 |
| **hcom_alltoall_ scale Q** | — | — | — | FLOAT | **5** | 140 |
| Scale pre-A2A Transpose（K） | — | — | — | `[4,10,74]→[4,74,10]` | 3 | 219 |
| **hcom_alltoall_ scale K** | — | — | — | FLOAT | **5** | 126 |
| Scale pre-A2A Transpose（V） | — | — | — | `[4,10,74]→[4,74,10]` | 3 | 212 |
| **hcom_alltoall_ scale V** | — | — | — | FLOAT | **5** | 116 |
| **A2A 总耗时（含 Transpose）** | | **~3495µs** | | | **~1843µs** | |

> **关键收益**：FP8 A2A 耗时 **491/524/492 µs**，较 BF16 的 **1044/978/988 µs** 降低约 **50%**（与理论 2× 一致）。Scale A2A 额外开销约 26 µs，可忽略。

#### 2.2.4 post-A2A 转置段（After 特有）

After 方案 alltoall 后，BSND 输出 `[B,S_full,N/P,D]` 需要转置为 FA kernel 所需的 BNSD 格式 `[B,N/P,S_full,D]`：

| 算子 | Input Shape | Output Shape | Duration (µs) |
|---|---|---|---|
| Transpose FP8 (Q) | `[75776,10,128]` | `[10,75776,128]` | 64 |
| Transpose FP8 (K) | `[75776,10,128]` | `[10,75776,128]` | 80 |
| Transpose FP8 (V) | `[75776,10,128]` | `[10,75776,128]` | 89 |
| Transpose scale (Q) | `[592,10]` | `[10,592]` | 4 |
| Transpose scale (K) | `[296,10]` | `[10,296]` | 3 |
| Transpose scale (V) | `[296,10]` | `[10,296]` | 3 |
| **小计** | | | **~243µs** |

> 这部分是 BNSD 方案特有的额外开销，BSND 最终方案只需 1 次 transpose（BSND→BNSD 供 FA kernel）。

#### 2.2.5 Flash Attention 段

| 指标 | Before | After |
|---|---|---|
| FA kernel | `FusedInferAttentionScore` | `FusedInferAttentionScore`（同名，v2 参数） |
| 输入 layout | BNSD `[1,1,75600,128]` FP8 | BNSD `[1,1,75776,128]` FP8 |
| scale 形状（Q） | `[1,1,591,1]` FLOAT | `[1,1,592,1]` FLOAT |
| scale 形状（K/V） | `[1,1,296,1]` FLOAT | `[1,1,296,1]` FLOAT |
| 调用次数 | 10（per-head loop） | 10（per-head loop） |
| 各次耗时（µs） | 5272, 5271, 5271, 7077, 7182, 8365, 7304, 7715, 9215, 9448 | 5386, 5831, 7208, 9413, 8883, 6748, 8891, 6814, 9973, 9803 |
| **FA 总耗时（µs）** | **~72,120** | **~78,950** |

> S_full 增加（75600→75776，+0.23%），FA 时间略有上涨，也受层间差异影响。两组数据来自不同采样批次，绝对值可比性有限，趋势参考为主。

#### 2.2.6 输出 All-to-All 段

| 算子 | Before | After |
|---|---|---|
| pre-A2A Transpose | `[4,18900,10,128]→[4,10,18900,128]` BF16 | `[4,18944,10,128]→[4,10,18944,128]` BF16 |
| hcom_alltoall_ (output) | **968 µs** BF16 | **971 µs** BF16 |
| post-A2A Transpose | `[40,18900,128]→[18900,40,128]` BF16 | `[40,18944,128]→[18944,40,128]` BF16 |

> 输出 All-to-All 仍为 BF16，两方案耗时相同。

### 2.3 性能汇总

| 阶段 | Before | After | 差异 |
|---|---|---|---|
| QKV A2A（3次） | 3,010 µs | 1,507 µs | **-1,503 µs（-50%）** |
| Scale A2A（3次） | 0 µs | ~15 µs | +15 µs |
| per-head 量化（×10） | ~1,310 µs | 0 µs | **-1,310 µs** |
| 量化预处理（一次性） | 0 µs | ~1,158 µs | +1,158 µs |
| post-A2A 转置（BSND→BNSD for FA） | 0 µs | ~243 µs | +243 µs |
| **通信+量化净收益** | | | **约 -1,397 µs** |

> **结论**：FP8 量化通信将 QKV A2A 节省 ~1500 µs，per-head 量化开销迁出循环节省 ~1310 µs，合计约 **1400 µs** 净收益（不含 FA 本身的变化）。

---

## 3. 为什么最终选择 BSND 方向

对比 `all_to_all_4D` 两个分支的内部 transpose 次数：

| 分支 | 调用方式 | pre-A2A contiguous 次数 | post-A2A contiguous 次数 | 总计 |
|---|---|---|---|---|
| `scatter=2, gather=1`（BSND） | `all_to_all_4D(x, 2, 1)` | **1**（reshape + `.transpose(0,2)` + `.contiguous()`） | **1**（reshape + `.transpose(0,1)` + `.contiguous()`） | **2** |
| `scatter=1, gather=2`（BNSD） | `all_to_all_4D(x, 1, 2)` | **2**（`.transpose(0,3)` + `.transpose(0,1)` + `.contiguous()`） | **1**（`.transpose(0,2)` + `.contiguous()`） | **3** |

BNSD 方向每次 A2A 比 BSND 多一次 `.contiguous()`（内存拷贝），3 路 Q/K/V × 多 1 次 = 额外 3 次拷贝，在大 tensor 上开销显著。**最终方案采用 BSND**（代码已通过 `scatter_idx=2, gather_idx=1` 固定，Before/After 两组 profiler 数据均为 BSND 方向）。

---

## 4. 最终方案：BSND FP8 量化通信

### 4.1 数据流对比

| 步骤 | Before（BF16 A2A） | After（FP8 A2A） |
|---|---|---|
| **输入** | `BSND [B, S_local, N, D]  BF16` | `BSND [B, S_local, N, D]  BF16` |
| **循环外预处理** | — | Q/K × R（整体）<br>`fa_block_quant_preprocess(layout="BSND")`<br>→ FP8 `[B,S_local,N,D]` + scale `[B,N,⌈S_local/bs⌉,1]` |
| **输入 All-to-All** | `all_to_all_4D(BF16, s=2, g=1)`<br>→ BF16 `[B,S_full,N/P,D]` | `all_to_all_4D(FP8, s=2, g=1)` **↓50% 通信量**<br>→ FP8 `[B,S_full,N/P,D]`<br>`all_to_all_4D(scale, s=1, g=2)` + trim<br>→ scale `[B,N/P,⌈S_full/bs⌉,1]` |
| **转置 + split** | `.transpose(1,2).contiguous()`<br>→ BNSD `[B,N/P,S_full,D]` BF16<br>`.split(1,dim=1)` → N/P × `[B,1,S_full,D]` | `.transpose(1,2).contiguous()`<br>→ BNSD `[B,N/P,S_full,D]` FP8<br>`.split(1,dim=1)` → N/P × `[B,1,S_full,D]` |
| **循环内（×N/P）** | Q/K × R ← **循环内**<br>`fa_block_quant_preprocess(head_i)` ← **循环内**<br>`npu_fused_infer_attention_score_v2(…)`<br>`.transpose(1,2)` → BSND `[B,S_full,1,D]` | （旋转 + 量化已在循环外完成）<br>`npu_fused_infer_attention_score_v2(`<br>`  quant_mode=7, dequant_scale_*)`<br>`.transpose(1,2)` → BSND `[B,S_full,1,D]` |
| **循环后 cat** | `.cat(dim=2)` → BSND `[B,S_full,N/P,D]` | `.cat(dim=2)` → BSND `[B,S_full,N/P,D]` |
| **输出 All-to-All** | `all_to_all_4D(BF16, s=1, g=2)`<br>→ BF16 `[B,S_local,N,D]` | `all_to_all_4D(BF16, s=1, g=2)`<br>→ BF16 `[B,S_local,N,D]` |

**关键差异小结**：

| | Before | After |
|---|---|---|
| A2A 通信类型 | BF16（2B/elem） | FP8（1B/elem）**↓50%** |
| 旋转矩阵 | 循环内逐头应用 | A2A 前整体应用（循环外） |
| 量化时机 | A2A 后，循环内逐头（×N/P 次） | A2A 前，循环外一次性 |
| FA 调用次数 | N/P 次（逐头） | N/P 次（逐头，结构相同） |
| per-head 量化开销 | ~1,310 µs（循环内） | 0 µs（已移至循环外） |

### 4.2 代码结构

FP8 量化通信逻辑通过子类与基线完全解耦：

```
xFuserLongContextAttention          ← 基线，保持 BF16 逻辑不变
    └── QuantAllToAllAttention       ← 子类，QUANT_ALLTOALL=1 时使用
            ├── _get_rot             @classmethod，seed=42，QR 正交分解，类级缓存
            ├── _fp8_attn            BNSD FP8 输入 → npu_fused_infer_attention_score_v2 → BSND
            ├── _scale_all_to_all    scale A2A + trim
            ├── _chunk_scale_a2a     overlap 模式逐 chunk scale A2A
            ├── forward              pre_quant_fp8 为 False 时调 super().forward()
            ├── _forward_fp8_no_overlap
            └── _forward_fp8_overlap
```

调用点选路（`wan/distributed/sequence_parallel.py`）：

```python
attn_cls = QuantAllToAllAttention if int(os.getenv('QUANT_ALLTOALL', 0)) \
           else xFuserLongContextAttention
x = attn_cls(args, ...)(None, query=q, key=k, value=v, seq_lens=seq_lens, ...)
```

`QuantAllToAllAttention.forward` 内部路由逻辑：

```python
pre_quant_fp8 = (
    self.quant_alltoall and
    self.algo == 3 and
    get_sp_group().ring_world_size == 1
)
if not pre_quant_fp8:
    return super().forward(...)          # 回退到 BF16 基线
elif self.fa_alltoall_overlap:
    return self._forward_fp8_overlap(...)
else:
    return self._forward_fp8_no_overlap(...)
```

### 4.3 正交旋转矩阵

```python
class QuantAllToAllAttention(xFuserLongContextAttention):
    _rot_matrices: dict = {}  # head_dim → orthogonal matrix (CPU fp32)，类级缓存

    @classmethod
    def _get_rot(cls, head_dim, device, dtype):
        """Q 和 K 共用同一正交旋转矩阵 R，保证 Q·R·(K·R)^T = Q·K^T。"""
        if head_dim not in cls._rot_matrices:
            gen = torch.Generator()
            gen.manual_seed(42)
            # 必须用 QR 分解；torch.randn 生成的矩阵不是正交阵
            rot, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=gen))
            cls._rot_matrices[head_dim] = rot  # CPU fp32 缓存
        return cls._rot_matrices[head_dim].to(device=device, dtype=dtype)
```

> **常见错误**：Q/K 用不同的随机矩阵，导致 `Q·R₁·(K·R₂)^T ≠ Q·K^T`，注意力分数被破坏。

### 4.4 FP8 量化 + BSND All-to-All

```python
import math
import torch
import torch_npu
import torch.distributed as dist
from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess

MAX_TOKEN = 2147483647


def fp8_bsnd_alltoall(query, key, value, ulysses_pg):
    """
    输入: query/key/value = BSND [B, S_local, N, D]，BF16
    输出: query/key/value = BSND [B, S_full, N/P, D]，FP8
          q_scale/k_scale/v_scale = [B, N/P, ceil(S_full/bs), 1]，FLOAT
    """
    origin_dtype = query.dtype  # 在量化前保存，后面 FA 输出 cast 用

    # 1. 旋转（Q/K 共用同一正交矩阵）
    rot = _get_rot(query.shape[-1], query.device, query.dtype)
    query = torch.matmul(query, rot)   # [B, S_local, N, D] @ [D, D]
    key   = torch.matmul(key,   rot)

    # 2. FP8 量化（BSND 布局，避免 BNSD 方向额外的显式 transpose）
    #    Q block_size=128，K/V block_size=256（FA kernel 要求）
    q_fp8, q_scale = fa_block_quant_preprocess(
        query, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
    k_fp8, k_scale = fa_block_quant_preprocess(
        key,   block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
    v_fp8, v_scale = fa_block_quant_preprocess(
        value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout="BSND")
    # scale 形状: [B, N, ceil(S_local/bs), 1]（始终 BNSD 风格，不随 layout 变）

    # 3. FP8 All-to-All：BSND scatter N(dim=2)→gather S(dim=1)
    #    内部仅 2 次 contiguous 拷贝（比 BNSD 方向少 1 次）
    query = all_to_all_4D(q_fp8, scatter_idx=2, gather_idx=1, group=ulysses_pg)
    key   = all_to_all_4D(k_fp8, scatter_idx=2, gather_idx=1, group=ulysses_pg)
    value = all_to_all_4D(v_fp8, scatter_idx=2, gather_idx=1, group=ulysses_pg)
    # 输出: FP8 [B, S_full, N/P, D]

    # 4. Scale All-to-All：scatter N(dim=1)→gather blocks(dim=2)，无内部 reshape
    q_scale = all_to_all_4D(q_scale, scatter_idx=1, gather_idx=2, group=ulysses_pg)
    k_scale = all_to_all_4D(k_scale, scatter_idx=1, gather_idx=2, group=ulysses_pg)
    v_scale = all_to_all_4D(v_scale, scatter_idx=1, gather_idx=2, group=ulysses_pg)
    # 输出: [B, N/P, P×ceil(S_local/bs), 1]

    # 5. Trim 多余 scale 块，必须 .contiguous()
    #    原因: P × ceil(S_local/bs) ≥ ceil(S_full/bs)，最多多 1 块
    s_full = query.shape[1]   # BSND: S 在 dim=1
    q_scale = q_scale[:, :, :math.ceil(s_full / 128), :].contiguous()
    k_scale = k_scale[:, :, :math.ceil(s_full / 256), :].contiguous()
    v_scale = v_scale[:, :, :math.ceil(s_full / 256), :].contiguous()

    return query, key, value, q_scale, k_scale, v_scale, origin_dtype


def fp8_attention_and_output_a2a(query, key, value,
                                 q_scale, k_scale, v_scale,
                                 origin_dtype, ulysses_pg):
    """
    输入: BSND FP8 [B, S_full, N/P, D]
    输出: BSND BF16 [B, S_local, N, D]
    """
    # 6. BSND→BNSD for FA kernel
    q_bnsd = query.transpose(1, 2)  # [B, N/P, S_full, D]
    k_bnsd = key.transpose(1, 2)
    v_bnsd = value.transpose(1, 2)
    b, n, s, d = q_bnsd.shape

    # 7. FP8 FlashAttention
    out = torch_npu.npu_fused_infer_attention_score_v2(
        q_bnsd, k_bnsd, v_bnsd,
        input_layout="BNSD",
        num_query_heads=n,
        softmax_scale=1.0 / math.sqrt(d),
        pre_tokens=MAX_TOKEN,
        next_tokens=MAX_TOKEN,
        query_quant_mode=7,    # FP8 block quant
        key_quant_mode=7,
        value_quant_mode=7,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=origin_dtype,  # 注意：用量化前保存的类型，不能用 FP8 tensor 的 dtype
    )[0]  # BNSD BF16 [B, N/P, S_full, D]

    out = out.transpose(1, 2)  # BNSD→BSND [B, S_full, N/P, D]
    if out.shape[1] != s:      # 如有序列 padding，裁掉
        out = out[:, :s, :, :]

    # 8. 输出 All-to-All（BF16，收拢全序列）
    out = all_to_all_4D(out, scatter_idx=1, gather_idx=2, group=ulysses_pg)
    # 输出: BF16 [B, S_local, N, D]
    return out
```

### 4.5 Scale All-to-All 细节

`fa_block_quant_preprocess` 返回的 scale 始终为 `[B, N, ceil(S/bs), 1]`（BNSD 风格），与 QKV 主体的 BSND 布局正交：

```
Scale 方向：scatter N（dim=1） → gather blocks（dim=2）
→ all_to_all_4D(scale, scatter_idx=1, gather_idx=2)

Scale 输入:  [B, N,   ceil(S_local/bs), 1]
Scale 输出:  [B, N/P, P×ceil(S_local/bs), 1]   ← 需 trim
Trim 后:     [B, N/P, ceil(S_full/bs),   1].contiguous()
```

**为什么需要 trim**：`P × ceil(S_local/bs) ≥ ceil(P × S_local / bs)`，最多比实际多出 1 块。
不 trim 会让 FA kernel 读到越界的无效 scale，导致数值错误。

---

## 5. Dual-Stream 模型（txt + img）的额外处理

当模型同时处理文本（txt）和图像（img）时，FP8 FA 前需将两路拼接为联合序列。
FA kernel 的 block 以固定步长 128 划分整个联合序列；若 `S_txt % 128 ≠ 0`，img 第一块的
scale 会被错误地使用 txt 最后一块的值，造成 img 左上角 token 的注意力数值错误（表现为图像左上角噪声/模糊）。

### 5.1 txt 长度对齐填充

```python
# 在拼接前（BNSD 布局，S 在 dim=2）
txt_seq_len_orig    = txt_query.shape[2]
txt_seq_len_aligned = math.ceil(txt_seq_len_orig / 128) * 128

if txt_seq_len_aligned > txt_seq_len_orig:
    pad_s = txt_seq_len_aligned - txt_seq_len_orig
    # F.pad 参数从最后一维倒序：(D_before, D_after, S_before, S_after)
    txt_query = F.pad(txt_query, (0, 0, 0, pad_s))
    txt_key   = F.pad(txt_key,   (0, 0, 0, pad_s))
    txt_value = F.pad(txt_value, (0, 0, 0, pad_s))
    # txt_*_scale 已覆盖所有块，无需 scale padding

# 拼接（BNSD，沿 S=dim2 拼）
joint_query = torch.cat([txt_query, img_query], dim=2)
joint_key   = torch.cat([txt_key,   img_key  ], dim=2)
joint_value = torch.cat([txt_value, img_value], dim=2)
joint_query_scale = torch.cat([txt_query_scale, img_query_scale], dim=2)
joint_key_scale   = torch.cat([txt_key_scale,   img_key_scale  ], dim=2)
joint_value_scale = torch.cat([txt_value_scale, img_value_scale], dim=2)
```

### 5.2 输出切分（关键：使用原始长度而非对齐长度）

```python
# out: BSND [B, S_joint_aligned, N/P, D]
# txt 用原始长度，img 从对齐边界开始
text_out  = out[:, :txt_seq_len_orig,    :, :].contiguous()  # ← 真实 txt
image_out = out[:, txt_seq_len_aligned:, :, :].contiguous()  # ← img 从对齐边界起
```

**常见 bug**：
```python
# 错误：padding 后 txt_query.shape[2] 已是 txt_seq_len_aligned
txt_seq_len = txt_query.shape[2]              # = aligned 长度
text_out  = out[:, :txt_seq_len, :, :]        # ← 包含 0 填充行，污染 txt 下游
image_out = out[:, txt_seq_len:, :, :]        # ← img 起点正确，但 text_out 错误
```

---

## 6. Overlap 模式 Profiler 对比分析

Overlap 模式将 Q/K/V All-to-All 通信与 Flash Attention 计算在双流（dual-stream）上交叠执行，
理想情况下通信被 FA 完全隐藏。本节基于完整 profiler 原始数据对两种实现进行算子级对比。

> "Overlap Before" = BF16 per-head AsStrided 切片 + 逐头异步 A2A（无 FP8）
> "Overlap After"  = 全头整体 FP8 量化 + 输入 A2A 批量同步执行 + FP8 FA

### 6.1 整体机制对比

| 维度 | Overlap Before（BF16 per-head） | Overlap After（FP8 BSND batch） |
|---|---|---|
| 输入 A2A 总次数 | 30（10 头 × Q/K/V 3 路） | 60（含 scale，33 在 FA[0] 前，27 在 FA[0] 后） |
| Q/K/V 类型 | BF16 | FP8 |
| Q/K 提取方式 | `AsStrided` 零拷贝逐头切片 | 全头整体旋转 + BNSD Transpose + DynamicQuantV2 |
| A2A 触发时机 | **逐头异步**：头 i 的 A2A 与 FA[i-1] 真正并行 | **批量同步**：33 路 A2A 在 FA[0] 前完成；27 路在 FA[0] 后、FA[1] 前完成 |
| FA[0] 启动延迟（A2A 开始→FA[0] 起） | 首头 A2A 触发后即可准备，实测 ~776µs | 所有输入 A2A 全部完成后才能启动，实测 **~19,008µs** |
| FA[1-9] 是否有 A2A overlap | **有**（A2A 在前一头 FA 期间完成） | **无**（FA[1-9] 在所有输入 A2A 完成后串行执行） |
| 输出 A2A | 10 路 per-head + Transpose + ConcatD | 10 路 per-head + Transpose + ConcatD |

### 6.2 输入 All-to-All 结构对比

#### 6.2.1 Overlap Before（BF16 per-head，AsStrided 零拷贝）

每头循环内依次触发 Q/K/V 三路异步 A2A，Q/K 由 `AsStrided` 从全头 buffer 零拷贝切片，V 从 buffer 直接取：

| 步骤 | 算子 | 张量形状 | 代表耗时（µs） | 说明 |
|---|---|---|---|---|
| Q/K 提取 | AsStrided（零拷贝） | `[4,18900,1,1,128]` BF16 | ~13–21 | 无内存拷贝 |
| Q A2A（async） | hcom_alltoall_ BF16 | per-head 小包 | **3135**（首次冷启动）/ 101–578（后续） | 异步返回，不阻塞 |
| K A2A（async） | hcom_alltoall_ BF16 | per-head 小包 | 101–578 | |
| V A2A（async） | hcom_alltoall_ BF16 | per-head 小包 | 101–578 | |
| **30 路 A2A 累计耗时** | | | **~11,606 µs** | 与 FA 重叠执行，有效隐藏 |

#### 6.2.2 Overlap After（FP8 BSND，批量同步 A2A）

**FA[0] 前同步执行（33 路，hcom 70–102）**：

| 步骤 | 算子 | 张量形状 | 耗时 |
|---|---|---|---|
| 旋转矩阵 Transpose（rot_T） | Transpose | `[128,128]` BF16 | ~5 µs |
| Q 全头旋转 | MatMulV3 | `[757760,128]×[128,128]` | ~105 µs |
| K 全头旋转 | MatMulV3 | `[757760,128]×[128,128]` | ~104 µs |
| BSND→BNSD（Q/K/V） | Transpose ×3 | `[18944,40,128]→[40,18944,128]` | ~143/141/137 µs |
| FP8 量化（Q/K/V） | DynamicQuantV2 ×3 | Q:`[1,40,148,16384]`；K/V:`[1,40,74,32768]` | ~150/119/119 µs |
| **输入 A2A 33 路（同步串行）** | hcom_alltoall_ | 包含 scale A2A 和 head 0 FP8 A2A | **~18,465 µs**（墙钟 ~19,008 µs）|
| **FA[0] 前总等待** | | | **~19,008 µs** |

**FA[0] 后、FA[1] 前同步执行（27 路，hcom 103–129）**：

| 批次 | 调用数 | 内容 | 总耗时（µs） | 墙钟占用 |
|---|---|---|---|---|
| heads 1-9 FP8 A2A | 27 路 | 各头 Q/K/V FP8 chunk | ~5,234 µs | ~9,682 µs（含间隔） |

> **关键发现**：Overlap After 的 27 路 FP8 A2A（heads 1-9）在 FA[0] 结束后统一批量执行，
> FA[1-9] 在全部 A2A 完成后才能开始，**完全没有通信与计算的流水线重叠**。

### 6.3 FA Wait 时间对比

`Task Wait Time` 是 FA kernel 在硬件队列中等待上一个 op 释放资源的时间，
反映调度延迟。Before 的 wait 来自旋转/量化预处理；After 的 wait 来自流间同步开销。

| 头编号 | Before Wait (µs) | Before Duration (µs) | After Wait (µs) | After Duration (µs) |
|---|---|---|---|---|
| FA[0] | 210.8 | 4,719.0 | **136.4** | 4,842.8 |
| FA[1] | 207.2 | 4,721.5 | 540.9 | 4,900.2 |
| FA[2] | 143.5 | 4,763.8 | 621.6 | 4,921.4 |
| FA[3] | 256.7 | 5,798.8 | 490.7 | 4,923.3 |
| FA[4] | 223.0 | 5,717.6 | 498.4 | 5,382.5 |
| FA[5] | 147.4 | 8,948.5 | 609.2 | 5,566.2 |
| FA[6] | 155.1 | 7,720.5 | 513.3 | 5,985.3 |
| FA[7] | 142.2 | 6,116.4 | 496.3 | 6,078.5 |
| FA[8] | 134.2 | 5,988.0 | 477.4 | 8,133.6 |
| FA[9] | 142.2 | 9,587.5 | 526.9 | 10,399.8 |
| **平均 wait** | **~176 µs** | | **~491 µs** | |
| **FA 总耗时** | | **~64,082 µs** | | **~61,134 µs** |

> FA[0] 的 wait：Before 为 211µs（pre-processing 队列），After 为 136µs（数据已全部就绪）。
> FA[1-9] 的 wait：Before 平均 **168µs**（A2A 早已完成，仅 compute 调度延迟），
> After 平均 **530µs**（所有输入 A2A 在 FA[1] 之前同步完成，wait 来自跨流同步开销）。

### 6.4 输出 All-to-All 对比

两种方案的输出 A2A 均采用 per-head 模式（10 路），每路含一次 Transpose + hcom_alltoall_：

| 指标 | Overlap Before | Overlap After |
|---|---|---|
| A2A 调用次数 | 10（hcom 70–79） | 10（hcom 130–139） |
| Transpose 形状 | `[4,18900,128]→[18900,4,128]` BF16 | `[4,18944,128]→[18944,4,128]` BF16 |
| 单次 Transpose 耗时 | ~16–23 µs | ~15–20 µs |
| 单次 A2A 耗时 | 103–828 µs（末尾 828µs 异常高） | 102–221 µs |
| **A2A + Transpose 合计** | **~2,378 µs** | **~1,430 µs** |
| ConcatD | `[1,18900,4,128]×10→[1,18900,40,128]`，134 µs | `[1,18944,4,128]×10→[1,18944,40,128]`，134 µs |

### 6.5 根因分析：Overlap After 为何无法实现流水线

```
Overlap Before 时间线（理想流水）：
  A2A[head0]  ──────────────────▶              （async，3135µs 首次冷启动）
  prep[head0] ──────────▶                       （rotation + quant，~55µs）
  FA[head0]              ─────────────────────▶  （wait=211µs，duration=4719µs）
  A2A[head1]       ──────────────▶              （async，fires DURING FA[0]）
  FA[head1]                            ──────▶  （wait=207µs，数据已就绪）
  ...

Overlap After 时间线（实际退化为串行）：
  scale A2A ×30 + FP8 A2A[head0] ×3  ──────────────────────────────────────────▶ (~19,008µs)
  FA[head0]                                                                        ─────────▶ (wait=136µs)
  FP8 A2A[heads1-9] ×27               ──────────────────────────────────────────────────────▶ (~9,682µs)
  FA[head1]                                                                                    ──▶ (wait=541µs)
  FA[head2]                                                                                       ──▶
  ...（FA[1-9] 完全串行，无任何通信重叠）
```

| 根因对比 | Overlap Before | Overlap After |
|---|---|---|
| 输入 A2A 触发策略 | 逐头 **异步**（fires during FA[i-1]） | **批量同步**：FA[0] 前 33 路，FA[0] 后 27 路 |
| FA[0] 前等待 | 首头 3 路 A2A（~776µs） | 全部 scale A2A + head0 FP8 A2A（**~19,008µs**） |
| FA[1-9] 的通信状态 | A2A 已在 FA[i-1] 期间完成，**无需额外等待** | 在 FA[1] 开始前 27 路 A2A 才刚全部完成，**无 overlap** |
| FA[1-9] wait 来源 | 计算调度延迟（~168µs） | 跨流同步 + 调度延迟（**~530µs**） |
| FA 总耗时 | ~64,082 µs（含 BF16 FA） | ~61,134 µs（FP8 FA kernel 更快） |

> **结论**：Overlap After 的根本问题不是 scale A2A 预计算，而是输入 A2A **整体批量同步执行**，
> 27 路 FP8 A2A（heads 1-9）全部堆积在 FA[0] 之后、FA[1] 之前，
> 导致 FA[1-9] 毫无流水线重叠（wait 从 168µs 恶化至 530µs）。
> FA kernel 本身因 FP8 加速略快（61,134 vs 64,082µs），
> 但该收益被通信串行化带来的额外开销所抵消。
> 改进方向：将 FP8 A2A 恢复为**逐头异步触发**（类似 Before 的 AsStrided 模式），
> 使 A2A[head i+1] 在 FA[head i] 期间真正并发执行。

---

## 7. 接入 Checklist

**代码结构**
- [ ] FP8 逻辑封装在 `QuantAllToAllAttention(xFuserLongContextAttention)` 子类，不修改基线类
- [ ] 调用点通过 `QUANT_ALLTOALL=1` 环境变量选路（`sequence_parallel.py` 中 `attn_cls` 判断）
- [ ] 子类 `forward` 检查 `quant_alltoall and algo==3 and ring_world_size==1` 后再走 FP8 路径，否则 `super().forward()` 回退

**算法正确性**
- [ ] `_get_rot`：固定 seed=42，`torch.linalg.qr` 正交分解，Q/K **共用同一矩阵**，类级缓存
- [ ] `origin_dtype = query.dtype` 在量化前保存（`_fp8_attn` 内 `out_dtype=origin_dtype`，不能用 FP8 tensor 的 dtype）
- [ ] `fa_block_quant_preprocess(layout="BSND")`，Q block_size=128，K/V block_size=256
- [ ] `all_to_all_4D(fp8, scatter_idx=2, gather_idx=1)` — BSND 方向（2 次内部 transpose）
- [ ] Scale `all_to_all_4D(scale, scatter_idx=1, gather_idx=2)` + trim + `.contiguous()`
- [ ] FA 前 `.transpose(1,2)` BSND→BNSD，`_fp8_attn` 接收 BNSD 输入
- [ ] FA kernel 改为 `npu_fused_infer_attention_score_v2`，`*_quant_mode=7`
- [ ] FA 后 `.transpose(1,2)` BNSD→BSND，必要时裁掉 seq padding
- [ ] Dual-stream：FA 前 txt pad 到 128 对齐；输出切分用 `txt_seq_len_orig` / `txt_seq_len_aligned`
- [ ] Overlap 模式：避免在循环外同步预计算全部 scale A2A；应在循环内按需异步触发
