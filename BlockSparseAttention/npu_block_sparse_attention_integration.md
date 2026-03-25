# MindIE-SD 接入 `block_sparse_attention` 实现方案（rf_v3）

> 文档版本：2026-04，与 MindIE-SD `dev` 分支当前代码一致。

---

## 一、背景与目标

### 1.1 算子概述

`aclnnBlockSparseAttention` 是昇腾 Atlas A2/A3/950 系列上的块稀疏注意力算子，通过 `blockSparseMask`（int8 二值掩码）指定每个 Q-block 参与计算的 KV-block 集合。

**计算公式**：

```
attentionOut = Softmax(scale · query · key_sparse^T) · value_sparse
```

**关键约束**：

| 约束 | 说明 |
|---|---|
| 布局 | 仅支持 `TND` 和 `BNSD`，**不支持 BSND** |
| dtype | query/key/value 须一致，支持 float16 和 bfloat16 |
| headDim | 仅支持 64 或 128 |
| blockShapeY | 必须为 128 的倍数 |
| S 对齐 | **无需调用方 pad**；通过 `actual_seq_lengths` 传入实际长度，算子内部处理非对齐输入 |
| blockSparseMask shape | `[batch, headNum, ceil(qSeq/blockX), ceil(kvSeq/blockY)]`，dtype=int8 |
| inner_precise | 950PR/950DT 传 **4**（混合精度）；其他芯片 0=fp32 高精，1=fp16 高性能 |
| attenMask / blockTable | 必须传 nullptr |
| preTokens / nextTokens | 仅支持 2147483647（全上下文窗口） |

### 1.2 目标

新增 `sparse_type='rf_v3'` 路径：
- 复用 `rf_v2` 的空间重排逻辑（`do_tensor_rearrange_pooling`）和 avgpool
- 在 C++ plugin 层封装 `aclnnBlockSparseAttention`，注册为 `torch.ops.mindiesd.block_sparse_attention`
- Python 层提供端到端函数 `bsa_sparse_attention_v3`，同时支持 BSND 和 BNSD 输入、mask 缓存复用，供多模型接入

---

## 二、整体流程

### 2.1 rf_v3 完整数据流

```
输入 q/k/v  BSND [B,S,N,D] 或 BNSD [B,N,S,D]
      │
      ├─── cached_mask 为 None ───────────────────────────────────┐
      │                                                           │
      ▼                                                           │
do_tensor_rearrange_pooling(q, k, v, ...)                        │
      │  对非第一帧做 (hn, wn, hb, wb) 空间 block 重排             │
      │  + avgpool 得到 pooled 表示                                │
      ├─→  q_, k_, v_   同 input_layout，S' = 重排后序列长度       │
      └─→  qkv_pool     [3B, S_pool, N, D]                       │
      │                                                           │
      ▼                                                           │
get_blockwise_mask(qkv_pool, ..., return_binary=True)            │
      │  pooled QKV einsum → softmax → score_matrix              │
      │  topk 筛选保留 (1-sparsity) 比例的 block                  │
      │  强制保留首帧 blocks（protect first-frame attention）      │
      └─→  new_mask  int8 [B, N, Q_blk, KV_blk]                 │
      │                                                           │
      ├─── cached_mask 非 None ───────────────────────────────────┘
      │    do_tensor_rearrange_only(q, k, v, ...)  ← 仅重排，跳过 avgpool+mask
      │    new_mask = cached_mask                  ← 直接复用
      │
      ▼
rain_fusion_attention_v3(q_, k_, v_, new_mask, ...)
      │  actual_seq_lengths = [q_.shape[s_dim]] * B  ← s_dim 按 layout 选正确维度
      │  BSND：内部 permute→BNSD 调用算子→permute 回 BSND
      │  BNSD：直接传算子，无额外 permute
      └─→  out，与输入同 layout
      │
      ▼
_bsa_inv_rearrange(out, tq, hq, wq, input_layout)
      │  对齐路径（hq%8==0 且 wq%8==0）：对全部 tq 帧整体逆块重排
      │  余数路径：第一帧直接透传，其余帧逐段逆还原
      │  BSND/BNSD 各走对应维度，无额外 transpose
      └─→  out，恢复原始 (t,h,w) 行扫描顺序

返回 (out, new_mask)   ← new_mask 可由调用方缓存，供下一步复用
```

### 2.2 rf_v2 与 rf_v3 对比

| 维度 | rf_v2 | rf_v3 |
|---|---|---|
| 空间重排 | `do_tensor_rearrange_pooling` | 完全相同 |
| 掩码格式 | `select_idx`（int64 索引）+ `select_num_idx` | int8 二值 `block_sparse_mask` |
| 底层算子 | `torch.ops.mindiesd.rainfusionattention` | `torch.ops.mindiesd.block_sparse_attention` |
| 逆重排 | `do_tensor_inv_rearrange`（einops） | `_bsa_inv_rearrange`（reshape+permute，正确处理首帧，支持 BSND/BNSD） |
| S 对齐要求 | 无 | **无需 pad**，通过 `actual_seq_lengths` 传实际长度，算子原生支持非对齐 |
| 布局支持 | TND / BNSD / BSND | **BSND 和 BNSD**；BNSD 直连算子，无额外 permute |
| mask 缓存 | 不支持 | 支持（`cached_mask` 参数，跳过 avgpool+mask 生成） |
| 入口函数 | `rain_fusion_attention`（低层） | `bsa_sparse_attention_v3`（端到端，返回 `(out, mask)`） |

---

## 三、关键实现

### 3.1 `_bsa_inv_rearrange` — 支持 BSND/BNSD 的逆重排

`rf_v2` 的 `do_tensor_inv_rearrange` 无法直接复用，原因是 rf_v3 正向重排中**第一帧在余数路径下不做块重排**，而对齐路径所有帧均匀处理。`_bsa_inv_rearrange` 新增 `input_layout` 参数，对两种 layout 原地处理，不引入额外 transpose：

```python
# sparse_flash_attn_rf_v3.py

def _bsa_inv_rearrange(out, tq, hq, wq, input_layout="BSND"):
    """Inverse of do_tensor_rearrange_pooling (text_len=0).

    Supports BSND [B, S, N, D] and BNSD [B, N, S, D] without extra transposes.
    Aligned path (hq%8==0 and wq%8==0): un-block-rearrange all tq frames.
    Remainder path: first frame is unchanged; remaining (tq-1) frames are un-rearranged.
    """
    bnsd = (input_layout == "BNSD")
    b = out.shape[0]
    n = out.shape[1] if bnsd else out.shape[2]
    d = out.shape[3]
    hn, wn = hq // 8, wq // 8

    if hq % 8 == 0 and wq % 8 == 0:
        # aligned: (f hn wn hb wb) -> (f hn hb wn wb)
        if bnsd:
            out = (out
                   .reshape(b, n, tq, hn, wn, 8, 8, d)
                   .permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                   .reshape(b, n, tq * hq * wq, d))
        else:
            out = (out
                   .reshape(b, tq, hn, wn, 8, 8, n, d)
                   .permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
                   .reshape(b, tq * hq * wq, n, d))
        return out

    # remainder path: split first frame (unchanged) from rest
    first_frame_len = hq * wq
    hq_block   = (hq // 8) * 8
    wq_block   = (wq // 8) * 8
    hq_rem     = hq % 8
    wq_rem     = wq % 8
    block_size = hn * wn * 64   # block-rearranged tokens/frame
    h_rem_size = hq_rem * wq    # h-remainder tokens/frame

    if bnsd:
        # BNSD: S is at dim 2
        out_first = out[:, :, :first_frame_len, :]
        out_rest  = out[:, :, first_frame_len:, :]
        out_rest  = out_rest.reshape(b, n, tq - 1, hq * wq, d)
        t_block = out_rest[:, :, :, :block_size, :]
        t_h_r   = out_rest[:, :, :, block_size:block_size + h_rem_size, :] if hq_rem > 0 else None
        t_w_r   = out_rest[:, :, :, block_size + h_rem_size:, :]           if wq_rem > 0 else None
        t_block = (t_block
                   .reshape(b, n, tq - 1, hn, wn, 8, 8, d)
                   .permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
                   .reshape(b, n, tq - 1, hq_block, wq_block, d))
        if wq_rem > 0:
            t_block = torch.cat([t_block, t_w_r.reshape(b, n, tq-1, hq_block, wq_rem, d)], dim=4)
        if hq_rem > 0:
            t_block = torch.cat([t_block, t_h_r.reshape(b, n, tq-1, hq_rem, wq, d)], dim=3)
        out_rest = t_block.reshape(b, n, (tq - 1) * hq * wq, d)
        return torch.cat([out_first, out_rest], dim=2)
    else:
        # BSND: S is at dim 1
        out_first = out[:, :first_frame_len, :, :]
        out_rest  = out[:, first_frame_len:, :, :]
        out_rest  = out_rest.reshape(b, tq - 1, hq * wq, n, d)
        t_block = out_rest[:, :, :block_size, :, :]
        t_h_r   = out_rest[:, :, block_size:block_size + h_rem_size, :, :] if hq_rem > 0 else None
        t_w_r   = out_rest[:, :, block_size + h_rem_size:, :, :]           if wq_rem > 0 else None
        t_block = (t_block
                   .reshape(b, tq - 1, hn, wn, 8, 8, n, d)
                   .permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
                   .reshape(b, tq - 1, hq_block, wq_block, n, d))
        if wq_rem > 0:
            t_block = torch.cat([t_block, t_w_r.reshape(b, tq-1, hq_block, wq_rem, n, d)], dim=3)
        if hq_rem > 0:
            t_block = torch.cat([t_block, t_h_r.reshape(b, tq-1, hq_rem, wq, n, d)], dim=2)
        out_rest = t_block.reshape(b, (tq - 1) * hq * wq, n, d)
        return torch.cat([out_first, out_rest], dim=1)
```

**permute 语义对照**（两种 layout 的逆重排等价）：

| layout | 中间 shape | permute | 效果 |
|---|---|---|---|
| BSND | `[b, tq, hn, wn, 8, 8, n, d]` | `(0,1,2,4,3,5,6,7)` | swap dims 3↔4（wn↔hb） |
| BNSD | `[b, n, tq, hn, wn, 8, 8, d]` | `(0,1,2,3,5,4,6,7)` | swap dims 4↔5（wn↔hb） |

### 3.2 `bsa_sparse_attention_v3` — 端到端入口（多模型复用）

```python
# sparse_flash_attn_rf_v3.py

def bsa_sparse_attention_v3(
    q, k, v,
    latent_shape_q,          # (t, h, w)，t*h*w == S
    latent_shape_k=None,     # default equals latent_shape_q
    txt_len=0,               # currently only 0 is supported
    pool_size=128,           # block size, must be multiple of 128
    sparsity=0.5,            # sparsity ratio [0, 1)
    input_layout="BSND",     # 'BSND' or 'BNSD'; BNSD avoids extra permutes in BSA call
    head_num=None,           # inferred from q if None
    num_key_value_heads=None,
    scale=None,              # default head_dim ** -0.5
    inner_precise=4,         # 950 chip requires 4; others use 0 or 1
    cached_mask=None,        # reuse mask from previous step to skip pool+mask gen
):
    # head_num: BNSD -> q.shape[1], BSND -> q.shape[2]
    if head_num is None:
        head_num = q.shape[1] if input_layout == "BNSD" else q.shape[2]

    tq, hq, wq = latent_shape_q
    s_dim = 2 if input_layout == "BNSD" else 1   # S 所在维度

    if cached_mask is None:
        # rearrange + avgpool -> generate new mask
        q_, k_, v_, tensor_pool = do_tensor_rearrange_pooling(...)
        new_mask = get_blockwise_mask(
            tensor_pool, txt_len, sparsity, scale, pool_size,
            latent_shape_q, latent_shape_k, input_layout,
            return_binary=True,              # 直接返回 int8，无需单独函数
        )
    else:
        # rearrange only, reuse cached mask
        q_, k_, v_ = do_tensor_rearrange_only(...)
        new_mask = cached_mask

    actual_seq_lens = [q_.shape[s_dim]] * q_.shape[0]   # 取正确的 S 维
    out = rain_fusion_attention_v3(
        q_, k_, v_, block_sparse_mask=new_mask,
        ..., input_layout=input_layout,
        actual_seq_lengths=actual_seq_lens, actual_seq_lengths_kv=actual_seq_lens,
    )

    out = _bsa_inv_rearrange(out, tq, hq, wq, input_layout)  # 传入 layout
    return out, new_mask
```

### 3.3 `sparse_attention` 中的 rf_v3 分支

```python
# sparse_flash_attn.py

from .sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

elif sparse_type == "rf_v3":
    out, _ = bsa_sparse_attention_v3(
        q, k, v,
        latent_shape_q=latent_shape_q,
        latent_shape_k=latent_shape_k,
        txt_len=txt_len,
        pool_size=block_size,
        sparsity=sparsity,
        input_layout=input_layout,
        head_num=head_num,
        scale=scale,
        inner_precise=inner_precise,
    )
```

### 3.4 模型侧调用示例

```python
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import bsa_sparse_attention_v3

# BNSD 输入：直连算子，无额外 permute，性能更优
out, cached_mask = bsa_sparse_attention_v3(
    q, k, v,                      # [B, N, S, D]
    latent_shape_q=(t, h, w),
    pool_size=128,
    sparsity=0.5,
    input_layout="BNSD",
    inner_precise=4,
)

# 后续步骤：复用 mask，跳过 avgpool + mask 生成（节省约 863µs/步）
out, cached_mask = bsa_sparse_attention_v3(
    q, k, v,
    latent_shape_q=(t, h, w),
    pool_size=128,
    sparsity=0.5,
    input_layout="BNSD",
    inner_precise=4,
    cached_mask=cached_mask,
)
```

**参数选择建议**：

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `input_layout` | `'BNSD'`（若模型原生 BNSD）| BNSD 省去两次 permute，BSND 内部自动转换 |
| `pool_size` | 128 | 固定，算子约束 blockShapeY ≥ 128 |
| `sparsity` | 0.5 ～ 0.9 | 越大越稀疏；建议从 0.5 开始调，精度损失可接受再提高 |
| `inner_precise` | 4（950） / 1（其他） | 询问算子方确认当前 plugin 版本支持的值 |
| `cached_mask` | 每隔若干步更新一次 | mask 反映注意力结构，扩散过程中变化较慢，可多步复用 |
| `txt_len` | 0 | 当前仅支持 0（无文本混合序列） |

---

## 四、文件变更总览

```
MindIE-SD/
├── csrc/
│   ├── CMakeLists.txt
│   │   └── 新增：./plugin/block_sparse_attention.cpp
│   └── plugin/
│       ├── block_sparse_attention.h      新增：C++ 函数声明
│       ├── block_sparse_attention.cpp    新增：EXEC_NPU_CMD 封装
│       └── register_ops.cpp
│           └── 新增：block_sparse_attention schema + impl 注册
│
└── mindiesd/layers/flash_attn/
    ├── sparse_flash_attn_rf_v3.py        新增
    │   ├── _bsa_inv_rearrange()          逆重排（BSND/BNSD，含对齐/余数两条路径）
    │   ├── do_tensor_rearrange_only()    仅重排（mask 缓存路径使用）
    │   ├── rain_fusion_attention_v3()    低层 BSA 调用（BSND/BNSD）
    │   └── bsa_sparse_attention_v3()    端到端入口，返回 (out, mask)
    ├── sparse_flash_attn_rf_v2.py
    │   └── get_blockwise_mask()：新增 return_binary 参数
    │       return_binary=True 直接返回 int8 掩码，供 rf_v3 复用
    └── sparse_flash_attn.py
        ├── check_params：sparse_type 合法值新增 'rf_v3'
        └── rf_v3 分支：out, _ = bsa_sparse_attention_v3(...)
```

---

## 五、关键机制说明

### 5.1 非对齐序列长度处理

视频场景中 `S = t × h × w`，例如 480×832 分辨率的 latent 网格 `h=60, w=104`，`S = t × 6240`，`6240 % 128 = 96`，不是 128 的整数倍。

`aclnnBlockSparseAttention` 通过 `actualSeqLengths` 原生支持非对齐输入，调用方只需传入实际长度，无需 pad：

```python
s_dim = 2 if input_layout == "BNSD" else 1
actual_seq_lens = [q_.shape[s_dim]] * q_.shape[0]   # 取 S 所在维度
out = rain_fusion_attention_v3(
    q_, k_, v_, ...,
    actual_seq_lengths=actual_seq_lens,
    actual_seq_lengths_kv=actual_seq_lens,
)
```

> **注意**：`actual_seq_lengths` 类型须为 `c10::optional<at::IntArrayRef>`（非 `c10::OptionalIntArrayRef`），否则 `EXEC_NPU_CMD` 的 `ConvertType` 重载无法匹配，参见 `block_sparse_attention.cpp` 中的显式转换。

### 5.2 mask 缓存机制

扩散模型推理中注意力稀疏结构随时间步变化较慢，avgpool + mask 生成约占 863µs/步（pool ~746µs，mask gen ~117µs），可通过 `cached_mask` 跳过：

```
cached_mask 为 None                        cached_mask 非 None
─────────────────────────────────────      ──────────────────────────
do_tensor_rearrange_pooling (重排+pool)    do_tensor_rearrange_only (仅重排)
get_blockwise_mask(..., return_binary=True) new_mask = cached_mask
                                           ≈ -863µs/步
```

`bsa_sparse_attention_v3` 始终返回 `(out, new_mask)`，调用方决定是否在下一步传入 `cached_mask`。

### 5.3 首帧保护

视频生成中第一帧是参考帧，稀疏注意力必须保证所有 token 都能 attend to 第一帧。`get_blockwise_mask`（`return_binary=True`）在生成掩码后强制将对应首帧的行和列全部置 1：

```python
firstframe_block_num = (first_frame_len + pool_size - 1) // pool_size
if firstframe_block_num > 0:
    mask[:, :, :firstframe_block_num, :] = True   # 所有 Q-block 都 attend 首帧
    mask[:, :, :, :firstframe_block_num] = True   # 首帧 attend 所有 KV-block
```

### 5.4 布局处理策略

`aclnnBlockSparseAttention` 不支持 BSND，`rain_fusion_attention_v3` 按 layout 分别处理：

```python
if input_layout == "BSND":
    # 需要两次 permute，引入额外开销
    query = query.permute(0, 2, 1, 3).contiguous()   # [B,S,N,D] -> [B,N,S,D]
    ...
    attention_out = attention_out.permute(0, 2, 1, 3).contiguous()  # 还原
else:  # BNSD
    # 直接传算子，零额外 permute
    pass
```

因此，**原生 BNSD 的模型建议直接传 `input_layout="BNSD"`**，可省去 2 次 permute。整个调用栈（`rain_fusion_attention_v3`、`_bsa_inv_rearrange`）均原地处理 BNSD，不引入额外转换。

---

## 六、接口层次

```
模型代码（Wan2.2、HunyuanVideo 等）
    │  直接调用（推荐，可自管 cached_mask，可选 BSND/BNSD）
    ▼
bsa_sparse_attention_v3(q, k, v, latent_shape_q, input_layout, ..., cached_mask)
    │                                              返回 (out, new_mask)
    ├─ do_tensor_rearrange_pooling()     sparse_flash_attn_rf_v2.py
    │  或 do_tensor_rearrange_only()     sparse_flash_attn_rf_v3.py（缓存路径）
    ├─ get_blockwise_mask(...,           sparse_flash_attn_rf_v2.py
    │      return_binary=True)
    ├─ rain_fusion_attention_v3()        sparse_flash_attn_rf_v3.py
    └─ _bsa_inv_rearrange(...,           sparse_flash_attn_rf_v3.py
           input_layout)

MindIE-SD sparse_attention() 统一入口
    │  通过 sparse_type='rf_v3' 间接调用（不复用 mask）
    └─ out, _ = bsa_sparse_attention_v3(...)
```

---

## 七、测试

| 文件 | 说明 |
|---|---|
| `tests/plugin/test_block_sparse_attention.py` | NPU op 冒烟测试（BNSD / TND layout） |
| `tests/plugin/test_block_sparse_attention_fake_op.py` | fake op shape/dtype 验证（Meta 设备，无需 NPU）；必须独立进程运行 |
| `tests/plugin/test_rf_v3_attention.py` | rf_v3 端到端测试 |

```bash
# rf_v3 端到端测试（需连接 Ascend 设备）
python tests/plugin/test_rf_v3_attention.py

# 跳过 NPU，仅跑 mask shape 等纯 CPU 用例
MINDIE_TEST_MODE=CPU python tests/plugin/test_rf_v3_attention.py

# fake op shape 验证（Meta 设备，不需要 NPU）
python tests/plugin/test_block_sparse_attention_fake_op.py
```

| 用例 | 覆盖点 |
|---|---|
| `test_block_sparse_mask_shape_bsnd` | `get_blockwise_mask(return_binary=True)` 返回 int8，shape 正确 |
| `test_firstframe_protection_in_mask` | sparsity=0.9 时首帧对应 blocks 仍全为 1 |
| `test_bsa_sparse_attention_v3_output_shape` | 输出 shape/dtype 与输入一致 |
| `test_bsa_sparse_attention_v3_unaligned_seq_len` | S 不整除 pool_size 时能正确返回原始 shape |
| `test_bsa_sparse_attention_v3_vs_dense` | sparsity=0 时与 npu_fusion_attention 统计量误差 < 10% |
