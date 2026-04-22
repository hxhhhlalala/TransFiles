# vllm-omni 代码结构学习指南
## —— 面向低精度量化演进（W8A8 / W4A4 / FA FP8 / MoE）

> 本指南基于 `D:/code/vllm_project/vllm-omni/` 仓库分析，最后更新：2026-04-22

---

## 一、整体架构定位

vllm-omni 是一个**多模态推理框架**，覆盖 LLM 文本生成（AR）和扩散模型（Diffusion / DiT）两条路径。它**不是 vLLM 的 fork**，而是以 vLLM 为底座，在其量化体系之上叠加自己的覆盖层。

```
vllm-omni
├── 量化入口层         vllm_omni/quantization/          ← 权重量化主战场
│   └── kv_quant_npu.py                               ← FA 注意力量化（独立）
├── Attention 调度层   vllm_omni/diffusion/attention/   ← 连接配置与 FA kernel
├── 扩散模型层         vllm_omni/diffusion/             ← 具体模型（DiT）
│   └── models/hunyuan_image3/                        ← 离线量化适配参考实现
├── LLM 文本生成层     vllm_omni/model_executor/        ← AR 路径，复用 vLLM 量化
├── 平台适配层         vllm_omni/platforms/             ← NPU/CUDA/ROCm 差异
│   └── npu/models/hunyuan_fused_moe.py               ← NPU MoE 实现
└── 底座               vLLM + vllm-ascend（外部依赖）
```

**两条独立的量化路径**：
- **权重量化（Linear）**：通过 vLLM 的 `LinearBase.get_quant_method()` 钩子透明注入，`quantization_config` 字段配置
- **FA 量化（注意力）**：通过 `Attention.forward()` 动态路由到量化 kernel，`kv_cache_dtype` 字段配置

---

## 二、量化框架全景地图

### 2.1 权重量化（Linear 层）

```
vllm_omni/quantization/
├── __init__.py           # 公开 API：build_quant_config, ComponentQuantizationConfig
├── factory.py            # 工厂函数 + _OVERRIDES 注册表（新增方法的入口）
├── component_config.py   # 多组件路由（transformer 用 fp8，vae 用 int8）
├── int8_config.py        # INT8 量化实现（452行，CUDA + NPU，在线/离线四套方案）
├── gguf_config.py        # GGUF 格式量化实现（dequant+GEMM）
└── kv_quant_npu.py       # FA FP8 量化（NPU 注意力计算，非权重量化）
```

### 2.2 FA（注意力）量化

```
vllm_omni/diffusion/attention/
├── layer.py                         # Attention 模块（KV 量化调度中枢）
├── backends/
│   ├── abstract.py                  # AttentionMetadata（含 kv_cache_dtype 传递字段）
│   ├── flash_attn.py                # FlashAttentionImpl（forward_fa_quant_npu）
│   └── sdpa.py                      # SDPA 后端（float32 fallback）
└── parallel/                        # 序列并行（Ulysses / Ring）
```

---

## 三、权重量化详解

### 3.1 工厂注册机制

所有量化方法通过 `factory.py` 的 `_OVERRIDES` 字典注册：

```python
# quantization/factory.py
_OVERRIDES: dict[str, Callable[..., QuantizationConfig]] = {
    "gguf":       _build_gguf,    # GGUF 格式（lazy import）
    "int8":       _build_int8,    # INT8 动/静态量化（CUDA + NPU）
    "inc":        _build_inc,     # INC/AutoRound
    "auto-round": _build_inc,     # auto-round 别名
}
# 其他 35+ 方法（fp8, gptq, awq...）走 vLLM 原生 QUANTIZATION_METHODS 注册表
```

新增方法只需在 `_OVERRIDES` 里加一行，其余不变。

### 3.2 调用链（从配置到 kernel）

```
quantization_config="int8"
    ↓ OmniDiffusionConfig.__post_init__()       [diffusion/data.py]
    ↓ build_quant_config("int8")                [quantization/factory.py]
    ↓ DiffusionInt8Config(...)                  [quantization/int8_config.py]
    ↓ DiffusersPipelineLoader.load_model()      [diffusion/model_loader/]
    ↓ LinearBase.get_quant_method(layer,prefix) [vLLM 内部钩子]
    ↓ Int8LinearMethod.apply(layer, x, bias)    [quantization/int8_config.py]
    ↓ vllm._custom_ops / torch_npu kernel
```

### 3.3 int8_config.py 实现模式（新增低精度的参考模板）

```
DiffusionInt8Config（QuantizationConfig 子类）
├── activation_scheme: "dynamic"
├── ignored_layers: [...]
├── is_checkpoint_int8_serialized: bool
└── get_quant_method(layer, prefix) → LinearMethod

四个 LinearMethod 实现（两个平台 × 两种激活方案）：
├── Int8LinearMethod          (CUDA，离线静态)    → vllm._custom_ops
├── NPUInt8LinearMethod       (NPU，离线静态)     → torch_npu.npu_quant_matmul
├── Int8OnlineLinearMethod    (CUDA，在线动态)    → ops.scaled_int8_quant
└── NPUInt8OnlineLinearMethod (NPU，在线动态)     → torch_npu.npu_dynamic_quant
```

---

## 四、FA（Flash Attention）量化详解

### 4.1 调用链（从配置到 kernel）

```
kv_cache_dtype="fp8"（独立于 quantization_config）
    ↓ Attention.__init__()              [diffusion/attention/layer.py:26]
      _kv_cache_dtype_resolved = False  （延迟解析，避免模型构建时 context 不可用）
    ↓ Attention.forward()               [layer.py:222]
      _resolve_kv_cache_dtype()         → 读 forward_context.omni_diffusion_config.kv_cache_dtype
      _should_apply_kv_cache_quant()    → 按 step/layer 粒度门控
      attn_metadata.kv_cache_dtype = "fp8"   → 通过 metadata 传给后端
    ↓ FlashAttentionImpl.forward_npu()  [flash_attn.py:202]
      kv_cache_dtype 非 None → forward_fa_quant_npu()
      kv_cache_dtype 为 None → forward_fa_npu()（普通 FA）
    ↓ forward_fa_quant_npu()            [flash_attn.py:219]
      输入 (B,S,H,D) → 转置 (B,N,S,D)（BNSD 布局）
    ↓ fp8_rotate_quant_fa(q,k,v)        [quantization/kv_quant_npu.py:28]
      QuaRot：q_f = Q @ Hadamard，k_f = K @ Hadamard（降低量化误差）
      fa_block_quant_preprocess：Q(block=128)，K/V(block=256) → FP8 + scale
      torch_npu.npu_fused_infer_attention_score_v2(
          q_fp8, k_fp8, v_fp8,
          query/key/value_quant_mode=7,     # per-block FP8
          dequant_scale_*=对应 scale,
          out_dtype=原始dtype,              # 输出自动反量化回 BF16
      )
    ↓ 输出转置回 (B,S,H,D) 返回
```

### 4.2 平台支持与扩展机制

```python
# flash_attn.py
class FlashAttentionImpl(AttentionImpl):
    _supported_kv_cache_dtypes = {
        # "cuda": {"fp8", "fp8_e4m3"},   # 预留，待实现
        # "rocm": {"fp8", "fp8_e4m3"},   # 预留，待实现
        "npu": {"fp8", "fp8_e4m3fn"},    # 当前仅 NPU
    }
```

### 4.3 配置字段（OmniDiffusionConfig）

| 字段 | 说明 |
|------|------|
| `kv_cache_dtype` | `"fp8"` / `"fp8_e4m3fn"` 启用 FA FP8，`None` 关闭 |
| `kv_cache_skip_steps` | 跳过量化的去噪步骤，格式 `"0-9,20,25-30"` |
| `kv_cache_skip_layers` | 跳过量化的注意力层，格式 `"0,5-10"` |

### 4.4 外部依赖

| 依赖 | 用途 |
|------|------|
| `torch_npu` | `npu_fused_infer_attention_score_v2` |
| `mindiesd.layers.quant.block_quant` | `fa_block_quant_preprocess`（FP8 分块量化） |
| `msmodelslim.processor.quarot` | `create_rot`（Hadamard 旋转矩阵） |

---

## 五、Hunyuan Image3 离线量化适配（参考实现）

HunyuanImage3 是目前框架内**第一个完整支持离线量化的 DiT 模型**，其适配代码是后续其他模型的参考模板。

### 5.1 整体架构

```
HunyuanImage3Pipeline                    [pipeline_hunyuan_image3.py]
    ↓ quant_config = od_config.quantization_config
    ↓ self.model = HunyuanImage3Model(config, quant_config=quant_config)
    
HunyuanImage3Model                       [hunyuan_image3_transformer.py:1895]
    ↓ self.quant_config = quant_config
    ↓ HunyuanImage3DecoderLayer(config, quant_config, layer_idx)
    
HunyuanImage3DecoderLayer                [transformer.py:1696]
    ├── HunyuanAttention(quant_config=quant_config)
    │     ├── QKVParallelLinear(quant_config=quant_config)   ← 自动量化
    │     └── RowParallelLinear(quant_config=quant_config)   ← 自动量化
    ├── HunYuanSparseMoeBlock(quant_config=quant_config)     ← MoE 层
    │     ├── ReplicatedLinear(gate, quant_config=quant_config)
    │     └── HunyuanFusedMoE(quant_config=quant_config)    ← Expert
    └── HunYuanMLP(quant_config=quant_config)               ← Dense 层
          ├── MergedColumnParallelLinear(quant_config=quant_config)
          └── RowParallelLinear(quant_config=quant_config)
```

**注意**：Attention 的 `Attention(...)` 对象（负责 FA 计算）没有传 `quant_config`，FA 量化走独立的 `kv_cache_dtype` 路径。两条路径互不干扰。

### 5.2 离线量化权重加载机制

`HunyuanImage3Model.load_weights()` 包含了完整的离线量化权重处理逻辑：

#### 关键处理 1：KV Cache Scale 加载（compressed-tensors 格式）

```python
# transformer.py:2066
if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
    param = params_dict[scale_name]
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight[0])
    continue
```

#### 关键处理 2：QKV Weight Scale 分片加载

离线量化 checkpoint 中 `qkv_proj.weight_scale` 是合并的，需要按 Q/K/V head 数量切分：

```python
# transformer.py:2003
split_params_mapping = [
    # 普通权重分片
    (".gate_up_proj",   ".gate_and_up_proj",  2, ...),
    (".qkv_proj.weight",".qkv_proj.weight", num_q+num_kv*2, [...], self._split_qkv_weight),
    # 量化 scale 也需要同样的分片逻辑
    (".qkv_proj.weight_scale", ".qkv_proj.weight_scale",
     num_q+num_kv*2, [("q",num_q),("k",num_kv),("v",num_kv)],
     self._split_qkv_weight),
]
```

#### 关键处理 3：FP8 KV Scale 名字重映射

```python
# transformer.py:2189
name = maybe_remap_kv_scale_name(name, params_dict)  # 来自 vllm
```

#### 关键处理 4：Expert 权重加载

MoE Expert 权重通过 `expert_params_mapping` 按 `(param_name, weight_name, expert_id, shard_id)` 四元组映射加载，支持量化后的 expert 权重和对应的 scale。

### 5.3 MoE 平台分发机制

Hunyuan Image3 的 MoE 实现是 vllm-omni 中第一个完整的 MoE 后端，采用**平台分发工厂模式**：

```python
# diffusion/models/hunyuan_image3/hunyuan_fused_moe.py

class HunyuanFusedMoE:
    def __new__(cls, **kwargs):
        # 通过平台接口获取当前平台的 MoE 实现类名
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname("hunyuan_fused_moe")
        )
        return impl(**kwargs)
```

平台 → 实现 的映射：

| 平台 | 实现类 | 继承自 |
|------|--------|--------|
| CUDA（默认） | `HunyuanFusedMoEDefault` | `vllm.SharedFusedMoE` |
| NPU（昇腾） | `AscendHunyuanFusedMoE` | `vllm_ascend.AscendSharedFusedMoE` |

NPU 实现的关键细节（`platforms/npu/models/hunyuan_fused_moe.py`）：
- 继承 `AscendSharedFusedMoE`，量化 kernel 由 vllm-ascend 的 MoE 实现提供
- 每次 `forward` 前调用 `_set_hunyuan_fused_moe_forward_context()`，设置 `moe_comm_type`（AllGather/AllToAll，按 SOC 型号选择）
- `prepare_hunyuan_fused_moe_runtime()` 在启动时初始化 MC2 通信组（仅 NPU EP 并行时需要）

### 5.4 加载一个离线量化模型的端到端流程

```python
from vllm_omni import Omni

# 使用离线 INT8 量化 checkpoint（is_checkpoint_int8_serialized=True）
model = Omni("path/to/hunyuan-image3-int8",
    quantization_config={
        "method": "int8",
        "is_checkpoint_int8_serialized": True,
    }
)
```

执行路径：
1. `build_quant_config({"method":"int8","is_checkpoint_int8_serialized":True})` → `DiffusionInt8Config`
2. `HunyuanImage3Pipeline.__init__()` → `HunyuanImage3Model(quant_config=cfg)`
3. 每个 `QKVParallelLinear(...,quant_config=cfg)` → `LinearBase.get_quant_method()` → `NPUInt8LinearMethod` / `Int8LinearMethod`
4. `LinearBase.create_weights()` 分配 int8 权重参数（int8 tensor + scale tensor）
5. `load_weights()` 迭代 checkpoint：
   - 普通权重 → `default_weight_loader` / `param.weight_loader`
   - `*.weight_scale` → `split_params_mapping` 按头数切分
   - `*.kv_scale` → `maybe_remap_kv_scale_name` 重映射
   - Expert 权重 → `expert_params_mapping` 四元组映射
6. `process_weights_after_loading()` 对每层做后处理（转置、squeeze scale）

---

## 六、两条量化路径对比总结

| 维度 | 权重量化（Linear）| FA 量化（注意力）|
|------|-----------------|----------------|
| **配置字段** | `quantization_config` | `kv_cache_dtype` |
| **量化对象** | 模型权重（W）+ 激活（A） | 注意力计算中的 Q/K/V |
| **量化时机** | 模型加载时（离线）/ forward 时（在线）| 每次 forward 动态量化 |
| **核心实现** | `int8_config.py` 等 LinearMethod | `kv_quant_npu.py` |
| **注册机制** | `factory.py` _OVERRIDES | `_supported_kv_cache_dtypes` |
| **调用路径** | LinearBase.get_quant_method() | Attention.forward() → FlashAttentionImpl |
| **平台** | CUDA + NPU (int8)，全平台 (vLLM 方法) | 当前仅 NPU |
| **可叠加** | 是 | 是 |
| **附加技术** | — | QuaRot (Hadamard 旋转) |

---

## 七、后续开发规划与指南

基于当前框架能力，下面是 MM 量化和 MoE 量化的完整开发路线图。

---

### 7.1 MM 量化总体框架

```
┌─────────────────────────────────────────────────────────┐
│                   MM 量化两条路径                         │
├─────────────────────────┬───────────────────────────────┤
│    在线量化（直转量化）     │       离线量化（校准量化）       │
│  推理时动态计算 scale      │   预先校准，checkpoint 中含 scale │
│  不需要修改 checkpoint    │   checkpoint 需提前量化          │
│  精度略低，部署最简单       │   精度更高，需要额外工具链         │
├─────────────────────────┴───────────────────────────────┤
│                     AR 模型路径                           │
│  vllm_omni/model_executor/ + vllm_omni/platforms/npu/    │
├─────────────────────────┬───────────────────────────────┤
│         DiT 模型路径     │                               │
│  vllm_omni/diffusion/   │                               │
└─────────────────────────┴───────────────────────────────┘
```

---

### 7.2 任务 A1：AR + DiT 在线量化（直转量化算法）

#### 目标

- AR 模型（Qwen2.5-Omni、Qwen3-Omni 等）支持推理时自动量化
- DiT 模型（Flux、HunyuanImage3 等）支持推理时自动量化
- 无需预量化 checkpoint，直接加载 FP16/BF16 权重后量化

#### 现状

- `int8_config.py` 中的 `Int8OnlineLinearMethod` / `NPUInt8OnlineLinearMethod` 已实现 INT8 在线量化
- 需要在此基础上：①覆盖更多精度（W4A8、W4A4）；②AR 路径复用 DiT 的量化框架

#### 开发路径

**阶段 1：INT8 在线量化覆盖 AR 路径**

AR 路径（`vllm_omni/model_executor/`）复用 vLLM 的量化钩子，理论上 `build_quant_config("int8")` 对 AR 也生效。需要验证：
- 文件：`vllm_omni/model_executor/models/qwen2_5_omni/` 中各 Linear 层是否正确接收 `quant_config`
- 测试入口：`Omni("Qwen2.5-Omni", quantization_config="int8")`

**阶段 2：扩展 W4A8 在线量化**

```
新建 vllm_omni/quantization/int4_online_config.py
```

核心逻辑：
- 权重在加载后量化到 INT4（`LazyWeightMixin` 模式）
- 激活值在每次 forward 时量化到 INT8（动态 per-tensor）
- GEMM：调用 W4A8 混合精度 kernel

```python
class DiffusionW4A8Config(QuantizationConfig):
    group_size: int = 128      # per-group 量化的组大小
    activation_bits: int = 8   # 激活值量化位数
    weight_bits: int = 4       # 权重量化位数

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if current_omni_platform.is_npu():
                return NPUW4A8LinearMethod(self)
            return W4A8LinearMethod(self)
```

**注册**（`factory.py`）：

```python
"w4a8": lambda **kw: DiffusionW4A8Config(**kw),
```

**算法**（权重量化部分）：

| 方案 | 量化粒度 | scale 存储 | 适用场景 |
|------|---------|-----------|---------|
| per-tensor | 全层一个 scale | 1 个 FP16 | 最快，精度最低 |
| per-channel | 每输出通道一个 scale | `out_features` 个 | 推荐 baseline |
| per-group | 每 group_size 列一个 scale | `in_features/group_size` 个 | 精度最好 |

**阶段 3：W4A4 在线量化**

在 W4A8 基础上，激活值也量化到 INT4：
- 激活值量化：`x_int4, x_scale = dynamic_int4_quant(x)` — 需要硬件支持 INT4 GEMM
- NPU：`torch_npu.npu_quant_matmul_v4`（假设存在，需确认算子可用性）
- CUDA：CUTLASS INT4 kernel 或 `bitsandbytes` INT4 GEMM

---

### 7.3 任务 A2：AR 离线量化（复用 vllm-ascend 算法）

#### 目标

AR 模型使用 vllm-ascend 的优化量化算法（例如 SmoothQuant、GPTQ-Ascend 等），加载预量化 checkpoint 进行推理。

#### 现状

vllm-ascend 通过 `NPUOmniPlatform` 继承 `NPUPlatform` 的方式将 vllm-ascend 的能力注入 vllm-omni AR 路径，vllm-ascend 已实现的量化方法（如 `"compressed-tensors"`、自定义的昇腾 INT8 方案）可以直接通过 vLLM 的注册表使用。

#### 开发路径

**验证当前复用情况**：

```python
# AR 路径验证：Qwen2.5-Omni + vllm-ascend 量化 checkpoint
model = Omni("path/to/qwen2.5-omni-int8",
    quantization_config="compressed-tensors",  # vllm-ascend 已注册
)
```

关键文件：
- `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py`：AR NPU 推理
- `vllm_omni/platforms/npu/platform.py`：`NPUOmniPlatform` 继承 vllm-ascend `NPUPlatform`

**如需扩展**：如果 vllm-ascend 的量化方法需要特殊的 AR 模型权重加载逻辑，在对应模型的 `load_weights()` 中按 HunyuanImage3 的模式添加：
- `quant_config.get_cache_scale(name)` 处理 KV scale
- `maybe_remap_kv_scale_name()` 处理 FP8 KV scale 重映射
- 合并权重（QKV、gate_up）的 scale 分片处理

---

### 7.4 任务 A3：DiT 离线量化（优化量化算法）

#### 目标

DiT 模型（以 HunyuanImage3 为代表）支持高精度离线量化推理，在 omni 框架上提供通用解决方案。

#### 现状

HunyuanImage3 已完成离线量化适配（见第五节），目前依托 vllm-ascend 的量化算子，但模型侧代码在 vllm-omni 仓中。下一步需要：
1. 将 HunyuanImage3 的适配模式泛化为框架级通用能力
2. 开发独立于 vllm-ascend 的优化量化算法（或在 omni 框架内封装 vllm-ascend 调用）

#### 开发路径

**步骤 1：提取通用的离线量化权重加载框架**

当前 HunyuanImage3 的量化权重加载逻辑散布在模型的 `load_weights()` 中，需要抽象成可复用组件：

```
新建 vllm_omni/quantization/weight_loader_utils.py
```

```python
def handle_quant_scale_weight(name, loaded_weight, quant_config, params_dict):
    """统一处理量化 scale weight 的加载，包括 KV scale 和 weight scale。"""
    # 1. compressed-tensors KV scale
    if quant_config and (scale_name := quant_config.get_cache_scale(name)):
        ...
    # 2. FP8 KV scale 重映射
    name = maybe_remap_kv_scale_name(name, params_dict)
    ...

def build_split_params_with_scale(stacked_params_mapping, num_q_heads, num_kv_heads):
    """在 stacked_params_mapping 基础上，自动补充对应 weight_scale 的分片规则。"""
    ...
```

**步骤 2：DiT 通用模型基类中集成量化加载**

```
vllm_omni/diffusion/models/base.py（如果存在或新建）
```

为 DiT 模型提供 `BaseQuantDiTModel`，内置通用的量化权重加载逻辑，各模型继承并覆盖需要定制的部分。

**步骤 3：实现优化量化算法（如 SmoothQuant for DiT）**

DiT 的激活分布与 LLM 不同（没有 outlier-heavy 的特性），但仍需分析各层激活统计。如果复用 vllm-ascend 的算法：

```python
# quantization/int8_config.py 扩展
class DiffusionInt8Config:
    smoothquant_alpha: float = 0.5   # SmoothQuant 迁移因子
    # 其他优化算法参数
```

如果在 omni 框架内重写：
- 新建 `vllm_omni/quantization/smoothquant_config.py`
- 实现 `SmoothQuantLinearMethod`：在 `process_weights_after_loading()` 中应用 alpha 缩放

**步骤 4：校准工具（可选）**

如果需要在 omni 框架内做 calibration：
```
新建 vllm_omni/quantization/calibration/
├── calibrator.py          # 运行模型收集激活统计
├── scale_searcher.py      # 搜索最优 scale
└── checkpoint_converter.py # 将 FP16 checkpoint 转为量化 checkpoint
```

---

### 7.5 任务 B：MoE 量化后端构建

#### 目标

在 omni 框架中为 DiT 模型构建通用的 MoE backend，支持：
- Expert 并行（EP）
- 量化（Expert 权重量化）
- 多平台（CUDA / NPU）

#### 现状分析

HunyuanImage3 已经有一个 MoE 后端实现，但是**强耦合于 hunyuan_image3 模型**：

```
当前实现（model-specific）：
  diffusion/models/hunyuan_image3/hunyuan_fused_moe.py  ← 工厂
  platforms/npu/models/hunyuan_fused_moe.py             ← NPU 实现

依赖链：
  HunyuanFusedMoE → get_diffusion_model_impl_qualname("hunyuan_fused_moe")
                  → AscendHunyuanFusedMoE（NPU）
                  → vllm_ascend.AscendSharedFusedMoE
```

需要将其泛化为框架级的 `DiffusionFusedMoE`，供所有 DiT MoE 模型复用。

#### 开发路径

**步骤 1：抽象通用 MoE 后端接口**

```
新建 vllm_omni/diffusion/layers/fused_moe.py
```

```python
class DiffusionFusedMoE:
    """通用 DiT MoE 后端，平台分发工厂。"""
    def __new__(cls, *, prefix="", **kwargs):
        op_name = "diffusion_fused_moe"
        current_omni_platform.prepare_diffusion_op_runtime(op_name)
        impl_cls = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname(op_name)
        )
        return impl_cls(prefix=prefix, **kwargs)

    @classmethod
    def make_expert_params_mapping(cls, model, ...):
        impl_cls = resolve_obj_by_qualname(...)
        return impl_cls.make_expert_params_mapping(model, ...)
```

**步骤 2：注册平台实现**

在平台接口中扩展 `get_diffusion_model_impl_qualname`：

```python
# platforms/npu/platform.py
def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
    if op_name == "hunyuan_fused_moe":
        return "...AscendHunyuanFusedMoE"
    if op_name == "diffusion_fused_moe":          # 新增通用 MoE
        return "...AscendDiffusionFusedMoE"
    return super().get_diffusion_model_impl_qualname(op_name)
```

**步骤 3：实现各平台 MoE 后端**

CUDA 默认实现（继承 vLLM `SharedFusedMoE`）：

```
新建 vllm_omni/diffusion/layers/cuda_fused_moe.py
class CUDADiffusionFusedMoE(SharedFusedMoE):
    def forward(self, hidden_states, router_logits):
        _set_forward_context_num_tokens(hidden_states.shape[0])
        return super().forward(hidden_states, router_logits)
```

NPU 实现（继承 `AscendSharedFusedMoE`，整合 MC2 初始化）：

```
新建 vllm_omni/platforms/npu/layers/ascend_diffusion_fused_moe.py
class AscendDiffusionFusedMoE(AscendSharedFusedMoE):
    # 基于 AscendHunyuanFusedMoE，但去掉 hunyuan 特定逻辑
```

**步骤 4：MoE 量化支持**

MoE expert 量化通过 vLLM 的 `FusedMoE` 量化机制支持，关键是 expert weight 的 `weight_loader` 在量化场景下能正确处理：

- `weight_loader(param, expert_weight, shard_id=shard_id, expert_id=expert_id, return_success=True)`
- 量化 checkpoint 中的 expert scale（`w2_scale`、`w13_scale` 等）的加载路径

需要在通用 `load_weights` 工具中支持 expert scale 的识别和加载：

```python
# 当前 HunyuanImage3 的 expert_params_mapping 处理方式（参考）：
for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
    weight_loader(param, loaded_weight[offset:offset+units],
                  name_mapped, shard_id=shard_id, expert_id=expert_id,
                  return_success=True)
```

**步骤 5：HunyuanImage3 迁移**

将 `HunyuanFusedMoE` 迁移到 `DiffusionFusedMoE`：
- `hunyuan_fused_moe.py` 改为 `DiffusionFusedMoE` 的别名/子类，保持向后兼容
- 新增模型直接使用 `DiffusionFusedMoE`

---

### 7.6 开发优先级与依赖关系

```
依赖关系图：

A1（在线量化）        →  不依赖其他任务，可先行
A2（AR 离线）         →  验证现有复用是否够用，低开发量
A3（DiT 离线）        →  依赖 A1（先建立框架），HunyuanImage3 是 baseline
B（MoE 后端）         →  依赖 HunyuanImage3 MoE 实现作为参考

推荐顺序：
  1. A2：验证 AR 离线量化复用（成本最低，验证当前框架边界）
  2. A1：INT8 在线量化打通 AR+DiT（框架对齐，工作量中等）
  3. A3 step1-2：提取通用加载框架（为其他 DiT 模型做铺垫）
  4. B step1-3：抽象 MoE backend（高优，影响多个模型）
  5. A1 扩展：W4A8 / W4A4（依赖硬件算子可用性）
  6. A3 step3-4：优化量化算法 + 校准工具（长期工作）
```

---

## 八、关键文件速查表

### 量化框架

| 文件 | 作用 |
|------|------|
| `vllm_omni/quantization/factory.py` | 方法注册（`_OVERRIDES`），新增量化方法的入口 |
| `vllm_omni/quantization/int8_config.py` | INT8 完整实现，新方法的参考模板 |
| `vllm_omni/quantization/kv_quant_npu.py` | FA FP8 量化（QuaRot + 分块量化 + NPU FA kernel）|
| `vllm_omni/quantization/component_config.py` | 多组件混合精度路由 |

### FA 量化

| 文件 | 作用 |
|------|------|
| `vllm_omni/diffusion/attention/layer.py` | KV 量化调度（按 step/layer 门控，metadata 传递）|
| `vllm_omni/diffusion/attention/backends/flash_attn.py` | 平台分发 + `forward_fa_quant_npu` |
| `vllm_omni/diffusion/attention/backends/abstract.py` | `AttentionMetadata.kv_cache_dtype` 字段 |

### HunyuanImage3（离线量化参考实现）

| 文件 | 作用 |
|------|------|
| `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py` | `quant_config` 传入模型 |
| `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py` | 完整量化权重加载逻辑 |
| `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_fused_moe.py` | MoE 平台分发工厂 |
| `vllm_omni/platforms/npu/models/hunyuan_fused_moe.py` | NPU MoE 实现（含 MC2 初始化）|
| `vllm_omni/platforms/npu/platform.py` | 平台 → 实现类名 映射 |

### 配置入口

| 文件 | 作用 |
|------|------|
| `vllm_omni/diffusion/data.py` | `OmniDiffusionConfig`：`quantization_config` + `kv_cache_dtype` + skip 字段 |

---

## 九、现有量化能力全景

### 已有

| 能力 | 路径 | 平台 |
|------|------|------|
| W8A8 动态（在线）量化 | int8_config.py（Int8OnlineLinearMethod）| CUDA + NPU |
| W8A8 静态（离线）量化 | int8_config.py（Int8LinearMethod）| CUDA + NPU |
| GGUF 量化推理 | gguf_config.py | CUDA |
| FP8 / GPTQ / AWQ 等 | vLLM 原生注册表（35+ 方法）| CUDA / ROCm |
| INC / AutoRound | factory.py _build_inc | CUDA |
| 多组件混合精度 | component_config.py | 全平台 |
| FA FP8 动态量化（QKV）| kv_quant_npu.py | NPU |
| FA 按步骤/层跳过 | Attention.layer.py | NPU |
| DiT 离线量化（HunyuanImage3）| transformer.py load_weights | NPU |
| DiT MoE（hunyuan）| HunyuanFusedMoE → AscendSharedFusedMoE | NPU |

### 待补充（演进方向）

| 能力 | 入手文件 | 依赖 |
|------|----------|------|
| W4A8 在线量化 | 新建 `quantization/int4_online_config.py` | W4A8 GEMM kernel |
| W4A4 在线量化 | 新建 `quantization/w4a4_config.py` | INT4 GEMM kernel |
| DiT 通用量化加载框架 | 新建 `quantization/weight_loader_utils.py` | — |
| 通用 DiT MoE backend | 新建 `diffusion/layers/fused_moe.py` | — |
| CUDA FA FP8 量化 | 新建 `quantization/kv_quant_cuda.py` | CUDA FP8 FA kernel |
| DiT 优化量化算法 | 扩展 `int8_config.py` 或新建 `smoothquant_config.py` | 校准数据 |
| 量化校准工具链 | 新建 `quantization/calibration/` | — |
