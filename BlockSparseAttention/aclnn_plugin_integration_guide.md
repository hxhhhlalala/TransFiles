# MindIE-SD 接入 aclnn 算子 Plugin 完整指南

> 以 `block_sparse_attention` 接入过程为基础整理。
> 后续新增算子时，按本文件逐节操作，重点检查"⚠️ 容易遗漏"标记处。

---

## 一、整体流程概览

```
算子开发方提供
  libcust_opapi.so          ← 自定义算子实现（aclnn C API）
  liboptiling.so            ← Tiling 策略库
  头文件 / 文档             ← aclnnXxxGetWorkspaceSize 参数顺序

          ↓ 接入 ↓

┌─────────────────────────────────────────────────────────┐
│                   MindIE-SD 接入层                       │
│                                                         │
│  【C++ Plugin 层】  csrc/plugin/                        │
│   ① xxx.h             函数声明                          │
│   ② xxx.cpp           EXEC_NPU_CMD 封装                 │
│   ③ register_ops.cpp  schema + impl 注册                │
│   ④ CMakeLists.txt    编译配置                          │
│                                                         │
│  【Python 层】  mindiesd/layers/                        │
│   ⑤ _custom_ops.py    Python wrapper + fake op 注册     │
│   ⑥ 业务调用文件      实际使用算子                       │
│                                                         │
│  【测试层】  tests/plugin/                              │
│   ⑦ test_xxx.py       单元测试                          │
└─────────────────────────────────────────────────────────┘

          ↓ 运行时 ↓

torch.ops.load_library(libPTAExtensionOPS.so)
  → C++ schema 注册到 PyTorch dispatcher
import mindiesd.layers._custom_ops
  → Python fake op 注册到 Meta dispatch key
torch.ops.mindiesd.xxx(...)
  → PrivateUse1 dispatch → EXEC_NPU_CMD → libcust_opapi.so
```

---

## 二、逐步操作清单

### 步骤 ① 新建头文件 `csrc/plugin/xxx.h`

声明 C++ 实现函数，供 `register_ops.cpp` 包含。

```cpp
#ifndef XXX_MINDIE_SD_IMPL_H
#define XXX_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

// 返回值视算子而定，常见：单 Tensor、tuple<Tensor, Tensor>
std::tuple<at::Tensor, at::Tensor> xxx_impl(
    const at::Tensor                &query,
    ...
    c10::OptionalIntArrayRef         actual_seq_lengths,   // ← 见§三.1
    int64_t                          some_flag);

#endif
```

**⚠️ 容易遗漏**：
- `include guard` 宏名与文件名保持一致，避免被其他头文件静默覆盖。
- `c10::OptionalIntArrayRef` 只能作为函数参数类型，不能直接传给 `EXEC_NPU_CMD`（见§三.1）。

---

### 步骤 ② 新建实现文件 `csrc/plugin/xxx.cpp`

```cpp
#include "pytorch_npu_helper.h"
#include "xxx.h"

namespace {
constexpr std::string_view OP_NAME = "aclnnXxx";   // ← 与算子文档一致

// 固定约束参数（询问算子开发方确认，不要自己猜）
constexpr int64_t MASK_TYPE   = 0;
constexpr int64_t BLOCK_SIZE  = 0;    // ← 见§三.2：此处 0 是算子方明确的约束
constexpr int64_t PRE_TOKENS  = 2147483647;
constexpr int64_t NEXT_TOKENS = 2147483647;
} // namespace

std::tuple<at::Tensor, at::Tensor> xxx_impl(...,
    c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv,
    int64_t some_flag)
{
    // 【必做】参数校验
    TORCH_CHECK(...);

    // 【必做】OptionalIntArrayRef 展开，见§三.1
    auto actSeqLen   = actual_seq_lengths.value_or(at::IntArrayRef{});
    auto actSeqLenKv = actual_seq_lengths_kv.value_or(at::IntArrayRef{});

    // 分配输出 Tensor
    at::Tensor out = at_npu::native::empty_with_format(
        query.sizes(), query.options(), at_npu::native::get_npu_format(query));

    // 【按需】optional 输出 Tensor，见§三.3
    c10::optional<at::Tensor> lseOpt = c10::nullopt;
    at::Tensor lse;
    if (some_flag != 0) {
        lse    = at_npu::native::empty_with_format(...);
        lseOpt = lse;
    }

    // 【关键】参数顺序严格按 aclnnXxxGetWorkspaceSize 文档
    // 判断是否需要 EXEC_BUILTIN_NPU_CMD，见§三.4
    EXEC_NPU_CMD<OP_NAME>(
        query, key, value,
        optional_mask,          // c10::optional<Tensor>  → AclTensor*(nullptr if nullopt)
        actSeqLen,              // at::IntArrayRef         → AclIntArray*
        actSeqLenKv,
        layout_cstr,            // const char*
        num_kv_heads,           // int64_t
        scale,                  // double
        inner_precise,
        BLOCK_SIZE,
        some_flag,
        out,
        lseOpt);                // c10::optional<Tensor>  → nullptr when nullopt

    return {out, lse};
}
```

**⚠️ 容易遗漏**：
- 参数顺序必须与文档 `aclnnXxxGetWorkspaceSize` 完全一致，哪怕差一个参数位置也会 crash。
- `const char*` 布局字符串需从 `std::string::c_str()` 取，不能直接传 `std::string`。
- 输出 Tensor（`out`、`lseOpt`）放在参数列表**最后**，顺序与文档一致。

---

### 步骤 ③ 修改 `csrc/plugin/register_ops.cpp`

**两处都要改，缺一不可：**

```cpp
// 1. 头文件
#include "xxx.h"

// 2. TORCH_LIBRARY 块：添加 schema（参数名/类型/默认值）
TORCH_LIBRARY(mindiesd, m) {
    ...
    m.def("xxx(Tensor query, Tensor key, Tensor value, \
        Tensor? optional_mask=None, \
        int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None, \
        str q_input_layout='BNSD', \
        int num_key_value_heads=1, float scale_value=1.0, int inner_precise=0, \
        int some_flag=0) -> (Tensor, Tensor)");
}

// 3. TORCH_LIBRARY_IMPL 块：绑定实现
TORCH_LIBRARY_IMPL(mindiesd, PrivateUse1, m) {
    ...
    m.impl("xxx", &xxx_impl);
}
```

**⚠️ 容易遗漏**：
- `TORCH_LIBRARY`（schema）和 `TORCH_LIBRARY_IMPL`（impl）是两个独立的宏块，**两处都要加**。只加 impl 不加 schema 编译报错；只加 schema 不加 impl 运行时 dispatch 失败。
- schema 里的参数名需与 Python wrapper 里 `getattr(torch.ops.mindiesd, "xxx")(...)` 的关键字参数名完全一致。
- 返回值类型 `-> Tensor` vs `-> (Tensor, Tensor)` 要与实际 C++ 返回匹配。

---

### 步骤 ④ 修改 `csrc/CMakeLists.txt`

```cmake
add_library(PTAExtensionOPS SHARED
    ./plugin/register_ops.cpp
    ...
    ./plugin/xxx.cpp)        # ← 新增这一行
```

**⚠️ 容易遗漏**：
- **这是最容易遗漏的步骤**。`.cpp` 文件写好了，但不加到 `CMakeLists.txt`，编译时不会报找不到文件的错，但链接时报：
  ```
  undefined reference to `xxx_impl`
  ```
- 加完后需要重新 `cmake` + `make`，只 `make` 可能不会重新扫描文件列表。

---

### 步骤 ⑤ 修改 `mindiesd/layers/_custom_ops.py`

**两处都要加：**

```python
# 1. Python wrapper（对外调用接口）
def xxx(
    query: torch.Tensor,
    ...
    actual_seq_lengths: Optional[List[int]] = None,
    some_flag: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return getattr(torch.ops.mindiesd, "xxx")(
        query=query,
        ...
        actual_seq_lengths=actual_seq_lengths,
        some_flag=some_flag,
    )


# 2. fake op（Meta 设备 shape 推导，必须在 wrapper 之后）
@register_ops.register_mindie_fake_op("xxx")
def xxx_fake(
    query: torch.Tensor,
    ...
    some_flag: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(query)
    # 按输出 shape 规则构造，不做实际计算
    lse = torch.empty([query.shape[0], query.shape[1], query.shape[2], 1],
                      device=query.device, dtype=torch.float32)
    return out, lse
```

**⚠️ 容易遗漏**：
- `@register_mindie_fake_op("xxx")` 的参数签名必须与 `register_ops.cpp` 里 schema 的参数名**完全一致**（包括默认值类型），否则 PyTorch dispatcher 匹配失败。
- fake op 里只能用 `torch.empty`/`torch.zeros`/`torch.empty_like`，不能做任何真实计算（meta 设备上没有数据）。
- `device=query.device` 必须写，否则输出 Tensor 的设备与输入不一致。
- `@register_mindie_fake_op` 内部会检查 C++ 算子是否已注册，**必须在 `load_library` 之后执行**，所以整个 `_custom_ops.py` 的 import 必须在 `load_library` 之后进行（见步骤⑦）。

---

### 步骤 ⑥ 业务调用层（可选）

如果需要封装成更高层的接口（如 `sparse_flash_attn_rf_v3.py`）：

```python
from .. import _custom_ops as ops

def xxx_attention(query, key, value, mask, ...):
    # 处理不支持的布局（如 BSND → BNSD）
    if input_layout == "BSND":
        query = query.permute(0, 2, 1, 3).contiguous()
        ...
        op_layout = "BNSD"

    out, _ = ops.xxx(query, key, value, mask, ...)

    if input_layout == "BSND":
        out = out.permute(0, 2, 1, 3).contiguous()
    return out
```

---

### 步骤 ⑦ 新建测试文件 `tests/plugin/test_xxx.py`

```python
import os
import sys
import math
import unittest
import torch

# ① 将项目根目录加入 sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ② 加载 C++ so（先于 import _custom_ops）
if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
    torch.ops.load_library("../mindiesd/plugin/libPTAExtensionOPS.so")

# ③ 触发 fake op（Meta backend）注册（必须在 load_library 之后）
import mindiesd.layers._custom_ops  # noqa: F401


class TestXxxFakeOp(unittest.TestCase):
    """不需要 NPU，验证 fake op 输出 shape/dtype。必须用 device='meta' tensor。"""
    META = torch.device("meta")

    def test_fake_basic_shape(self):
        B, N, S, D = 1, 4, 128, 128
        q = torch.empty(B, N, S, D, dtype=torch.float16, device=self.META)
        k = torch.empty(B, N, S, D, dtype=torch.float16, device=self.META)
        v = torch.empty(B, N, S, D, dtype=torch.float16, device=self.META)
        out, lse = torch.ops.mindiesd.xxx(query=q, key=k, value=v, ...)
        self.assertEqual(tuple(out.shape), (B, N, S, D))
        self.assertEqual(out.dtype, torch.float16)


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU tests when MINDIE_TEST_MODE=CPU"
)
class TestXxxNPU(unittest.TestCase):
    """NPU 冒烟测试：先保证不 crash、shape 正确，再加精度测试。"""

    def setUp(self):
        self.device = torch.device("npu:0")
        torch.npu.set_device(self.device)
        # ⚠️ inner_precise 值取决于自定义库的编译配置，询问算子方确认
        # 自定义库通常: 0=fp32高精度, 1=fp16高性能; 不支持 4
        # CANN 内置库在 950 上: 必须用 4
        self.inner_precise = 1

    def test_smoke_basic(self):
        """最小参数冒烟测试：不 crash，输出 shape 与 query 一致"""
        B, N, S, D = 1, 1, 128, 128
        q = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
        k = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
        v = torch.randn(B, N, S, D, dtype=torch.float16).to(self.device)
        out, lse = torch.ops.mindiesd.xxx(
            query=q, key=k, value=v,
            ...,
            inner_precise=self.inner_precise,
        )
        self.assertEqual(tuple(out.shape), (B, N, S, D))
        self.assertEqual(out.dtype, torch.float16)
```

**⚠️ 容易遗漏**：
- `sys.path` 设置和 `import _custom_ops` 必须出现在测试文件里，不能依赖外部环境。
- `load_library` → `import _custom_ops` 顺序不能颠倒。
- fake op 测试**必须**用 `device='meta'`，用 CPU tensor 会报 `NotImplementedError: CPU backend`。
- NPU 冒烟测试先用最小参数（`B=1, N=1, S=128, D=128`），确认不 crash 后再扩展。
- `inner_precise` 要向算子开发方确认，不同芯片、不同库版本合法值不同（见§四.2）。

---

## 三、核心机制与高频坑

### 3.1 `OptionalIntArrayRef` 必须 `.value_or()` 展开

**背景**：`EXEC_NPU_CMD` 用模板 `ConvertType` 把 C++ 参数转为 aclnn C 类型。
`ConvertType` 有以下重载，**但没有 `c10::OptionalIntArrayRef` 的重载**：

```
at::Tensor                 → AclTensor*
c10::optional<at::Tensor>  → AclTensor*（nullopt → nullptr）
at::IntArrayRef            → AclIntArray*
const char*                → char*
int64_t / double           → 值透传
```

如果直接传 `c10::OptionalIntArrayRef`，编译器会选最接近的重载，把 `OptionalArrayRef` 结构体的内存当作裸指针解释 → **运行时野指针 crash**，且没有编译期报错。

**正确写法**：
```cpp
// ✅
auto actSeqLen = actual_seq_lengths.value_or(at::IntArrayRef{});
EXEC_NPU_CMD<OP>(..., actSeqLen, ...);

// ❌ 直接传 OptionalIntArrayRef → crash
EXEC_NPU_CMD<OP>(..., actual_seq_lengths, ...);
```

---

### 3.2 固定约束参数值向算子方确认

某些参数表面上是数字，但有强约束：

| 参数 | 错误直觉 | 实际值 | 说明 |
|---|---|---|---|
| `blockSize` | 用 `block_shape[0]` | **0** | 算子方确认：不支持 PagedAttention，固定传 0 |
| `preTokens` | 自己推导 | `2147483647` | int32_max，不支持滑窗 |
| `nextTokens` | 自己推导 | `2147483647` | 同上 |
| `maskType` | 根据 mask 是否为空 | **0** | 当前版本固定 0 |

**结论**：接入前把所有"固定约束"参数的值逐一向算子开发方确认，不要凭直觉猜。

---

### 3.3 Optional 输出 Tensor 的处理模式

当输出 Tensor 是可选的（如 `softmaxLseOptional`），正确模式：

```cpp
// ✅ 始终分配 tensor（保证返回值合法），但按 flag 控制传给算子的是否为 nullptr
at::Tensor lse = at_npu::native::empty_with_format(...);
c10::optional<at::Tensor> lseOpt = (flag != 0)
    ? c10::optional<at::Tensor>(lse)
    : c10::nullopt;

EXEC_NPU_CMD<OP>(..., lseOpt);  // flag=0 → nullptr；flag!=0 → 实际地址
return {out, lse};  // 返回 lse（始终有效，flag=0 时内容未定义但 tensor 本身合法）

// ❌ flag=0 时不分配 → 返回 undefined tensor，Python 侧访问 .shape 崩溃
at::Tensor lse;  // 默认构造 = undefined
if (flag != 0) { lse = ...; }
return {out, lse};  // lse 可能是 undefined tensor
```

---

### 3.4 Symbol 冲突：同名 aclnn 函数有两个版本

**场景**：自定义 `libcust_opapi.so` 和 CANN 内置 `libopapi.so` 都导出同名函数 `aclnnXxx`，但**参数顺序或 tiling 结构不同**。

`GetOpApiFuncAddr` 搜索优先级（高 → 低）：
```
libcust_opapi.so（自定义）→ vendors/ → libopapi.so（CANN 内置）
```

加载自定义库后，**所有用 `EXEC_NPU_CMD` 调用同名函数的旧算子都会被重定向到自定义版本**，参数错位或 tiling 结构不匹配 → crash。
典型症状：原来运行正常的旧算子在加载新自定义库后报 `BinaryGetFunctionByEntry failed`。

**两种解决方案，按实际情况选择：**

#### 方案 A：禁用冲突的旧算子（已验证有效）

如果旧算子已无业务使用，最简单的做法是将其从编译产物中彻底移除：

```cmake
# csrc/CMakeLists.txt
# ./plugin/block_sparse_attention.cpp  # disabled: tiling struct conflicts with block_sparse_attention
./plugin/block_sparse_attention.cpp  # 新算子
```

```cpp
// csrc/plugin/register_ops.cpp
// m.def("block_sparse_attention ...");         // 注释掉 schema
// m.impl("block_sparse_attention", &impl);     // 注释掉 impl
m.def("block_sparse_attention ...");        // 新算子正常注册
m.impl("block_sparse_attention", &impl);
```

**优点**：彻底消除冲突，无运行时开销，实现最简单。
**缺点**：旧算子功能（如 `ada_bsa` 路径）不再可用；如需恢复，需解决结构体冲突后重新启用。

#### 方案 B：旧算子改用 `EXEC_BUILTIN_NPU_CMD` 固定走 CANN 内置库

当旧算子仍有业务需要时，改用 `EXEC_BUILTIN_NPU_CMD`（直接从 `libopapi.so` 取地址，跳过自定义库搜索链），新算子继续用 `EXEC_NPU_CMD` 走自定义库：

```cpp
// 旧算子：固定走 CANN 内置库，不受自定义库影响
EXEC_BUILTIN_NPU_CMD<"aclnnXxx">(...);

// 新算子：走自定义库
EXEC_NPU_CMD<"aclnnXxx">(...);
```

`EXEC_BUILTIN_NPU_CMD` 在 `pytorch_npu_helper.h` 中定义，与 `EXEC_NPU_CMD` 实现相同，只是把 `GetOpApiFuncAddr` 替换为 `GetFuncFromDefaultLib`。

**判断是否有冲突**：
- 询问算子开发方：新算子的 aclnn symbol 名是否与现有某个已接入算子相同？
- 检查 `register_ops.cpp`：现有算子里是否有用同一个 aclnn symbol 名的？

---

### 3.5 `inner_precise` 约束因库版本而异

`inner_precise` 被编码进 tiling key（如 `9050000050000003`）；key 对应的 binary 不存在时报 `BinaryGetFunctionByEntry failed`。同一个 aclnn symbol，不同芯片和不同库版本对 `inner_precise` 的约束可能不同：

| 芯片 | 常见合法值 | 说明 |
|---|---|---|
| 950PR / 950DT | **4**（混合精度） | 已在 `block_sparse_attention` 上验证；其他值在特定库版本下可能无对应 binary |
| 其他昇腾芯片 | 0 或 1 | 0=fp32 高精度，1=fp16 高性能 |

**重要**：`inner_precise` 的合法值**取决于算子开发方编译时打包了哪些 tiling binary**，与使用 CANN 内置库还是自定义库无直接关联。接入前向算子开发方索要当前版本支持的 `{芯片, dtype, inner_precise}` 组合矩阵，不要凭直觉猜测。

**本项目实践**：`block_sparse_attention` 在 Ascend 950 上使用 `inner_precise=4`，通过环境变量 `BSA_INNER_PRECISE`（默认 4）控制，非 950 芯片时需根据实际情况调整。

---

### 3.6 fake op 注册的完整依赖链

```
load_library("libPTAExtensionOPS.so")
  → TORCH_LIBRARY(mindiesd, m) { m.def("xxx ...") }
  → C++ schema 注册到 dispatcher ✓

import mindiesd.layers._custom_ops
  → from . import register_ops
      → if is_npu_available(): _load_mindie_ops_library()  ← 会再次 load（幂等）
  → @register_mindie_fake_op("xxx")
      → check_mindie_operator_exists("xxx")  ← 依赖上面的 schema 注册
      → register_fake("mindiesd::xxx")       ← 注册到 Meta dispatch key ✓
```

**两个依赖**：
1. `load_library` 必须先于 `import _custom_ops`（否则 `check_mindie_operator_exists` 返回 False，抛 RuntimeError）
2. `import _custom_ops` 必须发生（否则 Meta kernel 永远不注册，`device='meta'` 的调用报 `NotImplementedError`）

---

## 四、接入检查清单

每次接入新算子，按以下清单逐项确认：

### C++ 层

- [ ] `xxx.h`：头文件存在，include guard 正确，函数签名与 `.cpp` 一致
- [ ] `xxx.cpp`：`OptionalIntArrayRef` 已 `.value_or()` 展开
- [ ] `xxx.cpp`：`const char*` 布局参数已从 `std::string::c_str()` 取
- [ ] `xxx.cpp`：Optional 输出 Tensor 已按"始终分配+条件传入"模式处理
- [ ] `xxx.cpp`：固定约束参数值已向算子方确认（不凭直觉）
- [ ] `xxx.cpp`：参数顺序与 `aclnnXxxGetWorkspaceSize` 文档完全一致
- [ ] `xxx.cpp`：是否有 symbol 冲突？如有，选择方案 A（禁用旧算子）或方案 B（旧算子改用 `EXEC_BUILTIN_NPU_CMD`）
- [ ] `register_ops.cpp`：`#include "xxx.h"` 已添加
- [ ] `register_ops.cpp`：`TORCH_LIBRARY` 块里 `m.def(...)` 已添加
- [ ] `register_ops.cpp`：`TORCH_LIBRARY_IMPL` 块里 `m.impl(...)` 已添加
- [ ] **`CMakeLists.txt`：`./plugin/xxx.cpp` 已加入 `add_library`**（最容易漏）
- [ ] 重新 cmake + make，无链接错误

### Python 层

- [ ] `_custom_ops.py`：Python wrapper 函数已添加
- [ ] `_custom_ops.py`：`@register_mindie_fake_op("xxx")` fake 实现已添加
- [ ] fake 实现参数签名与 schema 参数名完全一致
- [ ] fake 实现里输出 shape 逻辑正确（按 layout/flag 分支）
- [ ] fake 实现里 `device=query.device` 已填

### 测试层

- [ ] 测试文件开头有 `sys.path.insert(0, _ROOT)` 和 `import mindiesd.layers._custom_ops`
- [ ] `import _custom_ops` 在 `load_library` 之后
- [ ] fake op 测试使用 `device='meta'` tensor，不用 CPU tensor
- [ ] NPU 冒烟测试 `inner_precise` 值已向算子方确认（不传 4 给自定义库）
- [ ] NPU 冒烟测试先用最小参数（`B=N=1, S=D=128`）

---

## 五、文件变更速查

每次接入一个新算子，需要修改/新建的文件：

| 文件 | 操作 | 说明 |
|---|---|---|
| `csrc/plugin/xxx.h` | **新建** | 函数声明 |
| `csrc/plugin/xxx.cpp` | **新建** | EXEC_NPU_CMD 封装 |
| `csrc/plugin/register_ops.cpp` | **修改** | +include, +m.def, +m.impl |
| `csrc/CMakeLists.txt` | **修改** | +xxx.cpp 到 add_library |
| `mindiesd/layers/_custom_ops.py` | **修改** | +wrapper, +@register_mindie_fake_op |
| `tests/plugin/test_xxx.py` | **新建** | fake op 测试 + NPU 冒烟测试 |
| 业务调用文件（可选） | 新建/修改 | 高层封装，如 sparse_flash_attn_rf_v3.py |

共 **4 个文件必须修改，2~3 个文件新建**。任意一处遗漏都会导致：
- 漏 `CMakeLists.txt` → 链接报 `undefined reference`
- 漏 `register_ops.cpp` m.def → 运行时 dispatch 失败
- 漏 `register_ops.cpp` m.impl → 运行时找不到实现
- 漏 `_custom_ops.py` fake → `device='meta'` 调用报 `NotImplementedError`
- 漏 `import _custom_ops` 在测试里 → 同上
