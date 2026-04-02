---
name: state
description: |
  Quantum state representation and manipulation module using PyTorch backend.
  Provides comprehensive tools for creating, measuring, and analyzing quantum states
  including normalization, inner products, tensor products, and measurement operations.
keywords:
  - quantum state
  - state vector
  - measurement
  - probability distribution
  - state collapse
  - tensor product
  - expectation value
---

# State 技能指南

## 一、算法介绍

### 1.1 基本概念

**State（量子态）** 是量子计算中的核心概念，表示一个量子系统的完整描述。State 类使用向量表示法在计算基下表达量子态，并通过 PyTorch 张量进行高效计算。

### 1.2 数学基础

#### 1.2.1 状态向量表示

对于 $n$ 个量子比特的系统，量子态可表示为：

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

其中：
- $\alpha_i \in \mathbb{C}$ 是第 $i$ 个基态的振幅
- $|i\rangle$ 是计算基态 $|i_1 i_2 \cdots i_n\rangle$
- 归一化条件：$\sum_{i=0}^{2^n-1} |\alpha_i|^2 = 1$

在 State 类中，此向量以长度为 $2^n$ 的一维 PyTorch 张量存储。

#### 1.2.2 量子操作的数学表示

**测量在计算基下**：
$$P(i) = |\alpha_i|^2$$

**内积**：
$$\langle \psi | \phi \rangle = \sum_{i=0}^{2^n-1} \alpha_i^* \beta_i$$

**期望值**：
$$\langle O \rangle = \langle \psi | O | \psi \rangle$$

**张量积**：
$$|\psi_1\rangle \otimes |\psi_2\rangle = \sum_{i,j} \alpha_i \beta_j |i\rangle \otimes |j\rangle$$

### 1.3 处理流程

```
创建 State
    ↓
初始化数据（比特数或向量）
    ↓
自动归一化
    ↓
查询：概率、期望值
    或
操作：内积、张量积
    或
测量：概率分布、采样、状态坍缩
    ↓
结果输出
```

### 1.4 核心特性

**灵活的初始化**
- 用比特数创建基态 $|0\rangle^{\otimes n}$
- 用数据向量初始化任意态

**自动数据处理**
- 支持多种输入格式（Tensor、List、Numpy）
- 自动维度检查和扁平化
- 自动在创建时归一化

**精确的量子操作**
- 支持复数振幅
- 端序灵活性（大端/小端）
- 数值稳定性保证

**高效的 PyTorch 实现**
- GPU 加速支持
- 向量化操作
- 内存高效

### 1.5 Endian（端序）约定

```
Little Endian（小端，Qiskit 默认）:
- 比特 0 是最右边（最低有效位）
- 张量表示：[q_{n-1}, q_{n-2}, ..., q_0]
- 二进制字符串 "101" 表示：q_0=1, q_1=0, q_2=1

Big Endian（大端）:
- 比特 0 是最左边
- 张量表示：[q_0, q_1, ..., q_{n-1}]
- 二进制字符串 "101" 表示：q_0=1, q_1=0, q_2=1
```

---

## 二、使用方法步骤

### 2.1 基础使用流程

#### 第一步：导入模块
```python
from core.State import State
import torch
import numpy as np
```

#### 第二步：创建状态
```python
# 方法 1: 用比特数（创建 |0⟩^⊗n）
state = State(3)  # 3-比特基态 |000⟩

# 方法 2: 用向量数据
state = State([1/np.sqrt(2), 1/np.sqrt(2)])

# 方法 3: 用 PyTorch 张量
data = torch.tensor([1, 1], dtype=torch.complex128)
state = State(data)  # 自动归一化
```

#### 第三步：查询状态信息
```python
# 获取基本信息
n_qubits = state.num_qubits
dimension = state.dim
prob_dist = state.probabilities()

# 获取数据
data = state.data
dtype = state.dtype
```

#### 第四步：执行量子操作
```python
# 查看概率分布
probs = state.probabilities_dict([0, 1])

# 计算期望值
matrix = torch.eye(4, dtype=torch.complex128)
exp_val = state.expectation_value(matrix)

# 测量（坍缩）
result = state.measure([0, 1])

# 采样
samples = state.sample_counts(shots=1024)
```

### 2.2 详细使用示例

#### 例子 1：创建和检查基本态

```python
import numpy as np
from core.State import State

# 创建 |0⟩ 态
state_0 = State(1)
print(state_0)
# 输出：State: qubits: 1, data: [1+0j, 0+0j]

# 创建 |1⟩ 态
state_1_data = [0, 1]
state_1 = State(state_1_data)
print(state_1)
# 输出：State: qubits: 1, data: [0+0j, 1+0j]

# 验证基本信息
print(f"比特数: {state_0.num_qubits}")    # 1
print(f"希尔伯特空间维度: {state_0.dim}") # 2
print(f"数据类型: {state_0.dtype}")       # torch.complex128
```

#### 例子 2：创建叠加态

```python
import numpy as np
from core.State import State

# 创建单比特叠加态 |+⟩ = (1/√2)(|0⟩ + |1⟩)
plus_state = State([1/np.sqrt(2), 1/np.sqrt(2)])
print(f"+ 态数据: {plus_state.data}")

# 创建 |-⟩ = (1/√2)(|0⟩ - |1⟩)
minus_state = State([1/np.sqrt(2), -1/np.sqrt(2)])
print(f"- 态数据: {minus_state.data}")

# 带相位的态 |i⟩ = (1/√2)(|0⟩ + i|1⟩)
i_state = State([1/np.sqrt(2), 1j/np.sqrt(2)])
print(f"i 态数据: {i_state.data}")

# 检查概率分布都是 50%-50%
print(f"概率分布: {plus_state.probabilities()}")
```

#### 例子 3：多比特基态

```python
from core.State import State
import torch
import numpy as np

# 创建 3 比特 |000⟩ 态
state = State(3)
print(f"比特数: {state.num_qubits}")      # 3
print(f"维度: {state.dim}")              # 8
print(f"数据: {state.data}")
# 只有第一个分量为 1，其他为 0

# 创建特定的 3 比特态 |101⟩
# 二进制 101 = 十进制 5
state_101_data = [0] * 8
state_101_data[5] = 1  # 位置 5 对应 |101⟩（小端序）
state_101 = State(state_101_data)
print(f"|101⟩ 态: {state_101.data}")
```

#### 例子 4：概率分布查询（非坍缩）

```python
from core.State import State
import numpy as np

# 创建贝尔态 |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
bell_plus_data = [
    1/np.sqrt(2), 0, 0, 1/np.sqrt(2)
]
state = State(bell_plus_data)

# 查询全部比特的概率
full_probs = state.probabilities_dict([0, 1])
print("完整概率分布:")
for bits, prob in full_probs.items():
    print(f"  {bits}: {prob:.4f}")

# 查询部分比特的概率（partial trace）
partial_0 = state.probabilities_dict([0])
print("\n仅比特 0 的概率:")
for bits, prob in partial_0.items():
    print(f"  {bits}: {prob:.4f}")
```

#### 例子 5：测量和状态坍缩

```python
from core.State import State
import numpy as np

# 创建均匀叠加态
uniform_data = np.ones(4) / 2  # (1/2)(|00⟩ + |01⟩ + |10⟩ + |11⟩)
state = State(uniform_data)

print(f"初始态概率: {state.probabilities()}")

# 测量第 0 个比特
result = state.measure([0])
print(f"测量结果: {result}")
print(f"测量后概率: {state.probabilities()}")
# 状态已坍缩到两个可能性之一

# 再测量第 1 个比特（状态已改变）
result2 = state.measure([1])
print(f"第二次测量结果: {result2}")
print(f"最终态: {state.data}")
# 现在状态是确定的基态
```

#### 例子 6：采样和统计

```python
from core.State import State
import numpy as np

# 创建不均匀分布
# P(|0⟩) = 0.25, P(|1⟩) = 0.75
weighted_state = State([0.5, np.sqrt(0.75)])

# 采样 10000 次
samples = weighted_state.sample_counts(shots=10000)
print("采样结果:")
for state_str, count in sorted(samples.items()):
    print(f"  {state_str}: {count}/10000 ({count/100:.1f}%)")

# 理论值
print("\n理论概率:")
print(f"  |0⟩: 25%")
print(f"  |1⟩: 75%")
```

### 2.3 与量子电路的集成

#### 集成方式 1：电路后获取态

```python
from core.GateSequence import GateSequence
from core.State import State
from core.register import Register

# 创建电路
qreg = Register("q", 2)
circuit = GateSequence(qreg)

# 构建电路
circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])

# 执行获取状态向量
result_vector = circuit.execute()

# 创建 State 对象
state = State(result_vector)
print(f"电路结果态: {state}")
print(f"概率分布: {state.probabilities()}")
```

#### 集成方式 2：分析电路结果

```python
from core.GateSequence import GateSequence
from core.State import State
from core.register import Register

# 创建 Bell 电路
qreg = Register("q", 2)
circuit = GateSequence(qreg)
circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])

# 执行
result_vector = circuit.execute()
state = State(result_vector)

# 分析概率分布
probs = state.calculate_state([0, 1])
print("完整分析:")
for bits, info in probs.items():
    print(f"  {bits}: 概率={info['prob']:.4f}, 整数值={info['int']}")

# 进行测量采样
samples = state.sample_counts(shots=1000)
print(f"\n1000 次采样: {samples}")
```

### 2.4 高级场景

#### 场景 1：内积计算

```python
from core.State import State
import numpy as np

# 创建两个态
state1 = State([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩
state2 = State([1/np.sqrt(2), -1/np.sqrt(2)])  # |-⟩

# 计算内积 ⟨+|-⟩
inner = state1.inner_product(state2)
print(f"⟨+|-⟩ = {inner}")
# 理论值：0

# 计算 ⟨+|+⟩（应该是 1）
self_inner = state1.inner_product(state1)
print(f"⟨+|+⟩ = {self_inner}")
```

#### 场景 2：期望值计算

```python
from core.State import State
import torch
import numpy as np

# 创建一个态
state = State([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩

# 定义 Pauli-Z 矩阵
pauli_z = torch.tensor([
    [1, 0],
    [0, -1]
], dtype=torch.complex128)

# 计算 ⟨Z⟩ 期望值
exp_z = state.expectation_value(pauli_z)
print(f"⟨Z⟩ = {exp_z}")
# 对于 |+⟩，⟨Z⟩ = 0

# 定义投影算子 P_0 = |0⟩⟨0|
proj_0 = torch.tensor([
    [1, 0],
    [0, 0]
], dtype=torch.complex128)

# 计算测量到 |0⟩ 的概率
prob_0 = state.expectation_value(proj_0).real
print(f"P(|0⟩) = {prob_0}")
```

#### 场景 3：张量积

```python
from core.State import State
import numpy as np

# 创建两个单比特态
state_0 = State(1)  # |0⟩
state_plus = State([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩

# 计算张量积 |0⟩ ⊗ |+⟩
combined = state_0.tensor(state_plus)

print(f"结果类型: {type(combined)}")
print(f"比特数: {combined.num_qubits}")  # 2
print(f"数据: {combined.data}")
# 应该是 (1/√2)|00⟩ + (1/√2)|01⟩
```

#### 场景 4：部分测量后继续操作

```python
from core.State import State
import numpy as np

# 创建贝尔态
bell_data = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
state = State(bell_data)

print("初始贝尔态:")
print(f"  完整概率: {state.probabilities_dict([0, 1])}")

# 测量第 0 个比特
result_0 = state.measure([0])
print(f"\n测量比特 0: {result_0}")

# 查看测量后的态
print(f"  坍缺后的全态概率: {state.probabilities()}")

# 部分态概率
partial = state.calculate_state([1])
print(f"  比特 1 的分布: {partial}")

# 继续测量比特 1
result_1 = state.measure([1])
print(f"\n测量比特 1: {result_1}")
print(f"  最终态: {state.data}")
```

---

## 三、关键 API 参考

### 3.1 构造函数

```python
State(data, num_qubits=None)
```

**参数：**
- `data: int | array-like` - 初始化数据
  - 如果是 `int`：创建 n 比特基态 $|0\rangle^{\otimes n}$
  - 如果是数组：状态向量数据（自动拉平为 1D）
- `num_qubits: int` - 可选，用于验证（通常自动推导）

**返回值：**
- 初始化且归一化的 State 对象

**异常：**
- `ValueError` - 如果维度不是 $2^n$ 或归一化失败

### 3.2 属性访问

```python
# 只读属性
state.num_qubits    # 比特数
state.dim           # 希尔伯特空间维度
state.data          # 状态向量（PyTorch张量）
state.dtype         # 数据类型（torch.complex128）

# 方法
state.norm()        # 計算向量范数
```

### 3.3 核心方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `normalize()` | - | self | 归一化状态 |
| `inner_product(other)` | other: State | complex | 计算内积 $\langle\psi\|\phi\rangle$ |
| `tensor(other)` | other: State | State | 张量积 $\|\psi\rangle\otimes\|\phi\rangle$ |
| `expectation_value(matrix)` | matrix: Tensor | complex | 期望值 $\langle\psi\|O\|\psi\rangle$ |
| `probabilities()` | - | Tensor | 所有基态的概率分布 |
| `probabilities_dict(targets)` | targets: list | dict | 指定比特的概率分布 |
| `measure(targets)` | targets: list | str | 测量并返回结果字符串 |
| `sample_counts(shots)` | shots: int | dict | 采样指定次数的结果统计 |
| `calculate_state(targets)` | targets: list | dict | 详细的状态分析 |

### 3.4 方法详解

#### probabilities_dict

```python
probabilities_dict(target_indices, endian='little', threshold=1e-9) -> dict
```

**参数：**
- `target_indices` - 要分析的比特索引
- `endian` - 字节序（'little' 或 'big'）
- `threshold` - 概率阈值（低于此值忽略）

**返回：**
```python
{
    '00': 0.5,
    '11': 0.5
}
```

#### measure

```python
measure(target_indices, endian='little') -> str
```

**参数：**
- `target_indices` - 要测量的比特索引
- `endian` - 字节序

**返回：**
- 测量结果的二进制字符串（如 '101'）

**副作用：**
- 修改 State 对象（状态坍缩）

#### calculate_state

```python
calculate_state(target_indices, endian='little', threshold=1e-5) -> dict
```

**返回：**
```python
{
    '00': {'prob': 0.5, 'int': 0},
    '11': {'prob': 0.5, 'int': 3}
}
```

---

## 四、数学操作详解

### 4.1 量子测量的数学模型

单次测量流程：

1. **概率计算**：$P(i) = |\alpha_i|^2$
2. **采样选择**：按概率从 $\{0, 1, ..., 2^n-1\}$ 采样
3. **状态坍缩**：$|\psi\rangle \rightarrow \frac{|i\rangle}{\sqrt{P(i)}}$

### 4.2 部分比特的测量与 Partial Trace

对子集比特的概率分布，通过 partial trace 计算：

$$\rho_A = \text{Tr}_B(\rho)$$

其对角元素即为测量 A 中比特的概率。

### 4.3 期望值与投影算子

测量到状态 $|i\rangle$ 的概率：

$$P(i) = \langle\psi|P_i|\psi\rangle$$

其中 $P_i = |i\rangle\langle i|$ 是投影算子。

---

## 五、常见模式

### 模式 1：创建标准基态

```python
def create_basis_state(n_qubits, target_state_int):
    """创建计算基态 |target_state⟩"""
    data = [0] * (2**n_qubits)
    data[target_state_int] = 1
    return State(data)

# 使用
state_5 = create_basis_state(3, 5)  # |101⟩
```

### 模式 2：创建均匀叠加

```python
def create_uniform_superposition(n_qubits):
    """创建均匀叠加态"""
    amplitude = 1 / np.sqrt(2**n_qubits)
    data = [amplitude] * (2**n_qubits)
    return State(data)

# 使用
uniform = create_uniform_superposition(3)
```

### 模式 3：分析量子电路结果

```python
def analyze_circuit_output(circuit, num_qubits, shots=1000):
    """分析电路输出"""
    result_vector = circuit.execute()
    state = State(result_vector)
    
    # 获取统计
    stats = state.sample_counts(shots=shots)
    
    # 获取理论概率
    theory = state.probabilities_dict(list(range(num_qubits)))
    
    return stats, theory
```

---

## 六、错误处理指南

### 6.1 常见错误

**错误 1：维度不是 2 的幂**
```python
# 错误！
try:
    state = State([1, 1, 1])  # 3 ≠ 2^n
except ValueError as e:
    print(f"错误: {e}")  # 输入数据长度 3 不是 2 的幂

# 正确
state = State([1, 1, 1, 1]) / 2  # 4 = 2^2
```

**错误 2：未归一化的向量**
```python
# 可以但会自动修复
state = State([1, 1])  # 未归一化
print(state.data)  # 自动归一化为 [1/√2, 1/√2]
```

**错误 3：维度不匹配的操作**
```python
state1 = State(1)   # 1 比特
state2 = State(2)   # 2 比特

# 错误！
try:
    inner = state1.inner_product(state2)
except ValueError as e:
    print(f"错误: {e}")  # 维度不匹配

# 正确：使用张量积
combined = state1.tensor(state2)
```

### 6.2 验证函数

```python
def validate_state(state):
    """验证 State 对象的有效性"""
    # 检查类型
    if not isinstance(state, State):
        raise TypeError("Expected State object")
    
    # 检查规范化
    norm = state.norm()
    if not np.isclose(norm, 1.0):
        print(f"警告：状态未正确规范化（范数={norm}）")
    
    # 检查维度
    if state.dim != 2**state.num_qubits:
        raise ValueError("维度和比特数不一致")
    
    return True
```

---

## 七、性能优化

### 7.1 使用 GPU 加速

```python
import torch
from core.State import State

# 如果有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建状态并移到 GPU（需要修改 State 类）
state_data = torch.tensor([1, 1], dtype=torch.complex128, device=device)
state = State(state_data)
```

### 7.2 批量采样的效率

```python
# 对大量比特进行采样
n_qubits = 20
n_shots = 100000

state = State(n_qubits)
samples = state.sample_counts(shots=n_shots)

# 比单次测量 n_shots 次更高效
```

### 7.3 复用 State 对象

```python
# 好的做法：创建一次，多次分析
state = State([1/np.sqrt(2), 1/np.sqrt(2)])

# 多次查询（不修改状态）
state.probabilities_dict([0])
state.calculate_state([0])
state.sample_counts(shots=100)

# 不要重复创建
# for i in range(100):
#     s = State([1/np.sqrt(2), 1/np.sqrt(2)])  # X
```

---

## 八、完整工作流示例

```python
import numpy as np
from core.State import State
from core.GateSequence import GateSequence
from core.register import Register

def complete_quantum_workflow():
    """完整的量子计算工作流"""
    
    # 步骤 1: 创建量子电路
    print("步骤 1: 构建量子电路")
    qreg = Register("q", 2)
    circuit = GateSequence(qreg)
    
    # 步骤 2: 应用量子门构建 Bell 态
    print("步骤 2: 应用量子门")
    circuit.h(qreg[0])
    circuit.cnot(qreg[0], qreg[1])
    
    # 步骤 3: 执行电路获取态向量
    print("步骤 3: 执行电路")
    result_vector = circuit.execute()
    
    # 步骤 4: 创建 State 对象
    print("步骤 4: 创建量子态对象")
    state = State(result_vector)
    
    # 步骤 5: 分析态
    print("步骤 5: 分析量子态")
    print(f"  比特数: {state.num_qubits}")
    print(f"  完整概率分布: {state.probabilities_dict([0, 1])}")
    
    # 步骤 6: 验证保理度
    print("步骤 6: 验证结果")
    fidelity = np.abs(state.data[0])**2 + np.abs(state.data[3])**2
    print(f"  贝尔态保真度: {fidelity:.4f}")
    
    # 步骤 7: 采样
    print("步骤 7: 量子采样")
    samples = state.sample_counts(shots=1000)
    print(f"  采样结果: {samples}")
    
    # 步骤 8: 测量（修改状态）
    print("步骤 8: 执行测量")
    result = state.measure([0, 1])
    print(f"  测量结果: {result}")
    print(f"  最终态: {state.data}")

# 执行
complete_quantum_workflow()
```

---

## 九、总结检查清单

使用 State 类时，请确保：

- [ ] 已正确导入 State 类
- [ ] 数据维度是 $2^n$ 或使用比特数创建
- [ ] 理解了状态的自动归一化
- [ ] 知道测量会修改状态（坍缺）
- [ ] 理解了 endian 参数的作用
- [ ] 使用正确的索引方式访问比特
- [ ] 区分了概率查询（非坍缺）和测量（坍缺）
- [ ] 正确处理了复数振幅
- [ ] 在必要时检验状态的规范化性
- [ ] 使用了适当的数值阈值处理浮点误差

