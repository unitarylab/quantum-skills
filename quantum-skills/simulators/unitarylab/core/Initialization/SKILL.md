---
name: initialization
description: |
  Quantum state initialization and preparation module for preparing arbitrary quantum states on qubits.
  Automatically constructs optimized quantum circuits that initialize qubits to target normalized state vectors
  using recursive decomposition techniques.
keywords:
  - state preparation
  - quantum initialization
  - state vector
  - quantum circuit synthesis
  - controlled operations
  - amplitude encoding
---

# Initialization 技能指南

## 一、算法介绍

### 1.1 基本概念

**State Preparation（量子态准备）** 是将量子系统初始化为特定目标态的过程。给定一个归一化的量子态向量 $|\psi\rangle$，Initialization 模块会自动构建一个量子电路，该电路通过一系列量子门的操作将初始的基态 $|00...0\rangle$ 演化为目标态。

### 1.2 数学基础

#### 1.2.1 量子态表示

一个 n 比特量子系统的状态向量表示为：

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

其中：
- $\alpha_i \in \mathbb{C}$ 是振幅（amplitude）
- $|i\rangle$ 是计算基态
- 归一化条件：$\sum_{i=0}^{2^n-1} |\alpha_i|^2 = 1$

#### 1.2.2 相位和振幅

每个振幅可以表示为极坐标形式：

$$\alpha_i = r_i e^{i\phi_i}$$

其中：
- $r_i = |\alpha_i|$ 是振幅大小（非负实数）
- $\phi_i = \arg(\alpha_i)$ 是相位

### 1.3 递归分解算法

Initialization 使用递归分解方法来构建态准备电路。

#### 1.3.1 算法流程

```
输入: 目标态向量 |v⟩
    ↓
检查归一化条件
    ↓
判断是否为单比特情景
    ├─ 是 → 直接应用单比特门
    └─ 否 → 向量分解
        ↓
    将 |v⟩ 分解为两部分
    v1 = v[0:2^(n-1)]
    v2 = v[2^(n-1):]
        ↓
    计算概率
    p1 = ||v1||²
    p2 = ||v2||²
        ↓
    应用 RY 门调整最后一个比特的概率
        ↓
    递归准备 v1 和 v2
    使用受控门限制
        ↓
    返回完整电路
```

#### 1.3.2 单比特初始化

对于单个量子比特，目标态为：

$$|\psi\rangle = \alpha_0 |0\rangle + \alpha_1 |1\rangle$$

初始化步骤：

1. **提取相位**：
   - $\phi_0 = \arg(\alpha_0)$
   - $\phi_1 = \arg(\alpha_1)$

2. **应用全局相位**：
   $$|\psi'\rangle = e^{-i\phi_0} |\psi\rangle$$

3. **计算旋转角**：
   $$\theta = \arccos(|\alpha_0|)$$

4. **应用 RY 旋转**：
   $$RY(2\theta) |0\rangle \rightarrow |\psi''\rangle$$

5. **应用相位门**：
   $$P(\phi_1 - \phi_0) |\psi''\rangle \rightarrow |\psi\rangle$$

```python
# 单比特初始化示意
初始态: |0⟩
    ↓ [应用全局相位]
初始态 (相位调整)
    ↓ [应用 RY(2θ)]
叠加态
    ↓ [应用相位门 P(Δφ)]
目标态: α₀|0⟩ + α₁|1⟩
```

#### 1.3.3 多比特递归分解

对于 n 比特情景，在第 (n-1) 个比特上应用 RY 旋转：

1. **概率分布计算**：
   - $p_1 = ||v_1||^2$ （前半部分比特在 0 的概率）
   - $p_2 = ||v_2||^2$ （前半部分比特在 1 的概率）

2. **旋转角计算**：
   $$\theta = \arccos(\sqrt{p_1})$$

3. **受控态准备**：
   - 当第 (n-1) 比特为 0 时，准备归一化的 $v_1/||v_1||$
   - 当第 (n-1) 比特为 1 时，准备归一化的 $v_2/||v_2||$

```
n 比特分解示意:
┌──────────────────────────────────┐
│  目标态 |v⟩ (2ⁿ 个振幅)           │
└──────────────────────────────────┘
            ↓ [分割]
    ┌───────────────────┐
    │   v1 (前2^(n-1))  │  v2 (后2^(n-1))
    └───────────────────┘
         ↓ [RY 旋转]
    概率重新分配
         ↓ [受控准备]
    ├─ 控制=0: 递归准备 v1
    └─ 控制=1: 递归准备 v2
```

### 1.4 复杂性分析

| 指标 | 复杂度 | 说明 |
|------|---------|------|
| 电路深度 | $O(n^2)$ | 递归分解深度 |
| 门数 | $O(2^n)$ | 指数级门操作 |
| 参数参与度 | $O(2^n)$ | 需要 $2^n$ 个振幅 |
| 时间复杂度 | $O(2^n)$ | 计算电路所需时间 |

---

## 二、使用方法步骤

### 2.1 基本使用流程

#### 第一步：导入模块
```python
from core.Initialization import state_preparation
from core.GateSequence import GateSequence
import numpy as np
```

#### 第二步：定义目标态向量
```python
# 目标态必须是归一化的
# 示例：单比特基态
state_0 = np.array([1, 0])
state_1 = np.array([0, 1])

# 示例：单比特叠加态
superposition = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# 示例：两比特 Bell 态
bell_plus = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
```

#### 第三步：调用 state_preparation
```python
# 生成态准备电路
circuit = state_preparation(superposition, backend='torch')
```

#### 第四步：验证和使用电路
```python
# 执行电路获取量子态
result_state = circuit.execute()

# 绘制电路
circuit.draw(title="State Preparation Circuit")

# 打印电路信息
print(f"电路名称: {circuit.name}")
print(f"量子比特数: {circuit.get_num_qubits()}")
```

### 2.2 详细使用示例

#### 例子 1：准备单比特叠加态

```python
import numpy as np
from core.Initialization import state_preparation

# 目标态: |+⟩ = (1/√2)(|0⟩ + |1⟩)
target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# 生成电路
circuit = state_preparation(target_state)

# 执行
result = circuit.execute()
print(f"目标态: {target_state}")
print(f"执行结果: {result}")

# 绘制
circuit.draw(title="Superposition State")
```

#### 例子 2：准备两比特纠缠态 (Bell 态)

```python
import numpy as np
from core.Initialization import state_preparation

# Bell 态 |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
bell_state = np.array([
    1/np.sqrt(2),  # |00⟩
    0,              # |01⟩
    0,              # |10⟩
    1/np.sqrt(2)   # |11⟩
])

# 生成电路
circuit = state_preparation(bell_state)

# 执行并验证
result = circuit.execute()

print(f"贝尔态: {bell_state}")
print(f"结果: {result}")

# 验证纠缠
print(f"是否准备成功: {np.allclose(result, bell_state)}")
```

#### 例子 3：准备复杂的三比特态

```python
import numpy as np
from core.Initialization import state_preparation

# 定义一个 GHZ 态：|GHZ⟩ = (1/√2)(|000⟩ + |111⟩)
ghz_state = np.zeros(8)
ghz_state[0] = 1/np.sqrt(2)
ghz_state[7] = 1/np.sqrt(2)

# 生成电路
circuit = state_preparation(ghz_state)

# 执行
result = circuit.execute()

print("目标 GHZ 态已成功准备！")
print(f"非零分量: {np.nonzero(result)[0]}")
```

#### 例子 4：带有相位的态准备

```python
import numpy as np
from core.Initialization import state_preparation

# 态中包含相位: |ψ⟩ = (1/√2)|0⟩ + (i/√2)|1⟩
# 其中 i = e^(iπ/2)
target_state = np.array([
    1/np.sqrt(2),           # 实部: 1/√2
    1j/np.sqrt(2)           # 虚部: i/√2
])

# 生成电路
circuit = state_preparation(target_state)

# 执行
result = circuit.execute()

print(f"带相位的目标态: {target_state}")
print(f"执行结果: {result}")
print(f"相位差: {np.angle(result[1]) - np.angle(result[0])}")
```

### 2.3 与 GateSequence 集成

#### 集成方式 1：直接在电路中初始化

```python
from core.GateSequence import GateSequence
from core.Initialization import state_preparation
import numpy as np

# 目标初始态
initial_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# 创建主电路
main_circuit = GateSequence(3)

# 使用 initialize 方法
main_circuit.initialize(initial_state, target=[0, 1])

# 在初始化后的态上应用更多操作
main_circuit.h(2)
main_circuit.cnot(0, 2)

# 执行
result = main_circuit.execute()
```

#### 集成方式 2：组合多个準備電路

```python
from core.GateSequence import GateSequence
from core.Initialization import state_preparation
import numpy as np

# 准备两个不同的态
state1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
state2 = np.array([1/np.sqrt(3), np.sqrt(2/3)])

# 生成对应电路
circuit1 = state_preparation(state1)
circuit2 = state_preparation(state2)

# 创建主电路
main_circuit = GateSequence(4)

# 分别在不同的比特上应用
main_circuit.append(circuit1, [0, 1])
main_circuit.append(circuit2, [2, 3])

# 在两个部分之间添加相互作用
main_circuit.cnot(1, 2)

# 执行
result = main_circuit.execute()
```

### 2.4 高级场景

#### 场景 1：参数化态准备

```python
def create_parameterized_state(theta):
    """创建参数化的单比特态"""
    # 态: |ψ(θ)⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
    state = np.array([np.cos(theta), np.sin(theta)])
    return state_preparation(state)

# 使用不同的参数
theta_values = [0, np.pi/4, np.pi/2]

for theta in theta_values:
    circuit = create_parameterized_state(theta)
    result = circuit.execute()
    print(f"θ = {theta:.4f}: {result}")
```

#### 场景 2：量子算法初始化

```python
def prepare_search_state(n):
    """为 Grover 搜索准备均匀叠加态"""
    # 创建均匀叠加: |s⟩ = (1/√2ⁿ) Σᵢ |i⟩
    state = np.ones(2**n) / np.sqrt(2**n)
    return state_preparation(state)

# 为 3 比特搜索准备
circuit = prepare_search_state(3)
result = circuit.execute()

print("均匀叠加态已准备")
print(f"所有振幅相等: {np.allclose(np.abs(result), 1/np.sqrt(8))}")
```

#### 场景 3：编码经典数据

```python
def amplitude_encode_data(data):
    """
    使用振幅编码来编码经典数据
    
    输入数据必须归一化到 [-1, 1]
    """
    # 归一化数据
    normalized = np.array(data, dtype=float)
    normalized = normalized / np.linalg.norm(normalized)
    
    # 创建态
    circuit = state_preparation(normalized)
    
    return circuit

# 编码数据 [1, 2, 3, 4]
data = [1, 2, 3, 4]
circuit = amplitude_encode_data(data)

result = circuit.execute()
print(f"编码的数据: {result}")
```

---

## 三、关键 API 参考

### 3.1 state_preparation 函数

```python
state_preparation(v, backend='torch')
```

**参数：**
- `v: array-like` - 目标量子态向量
  - 必须是逐元素复数数组
  - 必须满足归一化条件：$\sum_i |v_i|^2 = 1$
  - 长度必须是 2 的整数次幂 ($2^n$)

- `backend: str` - 使用的后端
  - 默认值：`'torch'`
  - 可选值：`'torch'`, 等其他支持的后端

**返回值：**
- `GateSequence` - 准备好的量子电路对象

**异常：**
- `ValueError` - 如果向量未归一化

### 3.2 返回的 GateSequence 对象

返回的电路对象支持以下操作：

```python
# 执行电路
state = circuit.execute()

# 获取电路矩阵
matrix = circuit.get_matrix()

# 复制电路
circuit_copy = circuit.copy()

# 绘制电路
circuit.draw(filename=None, title=None)

# 获取信息
num_qubits = circuit.get_num_qubits()
backend = circuit.get_backend_type()
```

---

## 四、数学细节与验证

### 4.1 归一化条件检查

```python
import numpy as np

def check_normalization(state_vector):
    """检查向量是否归一化"""
    norm = np.linalg.norm(state_vector)
    
    print(f"向量范数: {norm}")
    print(f"是否归一化（范数=1）: {np.isclose(norm, 1)}")
    
    # 检查每个振幅的模平方和
    prob_sum = np.sum(np.abs(state_vector)**2)
    print(f"概率总和: {prob_sum}")
    
    return np.isclose(prob_sum, 1)

# 测试
state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
check_normalization(state)
```

### 4.2 相位验证

```python
import numpy as np

def analyze_phases(state_vector):
    """分析态向量中的相位信息"""
    amplitudes = np.abs(state_vector)
    phases = np.angle(state_vector)
    
    print("振幅和相位分析:")
    for i, (amp, phase) in enumerate(zip(amplitudes, phases)):
        print(f"  |{i}⟩: 振幅={amp:.4f}, 相位={phase:.4f} rad ({np.degrees(phase):.1f}°)")
    
    return amplitudes, phases

# 测试
state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
analyze_phases(state)
```

### 4.3 保真度计算

```python
def fidelity(state1, state2):
    """
    计算两个量子态之间的保真度
    F = |⟨ψ₁|ψ₂⟩|²
    """
    # 计算内积
    inner_product = np.vdot(state1, state2)
    
    # 计算保真度
    fidelity_value = np.abs(inner_product)**2
    
    return fidelity_value

# 测试
target = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
circuit = state_preparation(target)
result = circuit.execute()
f = fidelity(target, result)
print(f"保真度: {f:.6f}")
```

---

## 五、常见模式

### 模式 1：简单单比特初始化

```python
def create_single_qubit_state(theta, phi):
    """
    create arbitrary single qubit state
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    """
    state = np.array([
        np.cos(theta/2),
        np.exp(1j * phi) * np.sin(theta/2)
    ])
    return state_preparation(state)
```

### 模式 2：多比特纠缠态

```python
def create_ghz_state(n):
    """Create n-qubit GHZ state"""
    state = np.zeros(2**n)
    state[0] = 1/np.sqrt(2)
    state[-1] = 1/np.sqrt(2)
    return state_preparation(state)

def create_w_state(n):
    """Create n-qubit W state"""
    state = np.zeros(2**n)
    # W state: (1/√n)(|100...0⟩ + |010...0⟩ + ... + |000...1⟩)
    for i in range(n):
        state[2**i] = 1/np.sqrt(n)
    return state_preparation(state)
```

### 模式 3：数据编码

```python
def encode_amplitude(values):
    """
    Amplitude encoding: encode values into quantum amplitudes
    values must sum to 1 in probability (|v|² sum to 1)
    """
    data = np.array(values)
    # Normalize
    data = data / np.linalg.norm(data)
    
    return state_preparation(data)
```

---

## 六、错误处理指南

### 6.1 常见错误

**错误 1：向量未归一化**
```python
# 错误！
state = np.array([1, 1])  # ||state|| = √2 ≠ 1

try:
    circuit = state_preparation(state)
except ValueError as e:
    print(f"错误: {e}")  # The vector is not unit!

# 正确做法
state = state / np.linalg.norm(state)
circuit = state_preparation(state)
```

**错误 2：向量长度不是 2 的幂**
```python
# 错误！
state = np.array([1/np.sqrt(3)] * 3)  # 3 ≠ 2^n

# 正确做法
state = np.zeros(4)  # 4 = 2²
state[0] = 1/np.sqrt(2)
state[3] = 1/np.sqrt(2)
```

**错误 3：复数振幅处理**
```python
# 可能出错
state = np.array([0.5 + 0.5j, 0.5 - 0.5j])

# 正确做法
state = np.array([0.5 + 0.5j, 0.5 - 0.5j])
state = state / np.linalg.norm(state)  # 先归一化
circuit = state_preparation(state)
```

### 6.2 验证措施

```python
def safe_state_preparation(state_vector):
    """带验证的态准备"""
    # 1. 转换为 numpy 数组
    state = np.array(state_vector, dtype=complex)
    
    # 2. 检查长度是否为 2 的幂
    n = len(state)
    if not (n & (n - 1) == 0):
        raise ValueError(f"State length {n} is not a power of 2")
    
    # 3. 检查和归一化
    norm = np.linalg.norm(state)
    if np.isclose(norm, 0):
        raise ValueError("State vector is all zeros")
    
    state = state / norm
    
    # 4. 再次验证
    if not np.isclose(np.linalg.norm(state), 1):
        raise ValueError("Normalization failed")
    
    # 5. 准备态
    return state_preparation(state)

# 使用
try:
    circuit = safe_state_preparation([1, 1])
    print("Success!")
except ValueError as e:
    print(f"Preparation failed: {e}")
```

---

## 七、性能优化

### 7.1 电路复杂度管理

```python
def estimate_circuit_complexity(n_qubits):
    """估计态准备电路的复杂度"""
    
    num_params = 2**n_qubits
    depth = n_qubits**2  # 递归分解深度
    num_gates = 3 * (2**n_qubits - 1)  # 近似门数
    
    print(f"n = {n_qubits}:")
    print(f"  参数数: {num_params}")
    print(f"  电路深度: {depth}")
    print(f"  门数: {num_gates}")
    
    return num_params, depth, num_gates

# 分析不同规模
for n in range(1, 6):
    estimate_circuit_complexity(n)
```

### 7.2 缓存策略

```python
# 缓存已生成的电路
circuit_cache = {}

def get_or_create_circuit(state_key, state_vector):
    """带缓存的电路创建"""
    if state_key not in circuit_cache:
        circuit_cache[state_key] = state_preparation(state_vector)
    
    return circuit_cache[state_key]

# 使用
circuit = get_or_create_circuit("bell", bell_state)
```

---

## 八、完整工作流示例

```python
import numpy as np
from core.Initialization import state_preparation
from core.GateSequence import GateSequence

def complete_workflow():
    """完整的态准备和处理工作流"""
    
    # 1. 定义目标态
    print("步骤 1: 定义目标态")
    target_state = np.array([
        1/np.sqrt(8),
        1/np.sqrt(8),
        1/np.sqrt(8),
        1/np.sqrt(8),
        1/np.sqrt(8),
        np.sqrt(3/8)
    ])
    
    # 2. 准备态
    print("步骤 2: 生成态准备电路")
    circuit = state_preparation(target_state)
    
    # 3. 执行电路
    print("步骤 3: 执行电路")
    result_state = circuit.execute()
    
    # 4. 验证结果
    print("步骤 4: 验证结果")
    fidelity = np.abs(np.vdot(target_state, result_state))**2
    print(f"  保真度: {fidelity:.6f}")
    
    # 5. 扩展电路
    print("步骤 5: 扩展电路")
    extended_circuit = GateSequence(6)
    extended_circuit.append(circuit, [0, 1, 2])
    extended_circuit.h(3)
    extended_circuit.cnot(2, 5)
    
    # 6. 绘制
    print("步骤 6: 绘制电路")
    circuit.draw(title="State Preparation Circuit")
    
    return circuit, result_state, fidelity

# 执行工作流
circuit, result, f = complete_workflow()
```

---

## 九、总结检查清单

使用 Initialization 模块时，请确保：

- [ ] 目标态向量已归一化（$\sum_i |v_i|^2 = 1$）
- [ ] 向量长度是 2 的整数次幂
- [ ] 使用了正确的数据类型（复数 numpy 数组）
- [ ] 验证了生成的电路的准确性（保真度接近 1）
- [ ] 了解递归分解的复杂性
- [ ] 对于大量子比特系统（n > 10），考虑计算资源
- [ ] 如果需要多次使用同一态，实现了缓存机制
- [ ] 正确处理了包含相位的态向量
- [ ] 在集成到主电路时，使用了正确的目标比特索引
- [ ] 对复杂的量子态，考虑使用特定的编码方法（如振幅编码）

