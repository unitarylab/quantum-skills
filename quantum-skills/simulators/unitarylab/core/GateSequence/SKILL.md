---
name: gatesequence
description: |
  Quantum Gate Sequence module for constructing, manipulating, and executing quantum circuits.
  Provides high-level abstractions for quantum gate operations with register management and 
  classical-quantum integration capabilities.
keywords:
  - quantum gates
  - gate sequence
  - quantum circuit
  - register management
  - quantum operations
  - gate composition
  - measurement
---

# GateSequence 技能指南

## 一、算法介绍

### 1.1 基本概念

**GateSequence（量子门序列）** 是量子电路的核心抽象，用于构建、管理和执行量子计算操作。它提供了一个高层接口来操作量子比特，同时管理与经典计算系统的交互。

### 1.2 核心功能架构

#### 1.2.1 寄存器管理系统

GateSequence 管理两种类型的寄存器：

**量子寄存器（Quantum Register）**
```
┌─────────────────────────────────────┐
│       Quantum Registers              │
├─────────────────────────────────────┤
│ Register 1: q0, q1, q2 (n_qubits=3) │
│ Register 2: q3, q4 (n_qubits=2)     │
│ ...                                  │
└─────────────────────────────────────┘
        ↓ (全局索引映射)
┌─────────────────────────────────────┐
│    Global Qubit Index Space         │
│  q0, q1, q2, q3, q4, ... (total_q) │
└─────────────────────────────────────┘
```

**经典寄存器（Classical Register）**
- 存储量子测量结果
- 支撑量子-经典混合算法
- 条件量子操作基础

#### 1.2.2 索引转换机制

GateSequence 自动处理本地索引到全局索引的转换：

```python
# 本地索引 → 全局索引转换
Register qreg = Register("q", 3)
GateSequence circuit = GateSequence(qreg)

circuit.x(qreg[0])  # 本地索引 0 → 全局索引 0
circuit.h(qreg[2])  # 本地索引 2 → 全局索引 2
```

#### 1.2.3 量子门分类

| 门类型 | 说明 | 示例 |
|--------|--------|------|
| **单比特门** | 作用于单个比特 | X, Y, Z, H, S, T, RX, RY, RZ |
| **旋转门** | 参数化旋转操作 | RX(θ), RY(θ), RZ(θ) |
| **控制门** | 一个控制比特 | CNOT, CZ, CH |
| **多控制门** | 多个控制比特 | MCX, MCY, MCZ |
| **自定义门** | 用户定义的幺正算子 | Unitary(matrix) |
| **测量** | 量子到经典转换 | Measure |

### 1.3 电路处理流程

```
创建GateSequence
    ↓
添加量子/经典寄存器
    ↓
构建电路（添加量子门）
    ↓
可选：电路分解/优化
    ↓
执行电路
    ↓
获取测量结果
    ↓
提取电路矩阵或绘制图像
```

### 1.4 设计特性

**灵活的索引系统**
- 支持寄存器对象索引
- 支持整数索引
- 支持切片操作
- 自动范围验证

**后端抽象**
- 独立于具体的量子模拟器
- 支持多种后端切换
- 一致的 API 接口

**组合性**
- 电路可复制、分解、反向
- 支持电路的级联组合
- 支持控制扩展

---

## 二、使用方法步骤

### 2.1 基础使用流程

#### 第一步：导入必要的模块
```python
from core.GateSequence import GateSequence
from core.register import Register
from core.Classicalregister import ClassicalRegister
```

#### 第二步：创建寄存器
```python
# 创建量子寄存器
qreg = Register("q", 3)        # 3个量子比特

# 创建经典寄存器（可选）
creg = ClassicalRegister("c", 3)  # 3个经典比特
```

#### 第三步：初始化 GateSequence
```python
# 方法 1: 使用寄存器对象
circuit = GateSequence(qreg, creg, name="my_circuit")

# 方法 2: 直接指定比特数
circuit = GateSequence(3)  # 创建3个量子比特，自动命名

# 方法 3: 混合方式
circuit = GateSequence(qreg, creg)
```

#### 第四步：构建电路
```python
# 添加单比特门
circuit.h(qreg[0])      # 在第0个比特上应用H门
circuit.x(qreg[1])      # 在第1个比特上应用X门
circuit.rz(3.14, qreg[2])  # 在第2个比特上应用RZ门

# 添加双比特门
circuit.cnot(qreg[0], qreg[1])  # 控制-非门

# 添加测量
circuit.measure(qreg, creg)
```

#### 第五步：执行和获取结果
```python
# 执行电路
state = circuit.execute()

# 获取测量结果
print(f"测量结果: {creg.values}")

# 获取电路矩阵
matrix = circuit.get_matrix()

# 绘制电路
circuit.draw(filename="my_circuit.png")
```

### 2.2 详细使用示例 - Bell 态制备

```python
from core.GateSequence import GateSequence
from core.register import Register
from core.Classicalregister import ClassicalRegister

# 步骤 1: 创建寄存器
qreg = Register("q", 2)
creg = ClassicalRegister("c", 2)

# 步骤 2: 初始化电路
circuit = GateSequence(qreg, creg, name="bell_state")

# 步骤 3: 构建 Bell 态电路
# |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
circuit.h(qreg[0])          # 在第0个比特上应用H门
circuit.cnot(qreg[0], qreg[1])  # 受控非门

# 步骤 4: 添加测量
circuit.measure(qreg, creg)

# 步骤 5: 执行
state = circuit.execute()

# 步骤 6: 检查结果
print(f"量子态: {state}")
print(f"测量结果: {creg.values}")

# 步骤 7: 绘制电路
circuit.draw(title="Bell State Circuit")
```

### 2.3 高级使用场景

#### 场景 1: 多寄存器电路

```python
# 创建多个寄存器
qreg1 = Register("q1", 2)
qreg2 = Register("q2", 3)
creg = ClassicalRegister("c", 5)

# 创建包含多个寄存器的电路
circuit = GateSequence(qreg1, qreg2, creg, name="multi_register")

# 独立操作各寄存器
circuit.h(qreg1[0])
circuit.x(qreg2[1])
circuit.cnot(qreg1[0], qreg2[0])

# 测量
circuit.measure(qreg1, creg[0:2])
circuit.measure(qreg2, creg[2:5])
```

#### 场景 2: 受控电路

```python
circuit = GateSequence(5, name="controlled_circuit")

# 创建一个基础电路
base_circuit = GateSequence(3)
base_circuit.h(0)
base_circuit.x(1)

# 添加控制比特
controlled = circuit.control(num_ctrl_qubits=2, control_sequence="11")

# 在主电路中添加受控子电路
circuit.append(controlled, [0, 1, 2])
```

#### 场景 3: 电路分解和优化

```python
circuit = GateSequence(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.rz(1.57, 2)

# 分解电路（展开为基础门）
decomposed = circuit.decompose(times=1)

# 获取逆电路
inverse = circuit.inverse()

# 获取反向电路
reversed_circuit = circuit.reverse()

# 重复电路 n 次
repeated = circuit.repeat(times=3)
```

#### 场景 4: 量子-经典混合算法

```python
qreg = Register("q", 3)
creg = ClassicalRegister("c", 3)
circuit = GateSequence(qreg, creg)

# 第一步: 创建叠加态
circuit.h(qreg)  # 在所有比特上加 H 门

# 第二步: 应用受控操作（基于前一步的隐式结果）
for i in range(3):
    circuit.rz(1.57, qreg[i])

# 第三步: 测量
circuit.measure(qreg, creg)

# 执行
state = circuit.execute()

# 分析结果
measured_values = creg.values
print(f"测量结果: {measured_values}")
```

#### 场景 5: 自定义幺正操作

```python
import numpy as np

circuit = GateSequence(2)

# 定义自定义 2 比特门（4×4 幺正矩阵）
custom_unitary = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

# 应用自定义门
circuit.unitary(custom_unitary, [0, 1])
```

---

## 三、关键 API 参考

### 3.1 构造函数

```python
GateSequence(*args, **kwargs)
```

**参数：**
- `num_qubits: int` - 量子比特数量（直接指定）
- `registers: Register | ClassicalRegister` - 寄存器对象序列
- `name: str` - 电路名称（可选，默认 'Sequence'）

**示例：**
```python
# 方式 1: 数字
circuit = GateSequence(3)

# 方式 2: 寄存器
circuit = GateSequence(qreg, creg)

# 方式 3: 混合
circuit = GateSequence(qreg, creg, name="hybrid")
```

### 3.2 单比特门

| 方法 | 参数 | 说明 |
|------|------|------|
| `x(qubit)` | qubit | Pauli-X 门 |
| `y(qubit)` | qubit | Pauli-Y 门 |
| `z(qubit)` | qubit | Pauli-Z 门 |
| `h(qubit)` | qubit | Hadamard 门 |
| `s(qubit)` | qubit | S 门 |
| `t(qubit)` | qubit | T 门 |
| `rx(angle, qubit)` | angle, qubit | RX 旋转门 |
| `ry(angle, qubit)` | angle, qubit | RY 旋转门 |
| `rz(angle, qubit)` | angle, qubit | RZ 旋转门 |

### 3.3 双比特门

| 方法 | 参数 | 说明 |
|------|------|------|
| `cnot(control, target)` | control, target | 受控-非 |
| `cz(control, target)` | control, target | 受控-Z |
| `ch(control, target)` | control, target | 受控-H |
| `swap(qubit1, qubit2)` | qubit1, qubit2 | 交换 |

### 3.4 多控制门

```python
# 多个控制比特
circuit.mcx(controls, target)      # 多控-X
circuit.mcy(controls, target)      # 多控-Y
circuit.mcz(controls, target)      # 多控-Z
circuit.mcrx(angle, controls, target)  # 多控-RX
```

### 3.5 电路操作

```python
# 复制
new_circuit = circuit.copy()

# 分解
decomposed = circuit.decompose(times=1)

# 逆操作
inverse = circuit.inverse()
dagger = circuit.dagger()  # 共轭转置

# 反向
reversed_circuit = circuit.reverse()

# 重复
repeated = circuit.repeat(times=3)

# 添加子电路
circuit.append(sub_circuit, target, control)
circuit.prepend(sub_circuit, target)

# 添加控制
controlled = circuit.control(num_ctrl_qubits=2)
```

### 3.6 测量与执行

```python
# 测量
circuit.measure(target_qubits, classical_bits)

# 执行
state = circuit.execute(initial_state=None)

# 获取矩阵
matrix = circuit.get_matrix(m=0)

# 绘制
circuit.draw(filename=None, title=None, style='dark', compact=True)
```

### 3.7 信息查询

```python
# 获取比特数
num_qubits = circuit.get_num_qubits()

# 获取后端类型
backend = circuit.get_backend_type()

# 获取电路名称
name = circuit.name

# 获取寄存器
regs = circuit.registers
cl_regs = circuit.classical_registers
```

---

## 四、索引方式详解

### 4.1 支持的索引类型

```python
circuit = GateSequence(qreg, creg)

# 单个比特
circuit.x(qreg[0])           # 整数索引

# 比特范围
circuit.h(qreg[0:2])         # 切片

# 多个比特
circuit.x(qreg[[0, 2, 3]])   # 列表索引

# 元组索引
circuit.y(qreg[(0, 2)])      # 元组

# 全部比特
circuit.z(qreg)              # 寄存器对象
```

### 4.2 索引转换内部机制

```
本地索引 (Local Index)
    ↓
通过寄存器对象查找
    ↓
获取全局起始位置
    ↓
计算全局索引 (Global Index)
    ↓
应用量子门
```

---

## 五、常见模式

### 模式 1: 简单电路构建

```python
def build_simple_circuit(n_qubits):
    circuit = GateSequence(n_qubits)
    
    # 创建 GHZ 态
    circuit.h(0)
    for i in range(1, n_qubits):
        circuit.cnot(0, i)
    
    return circuit
```

### 模式 2: 电路模板

```python
def grover_diffusion(circuit, qubits):
    """Grover 算法的扩散算子"""
    circuit.h(qubits)
    circuit.x(qubits)
    circuit.mcrz(3.14, qubits[:-1], qubits[-1])
    circuit.x(qubits)
    circuit.h(qubits)
```

### 模式 3: 参数化电路

```python
def parameterized_circuit(angles, qubits):
    circuit = GateSequence(len(qubits))
    
    for i, angle in enumerate(angles):
        circuit.rz(angle, qubits[i])
    
    for i in range(len(qubits)-1):
        circuit.cnot(qubits[i], qubits[i+1])
    
    return circuit
```

---

## 六、错误处理指南

### 6.1 常见错误

**错误 1: 索引超出范围**
```python
qreg = Register("q", 3)
circuit = GateSequence(qreg)

# 错误！
# circuit.x(qreg[5])  # IndexError

# 正确
circuit.x(qreg[2])
```

**错误 2: 寄存器未添加**
```python
qreg = Register("q", 3)
circuit = GateSequence(5)  # 创建 5 个独立比特

# 错误！
# circuit.x(qreg[0])  # KeyError: Register not found

# 正确
circuit.x(0)
```

**错误 3: 不匹配的测量**
```python
qreg = Register("q", 3)
creg = ClassicalRegister("c", 2)
circuit = GateSequence(qreg, creg)

# 错误！
# circuit.measure(qreg, creg)  # ValueError: 比特数不匹配

# 正确
circuit.measure(qreg[0:2], creg)
```

### 6.2 最佳实践

```python
# 好的做法：类型检查
def safe_apply_gate(circuit, target):
    try:
        if isinstance(target, Register):
            circuit.h(target)
        else:
            raise TypeError("Expected Register or int")
    except (KeyError, IndexError, ValueError) as e:
        print(f"电路操作失败: {e}")

# 避免：假设索引有效
# circuit.x(random_index)  # 可能失败
```

---

## 七、性能优化建议

### 7.1 批量操作

```python
# 不推荐：多次单独调用
for i in range(10):
    circuit.h(i)

# 推荐：在列表上应用
circuit.h([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### 7.2 电路重用

```python
# 创建一次，多次使用
base_circuit = GateSequence(3)
base_circuit.h(0)
base_circuit.cnot(0, 1)

# 复制并扩展
circuit1 = base_circuit.copy()
circuit1.x(2)

circuit2 = base_circuit.copy()
circuit2.z(2)
```

### 7.3 矩阵缓存

```python
# 第一次调用：计算并缓存
matrix1 = circuit.get_matrix()

# 后续调用：如果电路未修改，使用缓存
matrix2 = circuit.get_matrix()  # 快速
```

---

## 八、集成示例 - 完整的量子算法

```python
from core.GateSequence import GateSequence
from core.register import Register
from core.Classicalregister import ClassicalRegister

# Deutsch 算法实现
def deutsch_algorithm(oracle_type='balanced'):
    """
    实现 Deutsch 算法
    
    Parameters:
    oracle_type: 'constant' 或 'balanced'
    """
    # 创建寄存器
    qreg = Register("q", 2)
    creg = ClassicalRegister("c", 2)
    
    # 创建电路
    circuit = GateSequence(qreg, creg, name="deutsch")
    
    # 初始化：|01⟩
    circuit.x(qreg[1])
    
    # 应用 Hadamard 变换
    circuit.h(qreg[0])
    circuit.h(qreg[1])
    
    # 应用预言机（Oracle）
    if oracle_type == 'constant':
        # 常数预言机：什么都不做或全局相位
        # （此处省略实现细节）
        pass
    else:  # balanced
        # 平衡预言机
        circuit.cnot(qreg[0], qreg[1])
    
    # 最终 Hadamard 变换
    circuit.h(qreg[0])
    
    # 测量第一个比特
    circuit.measure(qreg[0], creg[0])
    
    return circuit

# 使用
circuit = deutsch_algorithm('balanced')
state = circuit.execute()
print(f"结果: {circuit.classical_registers[0].values}")
```

---

## 九、总结检查清单

在使用 GateSequence 时，请确保：

- [ ] 已正确创建量子和经典寄存器
- [ ] GateSequence 初始化时包含了所有必要的寄存器
- [ ] 使用了正确的索引方式（整数、寄存器对象、切片等）
- [ ] 单比特和双比特门的参数正确
- [ ] 测量前已添加经典寄存器
- [ ] 测量的比特数与经典位数相匹配
- [ ] 在执行前已完成电路构建
- [ ] 处理了可能的错误和异常
- [ ] 对性能敏感的场景使用了批量操作
- [ ] 正确理解了本地索引和全局索引的区别

