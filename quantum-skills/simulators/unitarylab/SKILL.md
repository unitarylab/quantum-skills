---
name: unitarylab
description: |
  Comprehensive quantum computing simulator framework integrating quantum state representation, 
  register management, gate operations, state initialization, and circuit visualization.
  Provides complete toolkit for building and simulating quantum circuits with PyTorch backend support.
version: 2.0
requirements:
  - PyTorch for tensor operations and GPU acceleration
  - Matplotlib for circuit visualization
  - Numpy for numerical operations
  - Python 3.7+ for type hints and modern syntax
tags:
  - quantum simulator
  - quantum circuits
  - gate sequences
  - state preparation
  - quantum visualization
  - register management
---

# UnitaryLab 量子计算框架完全指南

## 一、系统概览

### 1.1 UnitaryLab 整体架构

**UnitaryLab** 是一个完整的量子计算模拟框架，由多个相互配合的核心模块组成：

```
┌──────────────────────────────────────────────────────────┐
│            UnitaryLab 量子计算框架                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│    ┌─────────────────────────────────────────────┐     │
│    │      GateSequence（量子门序列）              │     │
│    │  - 构建和执行量子电路                        │     │
│    │  - 寄存器管理                                │     │
│    │  - 门操作调度                                │     │
│    └─────────────────────────────────────────────┘     │
│            ↑                          ↑                  │
│    ┌───────────────┐        ┌────────────────┐        │
│    │ Register      │        │ClassicalReg    │        │
│    │(量子寄存器)   │        │(经典寄存器)    │        │
│    └───────────────┘        └────────────────┘        │
│                                                          │
│    ┌──────────────┐      ┌──────────────┐             │
│    │   State      │      │Initialization│             │
│    │ (量子态)     │      │(态准备)      │             │
│    └──────────────┘      └──────────────┘             │
│                                                          │
│    ┌──────────────────────────────────────────┐       │
│    │ CircuitDrawer（电路可视化）                │       │
│    │ - 量子电路图绘制                          │       │
│    │ - 多种输出格式                            │       │
│    └──────────────────────────────────────────┘       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 核心模块功能总结

| 模块 | 功能 | 核心类/函数 | 应用域 |
|------|------|-----------|--------|
| **GateSequence** | 量子门序列构建与执行 | GateSequence | 电路设计 |
| **Register** | 量子比特管理 | Register | 寄存器抽象 |
| **ClassicalRegister** | 经典测量结果存储 | ClassicalRegister | 混合算法 |
| **State** | 量子态表示与操作 | State | 状态分析 |
| **Initialization** | 量子态准备 | StatePreparation | 态初始化 |
| **CircuitDrawer** | 电路可视化 | CircuitDrawer | 图形输出 |

### 1.3 工作流程概览

```
量子算法设计
    ↓
创建 Register（量子寄存器）
    ↓
初始化 GateSequence（电路）
    ↓
选项 1: 直接构建电路                 选项 2: 用 Initialization 准备态
├─ circuit.h(qreg[0])        或    ├─ target_state = [1/√2, 1/√2]
├─ circuit.cx(qreg[0], qreg[1])   └─ circuit.init(target_state, qreg)
└─ ...
    ↓
执行电路（模拟或实际硬件）
    ↓
测量结果存储到 ClassicalRegister
    ↓
分析结果 / 可视化电路
    ↓
循环优化或得出结论
```

---

## 二、核心模块详解

### 2.1 Register（量子寄存器）

#### 功能概述
- **用途**：组织和管理量子比特组
- **特性**：灵活的Python风格索引、范围验证、UUID唯一标识

#### 基础概念

**寄存器是什么**：
量子寄存器是对一组量子比特的逻辑分组。与经典计算中的变量名类似，寄存器为量子比特提供了有意思的名称和索引机制。

**索引系统**：

```python
from core.register import Register

qreg = Register("q", 3)  # 创建3个比特的寄存器

# 各种索引方式
qreg[0]        # 第0个比特        → [(qreg, [0])]
qreg[0:2]      # 第0-1个比特      → [(qreg, [0, 1])]
qreg[[0, 2]]   # 第0和2个比特     → [(qreg, [0, 2])]
qreg[(0, 1)]   # 第0和1个比特     → [(qreg, [0, 1])]
qreg[-1]       # 最后一个比特     → [(qreg, [2])]
```

#### 核心特性
- **灵活索引**：整数、切片、列表、元组、负索引
- **自动范围检验**：超出范围自动抛出异常
- **UUID唯一性**：每个Register实例都有唯一ID
- **集合友好**：支持作为字典键和集合元素

#### 使用示例
```python
# 创建多个寄存器
q1 = Register("q1", 2)
q2 = Register("q2", 3)

# 在电路中使用
from core.GateSequence import GateSequence
circuit = GateSequence(q1, q2)
circuit.h(q1[0])
circuit.cx(q1[0], q2[1])
```

---

### 2.2 ClassicalRegister（经典寄存器）

#### 功能概述
- **用途**：存储量子测量结果，实现量子-经典交互
- **特性**：与Register类似的索引，支持状态追踪

#### 核心工作过程

```
初始化
  ↓ (所有比特值 = -1，表示未测量)
测量量子比特
  ↓ (获得0或1的结果)
更新经典比特值
  ↓
在经典算法中使用这些值
  ↓ (条件分支、计算等)
返回量子操作
  ↓ (根据经典结果)
完成混合算法
```

#### 索引方式

```python
from core.Classicalregister import ClassicalRegister

creg = ClassicalRegister("result", 3)

# 索引方式同 Register
creg[0]
creg[0:2]
creg[[0, 1]]
```

#### 使用示例
```python
from core.GateSequence import GateSequence
from core.Classicalregister import ClassicalRegister
from core.register import Register

qreg = Register("q", 2)
creg = ClassicalRegister("c", 2)

circuit = GateSequence(qreg, creg)
circuit.h(qreg[0])
circuit.measure(qreg, creg)  # 测量并存储结果
```

---

### 2.3 GateSequence（量子门序列）

#### 功能概述
- **用途**：构建、管理并执行量子电路
- **特性**：高层API、自动索引转换、多种门类型

#### 架构设计

**寄存器管理**：
```
GateSequence
├─ 量子寄存器列表
│  ├─ Register "q": [q0, q1, q2]
│  └─ Register "anc": [q3, q4]
├─ 经典寄存器列表
│  └─ ClassicalRegister "c": [c0, c1]
└─ 全局索引映射
   [0→(q,0), 1→(q,1), 2→(q,2), 3→(anc,0), 4→(anc,1)]
```

**门类型**：

| 门类别 | 说明 | 示例 |
|--------|------|------|
| 单比特门 | 作用于单个比特 | H, X, Y, Z, S, T |
| 参数化门 | 旋转类操作 | RX(θ), RY(θ), RZ(θ) |
| 两比特门 | 受控或交换操作 | CNOT, CZ, SWAP |
| 多控制门 | 多控制比特 | MCX, MCY, MCZ |
| 测量 | 量子到经典转换 | MEASURE |
| 自定义门 | 用户定义 | UNITARY |

#### 基础流程

```python
from core.GateSequence import GateSequence
from core.register import Register

# 第1步：创建寄存器
qreg = Register("q", 3)

# 第2步：创建电路
circuit = GateSequence(qreg)

# 第3步：构建电路
circuit.h(qreg[0])                          # H门
circuit.cx(qreg[0], qreg[1])                # CNOT门
circuit.ry(qreg[2], 1.57)                   # RY(π/2)门
circuit.measure(qreg[0])                    # 测量

# 第4步：执行和获取结果
result = circuit.execute(backend='torch')
print(result)
```

---

### 2.4 State（量子态）

#### 功能概述
- **用途**：表示和操作量子态
- **特性**：自动归一化、复数振幅支持、PyTorch后端

#### 数学基础

**状态向量表示**：
$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

其中 $\alpha_i$ 是复数振幅，满足归一化条件 $\sum_i |\alpha_i|^2 = 1$

**操作类型**：

| 操作 | 说明 | 公式 |
|------|------|------|
| 测量 | 获得概率分布 | $P(i) = \|\alpha_i\|^2$ |
| 内积 | 两个态的重叠 | $\langle \psi \| \phi \rangle$ |
| 张量积 | 合并两个系统 | $\|\psi_1\rangle \otimes \|\psi_2\rangle$ |
| 期望值 | 可观测量期望 | $\langle O \rangle$ |

#### 创建和使用

```python
from core.State import State
import numpy as np

# 创建基态
state = State(3)  # |000⟩

# 从向量创建
data = [1/np.sqrt(2), 1/np.sqrt(2)]
state = State(data)  # (|0⟩ + |1⟩)/√2，自动归一化

# 查询和操作
probs = state.probabilities()              # 概率分布
data = state.data                          # 获取振幅数据
inner = state.inner_product(other_state)   # 内积计算
```

---

### 2.5 Initialization（量子态准备）

#### 功能概述
- **用途**：自动构建量子电路以准备目标态
- **特性**：递归分解、参数优化、自动化

#### 算法原理

**单比特初始化**：
```
目标态: α₀|0⟩ + α₁|1⟩
  ↓ [提取相位]
应用全局相位补偿
  ↓ [计算旋转角]
θ = arccos(|α₀|)
  ↓ [应用RY(2θ)]
得到振幅正确的叠加
  ↓ [应用相位门]
添加相对相位
  ↓
最终状态准确
```

**多比特递归分解**：
```
n比特目标态 |v⟩
  ↓ [分割]
前2^(n-1)个振幅 v1 | 后2^(n-1)个振幅 v2
  ↓ [计算概率]
p1 = ||v1||²  (比特=(n-1)为0的概率)
p2 = ||v2||²  (比特=(n-1)为1的概率)
  ↓ [RY旋转]
在比特(n-1)上应用RY(θ)，其中cos(θ/2) = √p1
  ↓ [受控递归]
当比特=(n-1)为0: 递归准备v1/||v1||
当比特=(n-1)为1: 递归准备v2/||v2||
  ↓
完成准备
```

#### 使用示例

```python
from core.GateSequence import GateSequence
from core.Initialization import StatePreparation
from core.register import Register
import numpy as np

# 目标态：(|00⟩ + |11⟩)/√2 (Bell态)
target_state = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]

qreg = Register("q", 2)
circuit = GateSequence(qreg)

# 使用初始化模块
StatePreparation.prepare(circuit, qreg, target_state)

# 电路已经包含了准备该态所需的所有门
result = circuit.execute()
```

---

### 2.6 CircuitDrawer（电路可视化）

#### 功能概述
- **用途**：将量子电路转换为可视化图像
- **特性**：灵活布局、自定义样式、支持多格式输出

#### 绘制流程

```
量子电路 GateSequence
    ↓
提取门列表并排序
    ↓
门层折叠（优化空间利用）
    ├─ 标准折叠: 按时间顺序排列
    └─ 紧凑折叠: 更高效的空间利用
    ↓
计算坐标信息
    ├─ 每个门的 (x, y) 位置
    ├─ 字体大小、线宽
    └─ 颜色和样式
    ↓
使用 Matplotlib 逐层绘制
    ├─ 背景和网格
    ├─ 比特线和寄存器标签
    ├─ 量子门
    ├─ 控制点和连接线
    └─ 参数标签
    ↓
导出或显示
    ├─ PNG/SVG 文件
    ├─ 屏幕显示
    └─ PDF（取决于后端）
```

#### 坐标系统

```
y轴（竖直向下为正）：
  0 ← 最顶部(寄存器标签)
  ↓
  -1, -2, -3 ← 量子比特线
  ↓
  -(n+1) ← 第二层起始

x轴（水平向右为正）：
  0 ← x偏移起点
  →
  gate_x ← 第一个门的x坐标
  →
  ...
```

#### 样式系统

```python
style = {
    'backgroundColor': 'white',
    'linecolor': 'black',
    'gatetextcolor': 'black',
    'fontsize': 12,
    'lwidth1': 1.0,           # 细线
    'lwidth2': 2.0,           # 中线
    'width_per_layer': 2.0,   # 每层宽度
    'margin': 0.5,            # 边距
    'displaytext': {          # 门名称映射
        'h': 'H',
        'x': 'X',
        # ...
    }
}
```

#### 使用示例

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

qreg = Register("q", 3)
circuit = GateSequence(qreg)
circuit.h(qreg[0])
circuit.cx(qreg[0], qreg[1])

# 绘制电路
drawer = CircuitDrawer(circuit)
drawer.draw('my_circuit.png', style='default')
drawer.show()  # 显示在屏幕上
```

---

## 三、集成工作流程

### 3.1 完整量子算法示例：Bell态准备和测量

```python
from core.register import Register
from core.Classicalregister import ClassicalRegister
from core.GateSequence import GateSequence
from drawer.circuit_drawer import CircuitDrawer

# 步骤1: 创建寄存器
qreg = Register("q", 2)
creg = ClassicalRegister("result", 2)

# 步骤2: 创建电路
circuit = GateSequence(qreg, creg)

# 步骤3: 构建Bell态电路
circuit.h(qreg[0])                    # (|0⟩ + |1⟩)/√2 on q0
circuit.cx(qreg[0], qreg[1])          # 控制-X: (|00⟩ + |11⟩)/√2

# 步骤4: 测量
circuit.measure(qreg, creg)

# 步骤5: 执行
result = circuit.execute(shots=1000)
print(f"测量结果分布: {result.counts}")

# 步骤6: 可视化
drawer = CircuitDrawer(circuit)
drawer.draw('bell_circuit.png')
drawer.show()
```

### 3.2 使用Initialization进行复杂态准备

```python
from core.register import Register
from core.GateSequence import GateSequence
from core.Initialization import StatePreparation
import numpy as np

# 目标: 量子傅立叶变换的特征态
# |ψ⟩ = (|0⟩ + e^(2πi/8)|1⟩ + e^(4πi/8)|2⟩ + ...)/√8
target = np.array([
    1/np.sqrt(8) * np.exp(2j*np.pi*k/8) 
    for k in range(8)
])

qreg = Register("q", 3)
circuit = GateSequence(qreg)

# 自动准备态
StatePreparation.prepare(circuit, qreg, target)

# 电路已包含所有必要的门
```

### 3.3 分析量子态

```python
from core.State import State
import numpy as np

# 创建Bell态 (|00⟩ + |11⟩)/√2
bell_state = State([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

# 分析
probs = bell_state.probabilities()
print("测量概率:", probs)          # [0.5, 0, 0, 0.5]

# 与其他态取内积
other = State([1, 0, 0, 0])      # |00⟩
overlap = bell_state.inner_product(other)
print("|⟨00|ψ⟩|²:", abs(overlap)**2)  # 0.5
```

---

## 四、学习路径

### 第一阶段：基础（1-2周）
1. 理解 Register 和 ClassicalRegister 的概念
2. 学习 GateSequence 的基本使用
3. 实现简单的量子电路（Bell态、GHZ态）
4. 使用 CircuitDrawer 可视化电路

**关键项目**：
- 构建 2-3 个比特的量子电路
- 执行和分析测量结果
- 绘制电路图

### 第二阶段：中级（2-3周）
1. 深入理解 State 类的数学基础
2. 学习 Initialization 模块的递归分解算法
3. 实现参数化量子电路
4. 处理复杂的多比特态

**关键项目**：
- 使用参数化门构建变分电路
- 准备特定的量子态（如特征态）
- 设计混合量子-经典算法

### 第三阶段：高级（3-4周）
1. 量子算法的完整实现（Grover、QFT、VQE等）
2. 电路优化和编译
3. 错误分析和缓解
4. 大规模模拟

**关键项目**：
- 实现完整的量子算法
- 性能优化和分析
- 与实际量子硬件接口

---

## 五、常见操作参考

### 常见任务快速指南

**任务1: 创建基础量子电路**
```python
qreg = Register("q", 3)
circuit = GateSequence(qreg)
circuit.h(qreg)         # 在所有比特上应用H门
circuit.cx(qreg[0], qreg[1])  # CNOT
```

**任务2: 准备特定量子态**
```python
target_state = [1/np.sqrt(2), 0, 1/np.sqrt(2), 0]
StatePreparation.prepare(circuit, qreg, target_state)
```

**任务3: 执行和测量**
```python
creg = ClassicalRegister("c", 3)
circuit = GateSequence(qreg, creg)
circuit.measure(qreg, creg)
result = circuit.execute(shots=1000)
```

**任务4: 可视化电路**
```python
drawer = CircuitDrawer(circuit)
drawer.draw('output.png')
```

---

## 六、架构依赖关系

```
CircuitDrawer
  └─ GateSequence
      ├─ Register (管理量子比特)
      ├─ ClassicalRegister (存储测量结果)
      ├─ Initialization (可选的态准备)
      └─ State (用于分析)

State
  └─ PyTorch (后端计算)

GateSequence
  └─ 所有其他模块
```

---

## 七、版本历史

- **v2.0**: 完整集成了所有子模块，添加了综合文档和工作流
- **v1.0**: 初始版本，包含基本的门序列和寄存器功能
