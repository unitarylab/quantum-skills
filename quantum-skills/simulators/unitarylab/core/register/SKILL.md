---
name: register
description: |
  Quantum Register module for managing groups of quantum qubits with flexible indexing.
  Provides high-level abstractions for accessing and manipulating qubits through 
  Python-style indexing patterns (integers, slices, lists, tuples).
keywords:
  - quantum register
  - qubit group
  - quantum indexing
  - qubit management
  - register abstraction
---

# Register 技能指南

## 一、算法介绍

### 1.1 基本概念

**Register（量子寄存器）** 是量子计算中用于组织和管理一组量子比特的抽象。它提供了灵活的索引机制，使得用户可以用 Python 风格的索引方式来访问和操作量子比特，类似于处理数组元素。

### 1.2 设计原理

#### 1.2.1 寄存器层次结构

```
├─ 物理量子硬件
│  └─ 物理量子比特（Qubits）
│
├─ 寄存器抽象层
│  ├─ Register 对象：逻辑分组
│  │  ├─ 比特 0
│  │  ├─ 比特 1
│  │  ├─ 比特 2
│  │  └─ ...
│  │
│  └─ 索引映射
│     ├─ 本地索引（Local）
│     └─ 全局索引（Global）
│
└─ 应用层
   ├─ 量子电路（GateSequence）
   ├─ 量子算法
   └─ 混合算法
```

#### 1.2.2 索引系统

Register 支持多种索引方式，将其转换为统一的内部表示：

| 索引类型 | 符号 | 返回值 | 说明 |
|---------|------|--------|------|
| 整数 | `r[0]` | `[(r, [0])]` | 单个比特 |
| 切片 | `r[0:2]` | `[(r, [0, 1])]` | 连续比特 |
| 列表 | `r[[0, 2]]` | `[(r, [0, 2])]` | 批量选择 |
| 元组 | `r[(0, 2)]` | `[(r, [0, 2])]` | 非连续选择 |
| 负索引 | `r[-1]` | `[(r, [n-1])]` | 倒数访问 |

#### 1.2.3 返回值结构

所有 `__getitem__` 操作都返回统一的格式：

```python
[(Register对象, [索引列表])]
```

这种设计的好处：

1. **统一接口**：无论使用何种索引方式，返回类型一致
2. **向上传导**：Register 信息可以传递给 GateSequence
3. **灵活性**：支持复合操作和链式调用
4. **可扩展性**：易于在 GateSequence 中进行全局索引转换

### 1.3 唯一性和哈希

#### 1.3.1 UUID 标识符

每个 Register 实例都获得一个唯一的 UUID：

```python
import uuid

r1 = Register("q", 3)
r2 = Register("q", 3)

print(r1.id)  # 例如：'a1b2c3d4e5f6...'
print(r2.id)  # 例如：'x9y8z7w6v5u4...'
print(r1.id == r2.id)  # False，即使名称和大小相同
```

#### 1.3.2 哈希和字典使用

Register 支持作为字典键和集合元素使用：

```python
# 用作字典键
mapping = {}
r = Register("q", 3)
mapping[r] = "value"

# 用于集合
registers_set = {r1, r2, r3}

# 在 Python 字典中使用
register_info = {
    r: {"type": "quantum", "size": 3}
}
```

### 1.4 范围验证机制

Register 自动检查索引范围的有效性：

```
索引提交
  ↓
类型检查（int, slice, tuple, list）
  ↓
范围验证
  ├─ max(indices) < n_qubits
  ├─ min(indices) >= -n_qubits
  └─ 都有效则返回，否则抛出异常
```

**负索引支持**：
- Python 风格的负索引被正确处理
- `-1` 表示最后一个比特
- `-n` 表示第一个比特
- 范围检查同时验证正负索引

---

## 二、使用方法步骤

### 2.1 基础使用流程

#### 第一步：导入模块
```python
from core.register import Register
```

#### 第二步：创建寄存器
```python
# 创建一个包含 3 个量子比特的寄存器
qreg = Register("q", 3)

# 创建多个不同的寄存器
qreg1 = Register("q1", 2)
qreg2 = Register("q2", 4)
```

#### 第三步：访问比特
```python
# 单个比特
bit0 = qreg[0]
bit_last = qreg[-1]

# 比特范围
bits_0_1 = qreg[0:2]
bits_all = qreg[:]

# 多个具体比特
bits_select = qreg[[0, 2]]

# 使用元组
bits_tuple = qreg[(0, 2, 1)]
```

#### 第四步：查询寄存器信息
```python
# 获取比特数
num_qubits = len(qreg)
num_qubits = qreg.n_qubits

# 获取名称
name = qreg.name

# 获取唯一 ID
unique_id = qreg.id

# 字符串表示
print(qreg)  # Register(name='q', n_qubits=3)
```

### 2.2 详细使用示例

#### 例子 1：基本寄存器创建和访问

```python
from core.register import Register

# 创建一个 5 比特寄存器
qreg = Register("quantum", 5)

# 信息查询
print(f"寄存器名称: {qreg.name}")
print(f"比特总数: {len(qreg)}")
print(f"寄存器 ID: {qreg.id}")

# 访问单个比特
print(qreg[0])  # 第一个比特
print(qreg[4])  # 最后一个比特
print(qreg[-1])  # 倒数第一个比特
print(qreg[-5])  # 倒数第五个（第一个）

# 访问范围
print(qreg[0:3])  # 前三个比特
print(qreg[2:])   # 从第二个开始的所有比特
print(qreg[::2])  # 每隔一个比特
```

#### 例子 2：灵活的索引方式

```python
from core.register import Register

qreg = Register("q", 8)

# 方式 1: 单个索引
result1 = qreg[3]
# 返回: [(qreg, [3])]

# 方式 2: 切片（Python 标准）
result2 = qreg[2:5]
# 返回: [(qreg, [2, 3, 4])]

# 方式 3: 步长切片
result3 = qreg[::2]
# 返回: [(qreg, [0, 2, 4, 6])]

# 方式 4: 列表选择
result4 = qreg[[1, 3, 5]]
# 返回: [(qreg, [1, 3, 5])]

# 方式 5: 元组选择
result5 = qreg[(0, 2, 7)]
# 返回: [(qreg, [0, 2, 7])]

# 方式 6: 倒数索引
result6 = qreg[-1]
# 返回: [(qreg, [7])]

# 方式 7: 倒数范围
result7 = qreg[-3:]
# 返回: [(qreg, [5, 6, 7])]

print(f"单个: {result1}")
print(f"切片: {result2}")
print(f"步长: {result3}")
print(f"列表: {result4}")
print(f"元组: {result5}")
print(f"负数: {result6}")
print(f"负数范围: {result7}")
```

#### 例子 3: 与 GateSequence 集成

```python
from core.register import Register
from core.GateSequence import GateSequence

# 创建寄存器
qreg = Register("q", 3)

# 创建量子电路
circuit = GateSequence(qreg, name="demo")

# 使用寄存器索引应用门
circuit.h(qreg[0])           # H 门在第 0 比特
circuit.x(qreg[1:3])         # X 门在第 1, 2 比特
circuit.cnot(qreg[0], qreg[1])  # CNOT：控制=0，目标=1

# 执行
result = circuit.execute()
print(f"执行完成")
```

#### 例子 4: 多寄存器管理

```python
from core.register import Register
from core.GateSequence import GateSequence

# 创建多个寄存器
qreg_data = Register("data", 3)
qreg_ancilla = Register("ancilla", 2)

# 创建电路
circuit = GateSequence(qreg_data, qreg_ancilla)

# 操作数据寄存器
circuit.h(qreg_data)

# 操作辅助寄存器
circuit.x(qreg_ancilla[0])

# 跨寄存器的相互作用
circuit.cnot(qreg_data[0], qreg_ancilla[1])

# 执行
result = circuit.execute()
print("多寄存器电路完成")
```

#### 例子 5: 寄存器比较和相等性

```python
from core.register import Register

# 创建寄存器
qreg1 = Register("q", 3)
qreg2 = Register("q", 3)
qreg3 = qreg1  # 引用同一对象

# 比较
print(f"qreg1 == qreg2: {qreg1 == qreg2}")  # False，不同对象
print(f"qreg1 == qreg3: {qreg1 == qreg3}")  # True，同一对象
print(f"qreg1.id == qreg2.id: {qreg1.id == qreg2.id}")  # False

# 用于集合
regs_set = {qreg1, qreg2, qreg3}
print(f"集合大小: {len(regs_set)}")  # 2（qreg1 和 qreg3 是同一个）
```

### 2.3 高级使用场景

#### 场景 1: 动态寄存器创建

```python
def create_registers_for_algorithm(n_data, n_ancilla):
    """为特定算法创建所需的寄存器"""
    data_reg = Register("data", n_data)
    ancilla_reg = Register("ancilla", n_ancilla)
    return data_reg, ancilla_reg

# 使用
n = 4  # 数据比特数
m = 2  # 辅助比特数

data_qreg, ancilla_qreg = create_registers_for_algorithm(n, m)

print(f"数据寄存器: {data_qreg}")
print(f"辅助寄存器: {ancilla_qreg}")
```

#### 场景 2: 寄存器分区操作

```python
from core.register import Register

qreg = Register("q", 8)

# 将寄存器分成若干部分
upper_half = qreg[0:4]    # 前一半
lower_half = qreg[4:]     # 后一半

# 奇偶分割
odd_qubits = qreg[::2]    # 偶数位置（0, 2, 4, 6）
even_qubits = qreg[1::2]  # 奇数位置（1, 3, 5, 7）

# 特定比特
important_bits = qreg[[0, 3, 7]]  # 关键比特

print(f"前一半: {upper_half}")
print(f"后一半: {lower_half}")
print(f"奇数位: {odd_qubits}")
print(f"偶数位: {even_qubits}")
print(f"关键位: {important_bits}")
```

#### 场景 3: 寄存器映射和跟踪

```python
from core.register import Register

# 创建多个寄存器
registers = [
    Register("input", 3),
    Register("work", 4),
    Register("output", 2)
]

# 使用字典进行映射
reg_map = {
    reg.id: {"name": reg.name, "size": len(reg)}
    for reg in registers
}

# 查询信息
for reg in registers:
    info = reg_map[reg.id]
    print(f"寄存器 {info['name']}: {info['size']} 比特")

# 反向查询
by_name = {
    reg.name: reg
    for reg in registers
}

input_reg = by_name["input"]
print(f"输入寄存器: {input_reg}")
```

#### 场景 4: 寄存器索引工具函数

```python
from core.register import Register

def get_qubit_indices(register, index_spec):
    """
    灵活的比特索引获取函数
    
    index_spec 可以是:
    - int: 单个索引
    - str: 'all', 'even', 'odd', 'first', 'last'
    - slice: 切片对象
    - list/tuple: 索引列表
    """
    if isinstance(index_spec, str):
        n = len(register)
        if index_spec == 'all':
            return register[:]
        elif index_spec == 'even':
            return register[::2]
        elif index_spec == 'odd':
            return register[1::2]
        elif index_spec == 'first':
            return register[0]
        elif index_spec == 'last':
            return register[-1]
        else:
            raise ValueError(f"Unknown spec: {index_spec}")
    else:
        return register[index_spec]

# 使用
qreg = Register("q", 6)

print(get_qubit_indices(qreg, 'all'))    # 所有比特
print(get_qubit_indices(qreg, 'even'))   # 偶数位置
print(get_qubit_indices(qreg, 'odd'))    # 奇数位置
print(get_qubit_indices(qreg, 'first'))  # 第一个
print(get_qubit_indices(qreg, 'last'))   # 最后一个
print(get_qubit_indices(qreg, [0, 3]))   # 特定位置
```

---

## 三、关键 API 参考

### 3.1 构造函数

```python
Register(name: str, n_qubits: int)
```

**参数：**
- `name: str` - 寄存器的标识符（用于调试和显示）
- `n_qubits: int` - 寄存器包含的量子比特数量

**属性：**
- `name: str` - 寄存器名称
- `n_qubits: int` - 比特数量
- `id: str` - 唯一的 UUID 标识符

**示例：**
```python
qreg = Register("quantum_register", 4)
```

### 3.2 索引方法 - `__getitem__`

```python
register[index] -> List[Tuple[Register, List[int]]]
```

**支持的索引类型：**

| 类型 | 示例 | 返回值 |
|------|------|--------|
| `int` | `r[0]` | `[(r, [0])]` |
| `slice` | `r[0:2]` | `[(r, [0, 1])]` |
| `list` | `r[[0, 2]]` | `[(r, [0, 2])]` |
| `tuple` | `r[(0, 2)]` | `[(r, [0, 2])]` |

**负索引支持：**
```python
r = Register("q", 5)
r[-1]      # 最后一个比特 → [(r, [4])]
r[-2:]     # 最后两个 → [(r, [3, 4])]
r[:-1]     # 除了最后一个 → [(r, [0, 1, 2, 3])]
```

**异常：**
- `TypeError` - 索引类型不支持
- `IndexError` - 索引超出范围

### 3.3 长度方法 - `__len__`

```python
len(register) -> int
```

返回寄存器中的比特数量。

**示例：**
```python
qreg = Register("q", 5)
print(len(qreg))  # 5
```

### 3.4 相等性比较 - `__eq__`

```python
register1 == register2 -> bool
```

两个 Register 对象相等当且仅当它们有相同的唯一 ID（即同一个对象）。

**示例：**
```python
r1 = Register("q", 3)
r2 = Register("q", 3)
r3 = r1

print(r1 == r2)  # False
print(r1 == r3)  # True（同一对象）
```

### 3.5 哈希方法 - `__hash__`

```python
hash(register) -> int
```

返回基于名称、比特数和 ID 的哈希值，使 Register 可用作字典键或集合元素。

**示例：**
```python
qreg = Register("q", 3)
reg_dict = {qreg: "value"}
reg_set = {qreg}
```

### 3.6 字符串表示 - `__repr__`

```python
repr(register) -> str
```

返回可读的字符串表示。

**示例：**
```python
qreg = Register("quantum", 3)
print(qreg)
# 输出: Register(name='quantum', n_qubits=3)
```

---

## 四、索引系统详解

### 4.1 Python 风格索引

Register 完全支持 Python 列表风格的索引：

```python
qreg = Register("q", 10)

# 基本索引
qreg[0]        # 第 0 个
qreg[9]        # 第 9 个
qreg[-1]       # 最后一个（第 9 个）
qreg[-10]      # 第一个

# 切片
qreg[2:5]      # 第 2, 3, 4 个
qreg[:3]       # 前 3 个（0, 1, 2）
qreg[7:]       # 从第 7 个开始
qreg[:]        # 所有

# 步长
qreg[::2]      # 每隔一个（0, 2, 4, 6, 8）
qreg[1::2]     # 从 1 开始每隔一个（1, 3, 5, 7, 9）
qreg[::-1]     # 反向（9, 8, 7, ..., 0）
```

### 4.2 返回值统一性

所有索引操作都返回相同的格式，便于与 GateSequence 集成：

```python
qreg = Register("q", 5)

# 虽然索引方式不同，返回格式相同
print(qreg[0])        # [(qreg, [0])]
print(qreg[0:1])      # [(qreg, [0])]
print(qreg[[0]])      # [(qreg, [0])]
print(qreg[(0,)])     # [(qreg, [0])]
```

### 4.3 索引范围检查

```python
qreg = Register("q", 3)

# 有效索引：0, 1, 2, -1, -2, -3
qreg[0]     # ✓
qreg[2]     # ✓
qreg[-1]    # ✓
qreg[-3]    # ✓

# 无效索引
# qreg[3]    # ✗ IndexError
# qreg[-4]   # ✗ IndexError
```

---

## 五、常见模式

### 模式 1: 量子-经典分离

```python
def setup_quantum_classical():
    """设置量子和经典寄存器"""
    from core.register import Register
    from core.Classicalregister import ClassicalRegister
    
    qreg = Register("q", 3)
    creg = ClassicalRegister("c", 3)
    
    return qreg, creg
```

### 模式 2: 多部分寄存器管理

```python
def create_modular_registers(n_encoding, n_work, n_output):
    """为模块化算法创建寄存器"""
    return {
        'input': Register("input", n_encoding),
        'work': Register("work", n_work),
        'output': Register("output", n_output)
    }

# 使用
regs = create_modular_registers(4, 2, 3)
encoding_reg = regs['input']
work_reg = regs['work']
output_reg = regs['output']
```

### 模式 3: 寄存器验证

```python
def validate_register(reg, expected_size=None):
    """验证寄存器的有效性"""
    if not isinstance(reg, Register):
        raise TypeError(f"Expected Register, got {type(reg)}")
    
    if expected_size and len(reg) != expected_size:
        raise ValueError(f"Expected size {expected_size}, got {len(reg)}")
    
    return True

# 使用
qreg = Register("q", 3)
validate_register(qreg, expected_size=3)  # ✓
```

### 模式 4: 动态分区

```python
def partition_register(register, partition_size):
    """将寄存器分割成指定大小的块"""
    n = len(register)
    partitions = []
    
    for i in range(0, n, partition_size):
        end = min(i + partition_size, n)
        partition = register[i:end]
        partitions.append(partition)
    
    return partitions

# 使用
qreg = Register("q", 8)
parts = partition_register(qreg, 2)
# parts = [qreg[0:2], qreg[2:4], qreg[4:6], qreg[6:8]]
```

---

## 六、错误处理指南

### 6.1 常见错误

**错误 1: 索引超出范围**
```python
qreg = Register("q", 3)

# 错误！
# qreg[3]  # IndexError: Register index [3] out of range

# 正确
qreg[2]  # 最后一个有效索引
qreg[-1]  # 或用负索引
```

**错误 2: 无效的索引类型**
```python
qreg = Register("q", 3)

# 错误！
# qreg[1.5]  # TypeError
# qreg["0"]  # TypeError

# 正确
qreg[1]      # int
qreg[0:2]    # slice
qreg[[0, 1]]  # list
```

**错误 3: 混淆本地和全局索引**
```python
# Register 总是使用本地索引
qreg = Register("q", 3)
result = qreg[0]  # 这是本地索引 0

# GateSequence 将其转换为全局索引
circuit = GateSequence(qreg)
circuit.h(qreg[0])  # GateSequence 内部处理转换
```

### 6.2 最佳实践

```python
def safe_index_register(register, index):
    """安全的寄存器索引"""
    try:
        # 验证类型
        if not isinstance(register, Register):
            raise TypeError(f"Expected Register, got {type(register)}")
        
        # 验证索引
        if isinstance(index, int):
            n = len(register)
            if not (-n <= index < n):
                raise IndexError(f"Index {index} out of range [0, {n-1}]")
        
        # 执行索引
        result = register[index]
        return result
        
    except (TypeError, IndexError) as e:
        print(f"索引失败: {e}")
        return None

# 使用
qreg = Register("q", 5)
safe_index_register(qreg, 2)   # ✓
safe_index_register(qreg, 10)  # ✗ 带错误处理
```

---

## 七、性能考虑

### 7.1 内存效率

```python
import sys

qreg = Register("q", 1000)

# Register 对象本身占用空间很小
size = sys.getsizeof(qreg)
print(f"Register 对象大小: {size} 字节")  # 约 60-100 字节

# UUID 字符串占用的额外空间
id_size = sys.getsizeof(qreg.id)
print(f"ID 字符串大小: {id_size} 字节")
```

### 7.2 索引操作性能

```python
from core.register import Register
import time

qreg = Register("q", 1000)

# 索引操作是 O(n)（n = 索引数量）
start = time.time()
for _ in range(10000):
    result = qreg[500]
end = time.time()

print(f"单个索引 10000 次: {end - start:.6f} 秒")

# 范围索引生成索引列表，也是 O(n)
start = time.time()
for _ in range(10000):
    result = qreg[0:100]
end = time.time()

print(f"范围索引 10000 次: {end - start:.6f} 秒")
```

---

## 八、完整工作流示例

```python
from core.register import Register
from core.GateSequence import GateSequence
from core.Classicalregister import ClassicalRegister

def complete_register_workflow():
    """完整的寄存器使用工作流"""
    
    # 步骤 1: 创建寄存器
    print("步骤 1: 创建寄存器")
    qreg = Register("quantum", 4)
    creg = ClassicalRegister("classical", 4)
    
    # 步骤 2: 检查寄存器信息
    print(f"  寄存器: {qreg}")
    print(f"  比特数: {len(qreg)}")
    
    # 步骤 3: 创建电路
    print("步骤 2: 创建量子电路")
    circuit = GateSequence(qreg, creg)
    
    # 步骤 4: 使用不同的索引方式应用门
    print("步骤 3: 应用量子门")
    circuit.h(qreg[0])          # 单个比特
    circuit.x(qreg[1:3])        # 范围
    circuit.z(qreg[[3]])        # 列表
    
    # 步骤 5: 添加相互作用
    print("步骤 4: 添加相互作用")
    circuit.cnot(qreg[0], qreg[1])
    circuit.cnot(qreg[2], qreg[3])
    
    # 步骤 6: 测量
    print("步骤 5: 测量")
    circuit.measure(qreg, creg)
    
    # 步骤 7: 执行
    print("步骤 6: 执行电路")
    result = circuit.execute()
    
    print(f"  执行完成")
    print(f"  测量结果: {creg.values}")
    
    return circuit, result

# 执行
circuit, result = complete_register_workflow()
```

---

## 九、总结检查清单

使用 Register 时，请确保：

- [ ] 已正确导入 Register 类
- [ ] 为寄存器指定了有意义的名称
- [ ] 寄存器大小正确反映了所需的比特数量
- [ ] 理解了返回值格式 `[(register, [indices])]`
- [ ] 使用了正确的索引方式（int、slice、list、tuple）
- [ ] 注意了负索引的支持
- [ ] 在范围内访问索引（不超出边界）
- [ ] 必要时将 Register 用作字典键或集合元素
- [ ] 理解了寄存器是通过 UUID 来比较相等性的
- [ ] 在与 GateSequence 集成时，正确传递了寄存器索引

