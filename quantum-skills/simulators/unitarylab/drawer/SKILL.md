---
name: circuitdrawer
description: |
  Quantum circuit visualization and drawing module using Matplotlib.
  Provides comprehensive tools for rendering quantum circuits with flexible layout options,
  gate styling, and support for various quantum gate types including single-qubit, multi-qubit,
  and controlled gates with customizable appearance and formatting.
keywords:
  - circuit drawer
  - quantum visualization
  - circuit rendering
  - gate drawing
  - circuit layout
  - matplotlib
  - quantum circuit diagram
---

# Circuit Drawer 技能指南

## 一、算法介绍

### 1.1 基本概念

**Circuit Drawer（量子电路绘制器）** 是将量子电路的抽象表示转换为可视化图像的工具。它负责：
- 将量子门序列转换为可理解的电路图
- 管理电路布局和空间利用
- 应用样式和美学设计
- 支持交互式或静态图像导出

### 1.2 核心架构

#### 1.2.1 绘制流程

```
量子电路 (GateSequence)
    ↓
提取门列表
    ↓
门层折叠 (Layering)
    ├─ 标准折叠: get_layered_gates
    └─ 紧凑折叠: compact_get_layered_gates
    ↓
计算坐标信息
    ├─ 门位置 (x, y)
    ├─ 字体大小
    ├─ 线宽
    └─ 颜色配置
    ↓
逐层绘制
    ├─ 绘制寄存器标签和线路
    ├─ 绘制各类量子门
    ├─ 绘制控制点和连接线
    └─ 绘制其他元素（标签、参数）
    ↓
最终图像 (PNG/SVG/显示)
```

#### 1.2.2 坐标系统

```
y轴方向（向下为正）：
  ↓ 0 ← 顶部
  ↓
  ↓ -1, -2, -3, ... ← 每个比特向下
  ↓
  ↓ -(n+1) ← 折叠后第二层起始
  ↓

x轴方向（向右为正）：
  ←─── 0 (x_offset)
       │
       gate1 ← [x, width]
       │
  gate2 ← [x+width, width]
```

#### 1.2.3 门折叠机制

**标准折叠 (Non-compact)**
```
Gate1  Gate2  Gate3
|─────|─────|─────|
Layer 0: 按时间顺序放置，超过宽度限制则折叠

Gate4  Gate5
|─────|─────|
Layer 1: 继续放置后续门
```

**紧凑折叠 (Compact)**
```
关键优化：
1. 每个比特独立维护"光标位置"
2. 无依赖关系的门可并行放置
3. 测量门自动对齐所有比特

结果：更高效的空间利用
```

### 1.3 样式系统

CircuitDrawer 使用分层样式系统：

```python
样式配置 (Style Dictionary)
├─ 颜色 (Color)
│  ├─ 背景颜色 (backgroundColor)
│  ├─ 线条颜色 (linecolor)
│  ├─ 文本颜色 (gatetextcolor)
│  └─ 显示颜色 (displaycolor)
├─ 尺寸 (Size)
│  ├─ 字体大小 (fontsize)
│  ├─ 线宽 (lwidth1/2/3/4)
│  ├─ 每层宽度 (width_per_layer)
│  └─ 边距 (margin)
├─ 显示文本 (Display Text)
│  └─ 门名称映射 (displaytext)
└─ 布局参数
   ├─ x/y 偏移 (x_offset, y_offset)
   └─ 比例因子 (scale_char)
```

### 1.4 门类型分类

| 门类型 | 绘制方法 | 特点 | 示例 |
|--------|---------|------|------|
| **单比特门** | `_gate()` | 单个量子比特上的操作 | H, X, Y, Z, S, T, RX, RY, RZ |
| **多比特门** | `_multiqubit_gate()` | 多个量子比特上的操作 | SWAP, CX, CZ, CH |
| **受控门** | `_control_gate()` | 带控制点的量子门 | CNOT, CCX, CRX |
| **Swap门** | `_swap()` | 交换两个量子比特 | SWAP |

---

## 二、使用方法步骤

### 2.1 基础使用流程

#### 第一步：导入模块
```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register
```

#### 第二步：创建量子电路
```python
# создать 寄存器和电路
qreg = Register("q", 3)
circuit = GateSequence(qreg)

# 构建电路
circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])
circuit.cnot(qreg[1], qreg[2])
```

#### 第三步：创建 CircuitDrawer
```python
# 创建绘制器
drawer = CircuitDrawer(
    scale=1.0,           # 缩放因子
    width_per_layer=30,  # 每层最大宽度
    style='dark'         # 样式：'dark' 或 'light'
)
```

#### 第四步：绘制电路
```python
# 绘制到屏幕
fig = drawer.draw(circuit, title="My Quantum Circuit")

# 或保存到文件
drawer.draw(circuit, filename="circuit.png", title="Bell State")
```

### 2.2 详细使用示例

#### 例子 1：简单电路绘制

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

# 创建 2 比特电路
qreg = Register("q", 2)
circuit = GateSequence(qreg, name="simple_circuit")

# 添加门
circuit.h(qreg[0])
circuit.h(qreg[1])
circuit.cnot(qreg[0], qreg[1])

# 绘制
drawer = CircuitDrawer()
figure = drawer.draw(circuit, title="Simple Circuit", filename="simple.png")
```

#### 例子 2：Bell 态电路

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

# 创建 Bell 态电路
qreg = Register("q", 2)
circuit = GateSequence(qreg, name="Bell State")

circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])

# 使用紧凑布局绘制
drawer = CircuitDrawer(width_per_layer=50, style='light')
drawer.draw(circuit, title="Bell State Circuit", compact=True)
```

#### 例子 3：复杂的多比特电路

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

# 创建 Grover 式电路
qreg = Register("q", 4)
circuit = GateSequence(qreg, name="Grover Circuit")

# 初始化层
for i in range(4):
    circuit.h(qreg[i])

# Oracle
circuit.mcx([qreg[0], qreg[1], qreg[2]], qreg[3])

# Diffusion
for i in range(4):
    circuit.h(qreg[i])
    circuit.x(qreg[i])

circuit.h(qreg[3])
circuit.mcx([qreg[0], qreg[1], qreg[2]], qreg[3])
circuit.h(qreg[3])

for i in range(4):
    circuit.x(qreg[i])
    circuit.h(qreg[i])

# 绘制
drawer = CircuitDrawer(width_per_layer=40)
drawer.draw(circuit, title="Grover Algorithm", filename="grover.png")
```

#### 例子 4：参数化电路

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register
import numpy as np

# 创建参数化电路
qreg = Register("q", 2)
circuit = GateSequence(qreg)

# 应用参数化门
angles = [np.pi/4, np.pi/3]
circuit.rx(angles[0], qreg[0])
circuit.rz(angles[1], qreg[1])
circuit.cnot(qreg[0], qreg[1])

# 绘制（参数会自动格式化显示）
drawer = CircuitDrawer(style='dark')
drawer.draw(circuit, title="Parameterized Circuit")
```

#### 例子 5：不同样式

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence

# 创建电路
circuit = GateSequence(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.x(2)

# 暗色样式
drawer_dark = CircuitDrawer(style='dark')
drawer_dark.draw(circuit, title="Dark Style", filename="circuit_dark.png")

# 亮色样式
drawer_light = CircuitDrawer(style='light')
drawer_light.draw(circuit, title="Light Style", filename="circuit_light.png")

# 自定义样式
custom_style = {
    'backgroundcolor': '#1a1a1a',
    'fontsize': 12,
    'gatefacecolor': '#00ff00'
}
drawer_custom = CircuitDrawer(style=custom_style)
drawer_custom.draw(circuit, title="Custom Style")
```

### 2.3 高级场景

#### 场景 1：大规模电路折叠

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence

# 创建大型电路
circuit = GateSequence(20)
for i in range(100):
    circuit.h(i % 20)
    circuit.cnot(i % 20, (i + 1) % 20)

# 使用较小的 width_per_layer 强制折叠
drawer = CircuitDrawer(width_per_layer=20, scale=0.8)
drawer.draw(circuit, filename="large_circuit.png", compact=True)
```

#### 场景 2：电路批量保存

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence

def save_circuit_family(n_qubits_list, output_dir='./circuits'):
    """保存一系列不同大小的电路"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for n in n_qubits_list:
        circuit = GateSequence(n)
        for i in range(n):
            circuit.h(i)
        
        drawer = CircuitDrawer()
        filename = os.path.join(output_dir, f"circuit_{n}qubits.png")
        drawer.draw(circuit, filename=filename, title=f"{n}-Qubit Circuit")
        print(f"Saved: {filename}")

# 使用
save_circuit_family([2, 3, 4, 5, 6])
```

#### 场景 3：自定义中文标题和标签

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

# 创建中文寄存器
qreg_data = Register("数据", 2)
qreg_work = Register("工作", 1)

circuit = GateSequence(qreg_data, qreg_work)
circuit.h(qreg_data)

# 绘制（自动使用中文字体）
drawer = CircuitDrawer()
drawer.draw(circuit, title="量子傅里叶变换电路", filename="qft_cn.png")
```

---

## 三、关键 API 参考

### 3.1 构造函数

```python
CircuitDrawer(scale=1.0, ax=None, width_per_layer=30, style='dark')
```

**参数：**
- `scale: float` - 整体缩放因子（默认 1.0）
- `ax: matplotlib.axes.Axes` - 可选的 matplotlib 轴对象
- `width_per_layer: float` - 每层门的最大宽度（默认 30）
- `style: str | dict` - 样式配置（'dark'、'light'、JSON 文件路径或字典）

### 3.2 主要方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `draw()` | circuit, filename, title, compact | Figure | 绘制并返回/保存电路 |
| `get_layered_gates()` | gate_list | - | 标准门层折叠 |
| `compact_get_layered_gates()` | gate_list | - | 紧凑门层折叠 |
| `get_text_width()` | text | float | 计算文本宽度 |

### 3.3 draw() 方法详解

```python
draw(gate_sequence: GateSequence, filename=None, title=False, compact=True) -> Figure
```

**参数：**
- `gate_sequence` - 要绘制的 GateSequence 对象
- `filename` - 输出文件名（如 'circuit.png'），不指定则显示
- `title` - 电路标题（默认使用电路名称）
- `compact` - 是否使用紧凑布局（默认 True）

**返回：**
- Matplotlib Figure 对象

**示例：**
```python
# 显示到屏幕
fig = drawer.draw(circuit, title="My Circuit")

# 保存到文件
drawer.draw(circuit, filename="output.png", title="Saved Circuit")
```

---

## 四、样式配置详解

### 4.1 预定义样式

```python
# 暗色主题
drawer = CircuitDrawer(style='dark')

# 亮色主题
drawer = CircuitDrawer(style='light')

# 从 JSON 文件加载
drawer = CircuitDrawer(style='my_style.json')

# JSON 字符串格式
json_string = '{"backgroundcolor": "#ffffff", "fontsize": 14}'
drawer = CircuitDrawer(style=json_string)
```

### 4.2 自定义样式参数

```python
custom_style = {
    # 颜色配置
    'backgroundcolor': '#1a1a1a',      # 背景色
    'linecolor': '#666666',            # 线条色
    'gatetextcolor': 'white',          # 门文本色
    'textcolor': '#999999',            # 常规文本色
    'subtextcolor': '#cccccc',         # 小文本色
    
    # 尺寸配置
    'fontsize': 11,                    # 主字体大小
    'subfontsize': 8,                  # 小字体大小
    'gatewidth': 0.9,                  # 门最小宽度
    'scale_char': 0.15,                # 字符缩放因子
    
    # 线宽配置
    'lwidth1': 0.5,                    # 细线
    'lwidth15': 1.0,                   # 中细线
    'lwidth2': 1.5,                    # 中线
    'lwidth3': 2.0,                    # 粗线
    'lwidth4': 2.5,                    # 最粗线
    
    # 布局配置
    'x_offset': 1.0,                   # x 方向偏移
    'y_offset': 0.5,                   # y 方向偏移
    'margin': [1.5, 1.5, 1.5, 1.5],   # [左, 右, 下, 上]
    'figwidth': 10.0,                  # 图形宽度
    'dpi': 150,                        # 分辨率
    
    # 门文本映射
    'displaytext': {
        'h': 'H',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        # ... 更多映射
    },
    
    # 显示颜色映射
    'displaycolor': {
        'target': ['#1F52F0', 'white'],
        'control': ['#666666', 'white'],
    }
}

drawer = CircuitDrawer(style=custom_style)
```

---

## 五、布局算法详解

### 5.1 标准折叠算法

```
流程：
1. 遍历所有门
2. 对每个门计算宽度
3. 如果 (当前x + 门宽) > 限制:
   - 转到下一层
   - 重置 x 坐标
4. 放置门
5. 更新 x 坐标
```

**复杂度：**
- 时间：O(n × m)（n 个门，m 个层）
- 空间：O(n)

### 5.2 紧凑折叠算法

```
流程：
1. 为每个比特维护一个"光标"(layer, x)
2. 对每个门:
   a. 确定涉及的比特范围
   b. 找出这些比特中最晚的层
   c. 计算该层中最远的 x 位置
   d. 如果是测量门，对齐所有比特的光标
   e. 更新涉及比特的光标
```

**优势：**
- 更紧凑的布局
- 测量门自动对齐
- 无依赖关系的门可并行放置

### 5.3 坐标计算

```python
# 单比特门位置
gate['x'] = current_x              # 左上角
gate['y'] = y_base - qmax - y_offset

# 多比特门位置
gate['x'] = current_x
gate['y'] = y_base - qmax - y_offset
height = qmax - qmin + 1

# 折叠后的 y 坐标调整
y_base = -(layer * (n_qubits + 1))
```

---

## 六、常见模式

### 模式 1：快速绘制

```python
def quick_draw(circuit, title="Circuit"):
    """快速绘制电路"""
    from drawer.circuit_drawer import CircuitDrawer
    
    drawer = CircuitDrawer()
    return drawer.draw(circuit, title=title)
```

### 模式 2：批量导出

```python
def export_circuits(circuits_dict, output_dir='./outputs'):
    """导出多个电路"""
    import os
    from drawer.circuit_drawer import CircuitDrawer
    
    os.makedirs(output_dir, exist_ok=True)
    drawer = CircuitDrawer()
    
    for name, circuit in circuits_dict.items():
        filename = os.path.join(output_dir, f'{name}.png')
        drawer.draw(circuit, filename=filename, title=name)
```

### 模式 3：样式预设

```python
def get_publication_style():
    """高质量出版样式"""
    return {
        'backgroundcolor': 'white',
        'fontsize': 12,
        'figwidth': 12.0,
        'dpi': 300,
        'lwidth2': 1.2,
    }

def get_presentation_style():
    """演示文稿样式"""
    return {
        'backgroundcolor': '#2B2B2B',
        'fontsize': 14,
        'figwidth': 10.0,
        'dpi': 150,
    }
```

---

## 七、性能优化

### 7.1 大规模电路优化

```python
# 对大电路使用紧凑布局
drawer = CircuitDrawer(
    width_per_layer=25,    # 更小的宽度强制更多折叠
    style='light'          # 亮色可能更快
)
drawer.draw(circuit, compact=True)

# 降低 DPI 以提高渲染速度
custom_style = {'dpi': 72}  # 屏幕预览
drawer = CircuitDrawer(style=custom_style)
```

### 7.2 文本宽度缓存

```python
# 内部已实现文本宽度缓存
# 使用 pylatexenc 库高效计算
```

---

## 八、错误处理指南

### 8.1 常见错误

**错误 1：缺少电路对象**
```python
# 错误！
drawer = CircuitDrawer()
drawer.draw(None)  # TypeError

# 正确
from core.GateSequence import GateSequence
circuit = GateSequence(3)
drawer.draw(circuit)
```

**错误 2：无效的样式字符串**
```python
# 错误！
drawer = CircuitDrawer(style='invalid_style')

# 正确
drawer = CircuitDrawer(style='dark')  # 或 'light'
```

**错误 3: 文件覆盖**
```python
# 需要检查文件是否存在再保存
import os
filename = "circuit.png"

if os.path.exists(filename):
    print(f"文件 {filename} 已存在")
else:
    drawer.draw(circuit, filename=filename)
```

---

## 九、完整工作流示例

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

def complete_visualization_workflow():
    """完整的量子电路可视化工作流"""
    
    # 步骤 1: 创建量子电路
    print("步骤 1: 创建量子电路")
    qreg = Register("q", 4)
    circuit = GateSequence(qreg, name="Complex Circuit")
    
    # 步骤 2: 构建电路
    print("步骤 2: 添加量子门")
    for i in range(4):
        circuit.h(qreg[i])
    
    for i in range(3):
        circuit.cnot(qreg[i], qreg[i+1])
    
    circuit.rx(1.57, qreg[0])
    circuit.rz(3.14, qreg[3])
    
    # 步骤 3: 创建绘制器
    print("步骤 3: 配置绘制参数")
    drawer = CircuitDrawer(
        style='dark',
        width_per_layer=40,
        scale=1.2
    )
    
    # 步骤 4: 绘制到屏幕
    print("步骤 4: 预览电路")
    fig = drawer.draw(
        circuit,
        title="Complex Quantum Circuit",
        compact=True
    )
    
    # 步骤 5: 保存到文件
    print("步骤 5: 保存电路图")
    drawer.draw(
        circuit,
        filename="complex_circuit.png",
        title="Complex Quantum Circuit"
    )
    
    print("完成！电路图已保存")

# 执行
complete_visualization_workflow()
```

---

## 十、总结检查清单

使用 CircuitDrawer 时，请确保：

- [ ] 已正确导入 CircuitDrawer 和 GateSequence
- [ ] 创建了有效的量子电路
- [ ] 选择了合适的样式（dark/light 或自定义）
- [ ] 设置了适当的 width_per_layer 值
- [ ] 如果需要特定的布局，选择了正确的折叠方式
- [ ] 指定了有意义的电路标题
- [ ] 如果要保存文件，指定了正确的文件名和路径
- [ ] 检查了文件覆盖问题
- [ ] 对大规模电路使用了紧凑布局
- [ ] 必要时自定义了样式以满足发表或演示需求

