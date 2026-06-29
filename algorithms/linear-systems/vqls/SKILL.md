---
name: "vqls"
description: "Variational Quantum Linear Solver for a fixed-structure linear system. This implementation constructs A internally as A = c₀A₀ + c₁A₁ + c₂A₂ and does not accept arbitrary user-provided A and b."
---

# VQLS

## Purpose

Use this skill for the `VQLS` algorithm implemented in `unitarylab_algorithms/linear_algebra/vqls`.

> **Important:** This implementation is a fixed-structure VQLS demo, not a general-purpose arbitrary-matrix linear solver. It does not accept arbitrary `A` and `b` as `run()` parameters. Instead, it internally constructs the linear system using `n_qubits` and `coefficients = [c0, c1, c2]`, forming `A = c_0 A_0 + c_1 A_1 + c_2 A_2`, while the right-hand side state `|b\rangle` is prepared internally.

## Source Of Truth

- Algorithm source: `unitarylab_algorithms/linear_algebra/vqls/algorithm.py`
- Parameter metadata: `unitarylab_algorithms/linear_algebra/vqls/parameters.json` or `setup.json`
- Readme notes: `unitarylab_algorithms/linear_algebra/vqls/README_zh.md`, `README_en.md`, or `README.md`

## Using The Provided Implementation

```python
import numpy as np
from unitarylab_algorithms.linear_algebra.vqls.algorithm import VQLSAlgorithm

algo = VQLSAlgorithm(text_mode="plain")
result = algo.run(n_qubits=3, coefficients=None, max_iterations=200, tolerance=1e-06, initial_spread=0.5, backend='torch', device='cpu', dtype=np.complex128)
print(result)
```

Adjust the parameters according to the table below and the source `run()` signature.

> **Do not** call `algo.run(A=A, b=b)`. This implementation does not expose arbitrary `A` or `b` inputs. The matrix `A` is always constructed internally as `A = c_0 A_0 + c_1 A_1 + c_2 A_2`, and `|b\rangle` is prepared by applying Hadamard gates to all system qubits.

## Core Parameters

| Parameter | Default | Description | Input Info |
|---|---|---|---|
| `n_qubits` | `3` | Number of system qubits. Must be greater than 0. | Use `2` or `3` for small verification runs. |
| `coefficients` | `[1.0, 0.2, 0.2]` | Coefficients `[c0, c1, c2]` used to construct the internal matrix `A = c0*A0 + c1*A1 + c2*A2`. This is **not** an arbitrary matrix input — `A0, A1, A2` are fixed Pauli-structured operators generated from `n_qubits`. | 请输入列表（如 [1.0, 0.2, 0.2]） |
| `max_iterations` | `200` | Max iterations | 请输入正整数，如 200 |
| `tolerance` | `1e-6` | Convergence tolerance | 请输入浮点数，如 1e-6 |
| `initial_spread` | `0.5` | Random initialization range for variational parameters. | Use a positive float. |

## Implementation Notes

- Main class: `VQLSAlgorithm`
- Run signature observed from source: `run(n_qubits=3, coefficients=None, max_iterations=200, tolerance=1e-06, initial_spread=0.5, backend='torch', device='cpu', dtype=np.complex128)`
- If result keys or generated output files change, update the usage example and return-field notes in this file.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | Execution status from the base return dict. |
| `Fidelity` | `float` | Fidelity between the normalized variational solution and classical reference solution. |
| `Relative Error` | `float` | Residual norm divided by `||b||`. |
| `Residual Norm` | `float` | Norm of `A @ x_quantum - b`. |
| `Solution State (Quantum)` | array-like | Normalized solution state produced by the optimized ansatz. |
| `Solution State (Classical)` | array-like | Normalized classical reference solution. |
| `Computation Time (s)` | `float` | Time spent in COBYLA optimization. |
| `circuit_path` | `str` | Saved SVG circuit path for the visualization circuit. |
| `plot` | `list` | Saved output file metadata from `save_txt()`. |
| `circuit` | `Circuit` | Example circuit built for visualization. |

## README-Derived Notes

# 变分量子线性求解器 (VQLS)

## 参数设置

- `coefficients`: 线性组合系数 `[c_0, c_1, c_2]`，默认值为 `None`。当不显式提供时，代码使用默认系数 `[1.0, 0.2, 0.2]` 来构造矩阵 `A = c_0 A_0 + c_1 A_1 + c_2 A_2`。
- `max_iterations`: 最大优化迭代次数，默认值为 `200`。
- `tolerance`: COBYLA 优化的收敛容差，默认值为 `1e-6`。

> **总结**：该算法使用变分量子线路求解特定结构的线性系统，先构造问题矩阵和右端态，再通过局部 Hadamard 测试定义损失函数，并用 COBYLA 优化参数化 Ansatz。最终计算得到的结果包括量子解与经典解的保真度、相对误差、残差范数、两种解态以及生成的量子电路文件。

---

## 目录

- [运行流程](#运行流程)
- [核心思想](#核心思想)
- [数学原理](#数学原理)
- [算法步骤](#算法步骤)
- [量子优势](#量子优势)
- [复杂度分析](#复杂度分析)
- [应用与影响](#应用与影响)

---

## 运行流程

1. **参数准备与问题初始化**：检查 `n_qubits` 是否有效，设置系统量子位、辅助位和线性组合系数，并根据量子位数构造问题矩阵 `A_num`。
2. **变分线路构造**：生成 `|b>` 的制备线路 `U_b`、参数化 Ansatz 线路，以及用于展示的局部 Hadamard 测试示例线路。
3. **变分优化执行**：以随机初始化参数为起点，使用 COBYLA 最小化局部损失函数 `_cost_loc`，得到最优参数与最终损失。
4. **经典后处理与精度分析**：根据最优参数提取量子解态，同时计算经典精确解，并评估保真度、相对误差和残差范数。
5. **结果导出**：保存示例量子电路图、文本结果文件，并返回统一结果字典。

---

## 核心思想

VQLS 的核心思想是用一个参数化量子线路去表示候选解态 `|x(theta)>`，再通过专门设计的代价函数衡量 `A|x(theta)>` 与目标态 `|b>` 的接近程度。只要这个代价函数被压低到足够小，就说明当前参数对应的量子态已经接近线性方程组的解。与直接构造矩阵逆不同，VQLS 用“量子线路表示 + 经典优化器调参”的方式，把求解问题转化成混合量子经典优化任务。

---

## 数学原理

VQLS 使用变分量子算法求解线性方程组 `Ax = b`。

> **关键约束：** 当前实现中的 VQLS **不是通用矩阵接口**。用户**不能直接传入任意矩阵 `A` 或向量 `b`**。代码会根据 `n_qubits` 生成固定结构的算子项 `A_0, A_1, A_2`，再通过 `coefficients = [c0, c1, c2]` 组合得到问题矩阵：
$$
A = c_0 A_0 + c_1 A_1 + c_2 A_2,
$$
其中 `A_0` 是单位矩阵，而 `A_1`、`A_2` 由代码根据 `n_qubits` 生成对应的 Pauli 结构；右端态 `|b\rangle` 则通过对每个系统量子位施加 Hadamard 门得到，同样不接受外部输入。

VQLS 的关键是定义局部代价函数。代码通过局部 Hadamard 测试估计复数系数
$$
\mu_{l,l',j},
$$
并进一步计算
$$
\langle \psi | \psi \rangle
$$
与局部损失
$$
C_L = \frac{1}{2} - \frac{1}{2} \cdot \frac{|\sum_{l,l',j} c_l c_{l'}^* \mu_{l,l',j}|}{n\, \langle \psi | \psi \rangle}.
$$
当该损失趋近于零时，说明参数化解态已经较好满足 `A|x> ≈ |b>`。代码使用 COBYLA 迭代优化参数，并在优化结束后，将量子解态与经典解 `np.linalg.solve(A_num, b_state)` 进行对比。

---

## 算法步骤

1. 根据 `n_qubits` 和 `coefficients` 构造问题矩阵与目标态 `|b>`。
2. 搭建 Ansatz、受控 `A_l` 操作和局部 Hadamard 测试线路。
3. 用 COBYLA 优化变分参数，最小化局部损失函数。
4. 提取量子解态，并计算经典参考解。
5. 输出保真度、相对误差、残差范数和电路文件。

---

## 量子优势

| 任务 | 经典方法 | VQLS 优势 |
|---|---|---|
| 线性方程组求解 | 直接求逆或分解矩阵往往需要完整经典表示 | 可把求解过程转化为适合近似量子态制备和混合优化的问题，适合 NISQ 风格的变分框架 |

---

## 复杂度分析

该实现的成本主要由三部分决定：局部 Hadamard 测试线路的调用次数、每次量子态仿真的开销，以及 COBYLA 优化的迭代次数 `max_iterations`。由于损失函数在每次迭代中都要对多个 `l, l', j` 组合求值，量子线路评估次数会随系统量子位数和算子项数增长。因此，VQLS 的优势在于可变分近似求解，但代价是需要较多重复测量和经典优化回合。

---

## 应用与影响

- 适用于研究 NISQ 条件下的混合量子经典线性求解方法。
- 可作为变分量子算法、局部 Hadamard 测试和量子线性代数教学示例。
- 为进一步扩展到更一般的问题哈密顿量和参数化 Ansatz 提供基础框架。

## Maintenance Checklist

When updating this skill after algorithm changes:

1. Re-read `algorithm.py` and parameter metadata.
2. Update parameter defaults, constraints, return fields, and examples.
3. Run or dry-run the updater skill script from the workspace root.
4. Keep this leaf skill focused on usage and implementation guidance; keep category routing in the parent `SKILL.md`.
5. Confirm whether the implementation still uses the fixed structure `A = c_0 A_0 + c_1 A_1 + c_2 A_2` or has been upgraded to accept arbitrary `A` and `b`. If upgraded, update the frontmatter description, Purpose notice, and all fixed-structure warnings accordingly.
