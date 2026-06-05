---
name: "qsvt-qlsa"
description: "Quantum Singular Value Transformation (QSVT) based Linear System Solver (QLSA) implements matrix inversion through polynomial approximation, offering significant asymptotic complexity advantages."
---

# QSVT QLSA

## Purpose

Use this skill for the `QSVT QLSA` algorithm implemented in `unitarylab_algorithms/linear_algebra/qsvt_qlsa`.

## Generated Verification Script

```bash
python ./scripts/algorithm.py
```

Use this command only after generating or updating `scripts/algorithm.py` from this skill. The script is a verification artifact for testing whether the skill can produce correct runnable code; it is not copied from the source implementation.

## Source Of Truth

- Algorithm source: `unitarylab_algorithms/linear_algebra/qsvt_qlsa/algorithm.py`
- Parameter metadata: `unitarylab_algorithms/linear_algebra/qsvt_qlsa/parameters.json` or `setup.json`
- Readme notes: `unitarylab_algorithms/linear_algebra/qsvt_qlsa/README_zh.md`, `README_en.md`, or `README.md`

## Using The Provided Implementation

```python
import numpy as np
from unitarylab_algorithms.linear_algebra.qsvt_qlsa.algorithm import QSVTLinearSolverAlgorithm

A = np.array([[0.8, 0.0], [0.0, 0.4]])
b = np.array([1.0, 2.0])

algo = QSVTLinearSolverAlgorithm(text_mode="plain")
result = algo.run(A=A, b=b, epsilon=0.0001, backend='torch', device='cpu', dtype=np.complex128)
print(result)
```

Adjust the parameters according to the table below and the source `run()` signature.

## Core Parameters

| Parameter | Default | Description | Input Info |
|---|---|---|---|
| `A` | `[[0.8, 0], [0, 0.4]]` | Matrix | 请输入二维数组（矩阵），如 [[0.8, 0], [0, 0.4]] |
| `b` | `[1, 2]` | Source Term b | 请输入数组，如 [1, 2] |
| `epsilon` | `0.0001` | Solution accuracy | 请输入浮点数，如 0.0001，范围在1e-10到1之间 |

## Implementation Notes

- Main class: `QSVTLinearSolverAlgorithm`
- Run signature observed from source: `run(A, b, epsilon, backend='torch', device='cpu', dtype=np.complex128)`
- Keep `scripts/algorithm.py` as a generated verification artifact, not a direct copy of the source implementation.
- If result keys or generated output files change, update the usage example and return-field notes in this file.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | Execution status from the base return dict. |
| `Solution vector` | array-like | Solution vector returned by `QSVTSolver(A, b, epsilon)`. |
| `Scaling factor applied` | numeric | Scaling factor returned by the QSVT linear solver. |
| `Simulation time (s)` | `float` | Wall-clock time for the solver call. |
| `circuit_path` | `str` | Saved SVG circuit path. |
| `plot` | `list` | Saved output file metadata from `save_txt()`. |
| `circuit` | `Circuit` | Circuit returned by the underlying QSVT solver. |

## README-Derived Notes

# 基于 QSVT 的线性方程组求解算法详解

## 参数设置

- `A`: 线性方程组中的系数矩阵。
- `b`: 右端向量。
- `epsilon`: 目标逼近精度，用于传递给底层 QSVT 线性求解器。

> **总结**：该算法接收矩阵 `A`、向量 `b` 和目标精度 `epsilon`，并调用仓库中的 `QSVTSolver(A, b, epsilon)` 完成基于 QSVT 的线性求解流程。最终计算得到的结果包括求解得到的解向量、求解过程中使用的缩放因子、运行时间以及生成的量子电路文件。

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

1. **输入记录与参数传递**：接收 `A`、`b` 与 `epsilon`，并将其记录到算法输入信息中。
2. **调用底层 QSVT 求解器**：通过 `QSVTSolver(A, b, epsilon)` 执行实际的 QSVT 线性求解流程，返回量子线路、解向量和缩放因子。
3. **运行时间统计**：记录求解器完成所需的总时间。
4. **结果整理**：将解向量、缩放因子和仿真时间写入输出结构，并生成执行摘要。
5. **文件导出**：保存量子电路图和文本结果文件，并返回统一结果字典。

---

## 核心思想

QSVT 线性求解的核心思想，是把矩阵求逆问题转化为对奇异值的函数变换问题。若能够把矩阵 `A` 通过块编码嵌入到幺正算子中，那么就可以借助量子奇异值变换对其奇异值近似施加 `1/x` 型映射，从而在量子态上实现与 `A^{-1}` 成比例的效果。当前这个算法文件本身并不手写这些门级细节，而是把实际实现委托给 `unitarylab.library.linear_solver.QSVTSolver`。

---

## 数学原理

设矩阵的奇异值分解为
$$
A = U \Sigma V^{\dagger}.
$$
QSVT 的基本思想是：若一个块编码幺正算子能够表示矩阵 `A`，则可以通过交替相位旋转，把一个多项式函数施加到奇异值对角矩阵 `\Sigma` 上，从而得到对目标矩阵函数的逼近。对于线性方程组
$$
A x = b,
$$
关键目标是近似构造与
$$
f(x) = \frac{1}{x}
$$
对应的奇异值变换。于是可以把 `A` 的奇异值映射为其倒数，并在合适的归一化和后处理下得到与解向量成比例的量子结果态。

在当前仓库实现中，这些具体的块编码、多项式设计、相位序列生成与线路构造过程都封装在 `QSVTSolver` 中；当前 `algorithm.py` 的职责是组织输入、调用求解器，并保存返回的线路和求解结果。

---

## 算法步骤

1. 读取矩阵 `A`、向量 `b` 和误差参数 `epsilon`。
2. 调用底层 `QSVTSolver` 构造 QSVT 线性求解线路并执行相关求解流程。
3. 接收返回的量子电路、解向量和缩放因子。
4. 统计运行时间并写入输出结果。
5. 导出电路图和文本结果文件。

---

## 量子优势

| 任务 | 经典求解思路 | QSVT 量子优势 |
|------|--------------|----------------|
| 线性方程组求解 | 通常依赖矩阵分解或迭代法处理逆矩阵问题 | 可把求逆问题转写为奇异值变换问题，为高精度线性代数子程序提供统一量子框架 |

---

## 复杂度分析

从当前文件可直接确认的成本主要包括一次底层 `QSVTSolver` 调用以及电路导出开销。更细的复杂度则取决于底层 QSVT 实现中的块编码方法、多项式次数、目标精度 `epsilon` 以及矩阵条件性质。也就是说，本文件是一个轻量包装层，而真正的线路深度和资源复杂度由库内求解器决定。

---

## 应用与影响

- 可作为量子线性代数与量子机器学习任务中的线性求解子程序。
- 是把 QSVT 理论框架落到线性系统求解上的代表性接口之一。
- 适合作为进一步研究块编码、矩阵函数逼近和高级量子数值算法的入口。

## Maintenance Checklist

When updating this skill after algorithm changes:

1. Re-read `algorithm.py` and parameter metadata.
2. Update parameter defaults, constraints, return fields, and examples.
3. Preserve existing `scripts/algorithm.py` unless explicitly regenerating the verification artifact from this skill.
4. Run or dry-run the updater skill script from the workspace root.
5. Keep this leaf skill focused on usage and implementation guidance; keep category routing in the parent `SKILL.md`.
