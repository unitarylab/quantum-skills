---
name: "qsvt-qlsa"
description: "Quantum Singular Value Transformation (QSVT) based Linear System Solver (QLSA) implements matrix inversion through polynomial approximation, offering significant asymptotic complexity advantages. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# QSVT QLSA

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement QSVT QLSA.

Use this skill for the `QSVT QLSA` algorithm implemented in `unitarylab_algorithms/linear_algebra/qsvt_qlsa`.

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Overview

The core idea of QSVT linear solving is to turn matrix inversion into a singular-value transformation problem. If the matrix `A` can be embedded into a unitary through block encoding, quantum singular value transformation can approximately apply a `1/x`-type map to its singular values, producing a quantum state proportional to the effect of `A^{-1}`. This algorithm file does not manually implement the gate-level details; it delegates the actual solve to `unitarylab.library.linear_solver.QSVTSolver`.

---

## Reference Implementation Example

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

## Core Parameters Explained

| Parameter | Default | Description | Input Info |
|---|---|---|---|
| `A` | `[[0.8, 0], [0, 0.4]]` | Matrix | Pass a 2D array such as `[[0.8, 0], [0, 0.4]]`. |
| `b` | `[1, 2]` | Source Term b | Pass an array such as `[1, 2]`. |
| `epsilon` | required | Solution accuracy | Pass a float such as `1e-4`; expected range is from `1e-10` to `1`. The public source API has no default. |

- `A`: Coefficient matrix of the linear system.
- `b`: Right-hand-side vector.
- `epsilon`: Target approximation accuracy passed to the underlying QSVT linear solver.

> **Summary**: This algorithm receives matrix `A`, vector `b`, and target accuracy `epsilon`, then calls `QSVTSolver(A, b, epsilon)` to perform the QSVT-based linear solve. The returned results include the solution vector, the scaling factor used by the solver, runtime, and generated circuit files.

---

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

## Implementation Architecture

- Main class: `QSVTLinearSolverAlgorithm`
- Run signature observed from source: `run(A, b, epsilon, backend='torch', device='cpu', dtype=np.complex128)`
- If result keys or generated output files change, update the usage example and return-field notes in this file.

1. **Input recording and parameter forwarding**: Receive `A`, `b`, and `epsilon`, then record them in the algorithm input metadata.
2. **Underlying QSVT solver call**: Execute the actual QSVT linear-solver workflow through `QSVTSolver(A, b, epsilon)`, returning the quantum circuit, solution vector, and scaling factor.
3. **Runtime measurement**: Record the total time required by the solver call.
4. **Result assembly**: Write the solution vector, scaling factor, and simulation time into the output structure, then generate the execution summary.
5. **File export**: Save the quantum circuit diagram and text result file, then return the unified result dictionary.

---

1. Read matrix `A`, vector `b`, and error parameter `epsilon`.
2. Call the underlying `QSVTSolver` to construct the QSVT linear-solver circuit and execute the solve workflow.
3. Receive the returned quantum circuit, solution vector, and scaling factor.
4. Measure runtime and write it into the output result.
5. Export the circuit diagram and text result file.

---

## Critical Theoretical Constraints

The QSVT linear-system solver is valid only when the matrix access model and polynomial transform satisfy the theoretical requirements below. Violating these assumptions can produce results that look numerically plausible but are not guaranteed to correspond to a realizable QSVT linear solver.

- **Block encoding:** `A` must admit an $(\alpha, a, 0)$-block encoding with subnormalization factor $\alpha \ge \|A\|$. The effective QSVT condition parameter is $K = \alpha / \sigma_{\min}(A)$, not simply the ordinary matrix condition number $\kappa(A)$.
- **Bounded polynomial:** The inverse polynomial approximation $P(x) \approx 1/(2Kx)$ must obey $|P(x)| \le 1$ for every $x \in [-1, 1]$, not only on the spectral interval $[1/K, 1]$. A polynomial that fits the singular values well but exceeds 1 in the spectral gap is not physically realizable by QSVT.
- **Post-selection probability:** The linear-solver success probability scales as $\Omega(1/\kappa_A^2)$ for the relevant scaled system. Ill-conditioned inputs may require amplitude amplification, adding an $O(\kappa_A)$ factor.
- **End-to-end complexity:** The polynomial degree $d = O(K \log(K/\epsilon))$ is the cost per block-encoding query. Full solution-state preparation with amplitude amplification costs $O(K \cdot \kappa_A \cdot \log(K/\epsilon))$ block-encoding queries. Do not report the polynomial degree alone as total query complexity.

---

## Mathematical Deep Dive

Let the singular value decomposition of the matrix be
$$
A = U \Sigma V^{\dagger}.
$$
The basic idea of QSVT is that if a block-encoded unitary represents matrix `A`, alternating phase rotations can apply a polynomial function to the singular-value diagonal matrix `\Sigma`, thereby approximating a target matrix function. For the linear system
$$
A x = b,
$$
the key target is to approximately construct the singular-value transformation corresponding to
$$
f(x) = \frac{1}{x}
$$
so that the singular values of `A` are mapped to their reciprocals. With proper normalization and postprocessing, this yields a quantum result state proportional to the solution vector.

In the current repository implementation, the block encoding, polynomial design, phase-sequence generation, and circuit construction are encapsulated inside `QSVTSolver`. The role of this `algorithm.py` wrapper is to organize inputs, call the solver, and save the returned circuit and solve results.

---

| Task | Classical Solve Pattern | QSVT Advantage |
|------|--------------|----------------|
| Linear-system solving | Usually relies on matrix factorization or iterative methods to handle inverse problems | Rewrites inversion as a singular-value transformation problem, providing a unified quantum framework for high-precision linear-algebra subroutines |

---

The directly visible cost in this wrapper mainly consists of one underlying `QSVTSolver` call plus circuit-export overhead. Finer-grained complexity depends on the block-encoding method, polynomial degree, target accuracy `epsilon`, and matrix conditioning inside the underlying QSVT implementation. This file is therefore a lightweight wrapper; the actual circuit depth and resource complexity are determined by the library solver.

The inverse-function approximation degree $d = O(K \log(K/\epsilon))$ describes each polynomial transformation per block-encoding query, where $K = \alpha / \sigma_{\min}(A)$. Complete preparation of a linear-system solution state must also account for post-selection success probability; with amplitude amplification, describe the total query scaling at the level of $O(K \cdot \kappa_A \cdot \log(K/\epsilon))$ rather than quoting polynomial degree alone as the end-to-end cost.

---

## Hands-On Example

- Useful as a linear-solver subroutine for quantum linear algebra and quantum machine-learning tasks.
- Serves as a representative interface that applies the QSVT framework to linear-system solving.
- Provides an entry point for studying block encoding, matrix-function approximation, and advanced quantum numerical algorithms.

## Prerequisites

- Linear algebra: singular value decomposition $A = U\Sigma V^\dagger$, condition number $\kappa$
- Quantum Singular Value Transformation (QSVT) theory: block encoding, polynomial approximation of matrix functions
- Understanding of the linear-system problem: $Ax = b$ → quantum state $|x\rangle \propto A^{-1}|b\rangle$
- Python: `numpy`, `unitarylab` core

## Understanding the Key Quantum Components

1. **Block encoding of $A$**: The matrix $A$ is embedded into a larger unitary $U_A$ such that $\langle 0^m|U_A|0^m\rangle = A/\alpha$, where $\alpha \geq \|A\|$ is the subnormalization factor. The QSVT framework applies polynomial transformations to the singular values of this block-encoded matrix.
2. **Polynomial approximation of $1/x$**: The target function $f(x) = 1/(2Kx)$ is approximated by a polynomial $P(x)$ of degree $d = O(K \log(K/\epsilon))$, where $K = \alpha/\sigma_{\min}(A)$ is the effective condition parameter. The polynomial must satisfy $|P(x)| \leq 1$ for all $x \in [-1, 1]$.
3. **Phase sequence generation**: Alternating phase rotations implement the Chebyshev polynomial approximation on the block-encoded singular values. The phase angles are computed from the target polynomial coefficients.
4. **Post-selection**: The solution state is obtained after post-selecting on the block-encoding ancillas being in $|0^m\rangle$ and the QSVT signal qubit being in $|1\rangle$. Success probability scales as $\Omega(1/\kappa_A^2)$.
5. **Wrapper architecture**: `QSVTLinearSolverAlgorithm` is a lightweight wrapper — it delegates the actual QSVT construction, phase-sequence design, and circuit building to `QSVTSolver` from `unitarylab.library`.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Matrix $A$ | `A` parameter — 2D `np.ndarray` |
| Vector $b$ | `b` parameter — 1D `np.ndarray` |
| Target accuracy $\epsilon$ | `epsilon` parameter — controls polynomial degree |
| QSVT solver | `QSVTSolver(A, b, epsilon)` from `unitarylab.library.linear_solver` |
| Solution vector | `result["Solution vector"]` |
| Scaling factor | `result["Scaling factor applied"]` |
| Block encoding | Handled internally by `QSVTSolver` |
| Polynomial degree $d = O(K \log(K/\epsilon))$ | Computed inside `QSVTSolver`; not directly exposed |
| Post-selection probability | Implicit in solver; $\Theta(1/\kappa_A^2)$ |
| Circuit export | `result["circuit"]` and `result["circuit_path"]` |

## Minimal Manual Implementation

```python
import numpy as np

def qsvt_skeleton(A, b, epsilon=1e-4):
    """Simplified QSVT linear solver wrapper.

    In practice, the implementation delegates to QSVTSolver from
    unitarylab.library. This skeleton shows the conceptual flow:
    block-encode A, apply polynomial ~1/x to singular values,
    extract solution state proportional to A^{-1}|b>.
    """
    # 1. Compute SVD for classical reference
    U, s, Vh = np.linalg.svd(A)
    kappa = s.max() / s.min()

    # 2. Classical solution (reference)
    x_classical = np.linalg.solve(A, b)

    # 3. In QSVT:
    #    a. Block-encode A with subnormalization alpha >= ||A||
    alpha = np.linalg.norm(A, 2)
    K = alpha / s.min()  # effective condition parameter

    #    b. Polynomial degree: d = O(K * log(K/epsilon))
    d = int(np.ceil(K * np.log(K / epsilon)))

    #    c. Build QSVT circuit with phase sequence for 1/x approximation
    #    d. Post-select on ancilla = success
    p_success = 1.0 / (kappa ** 2)  # approximate

    return {
        "classical_solution": x_classical,
        "condition_number": kappa,
        "polynomial_degree": d,
        "expected_success_probability": p_success,
    }
```

Note: This skeleton illustrates the conceptual QSVT pipeline. The actual `QSVTSolver` handles block encoding, phase-sequence optimization, circuit construction, and statevector extraction.

## Debugging Tips

1. **Ill-conditioned matrices**: When $\kappa(A) > 100$, the post-selection probability drops to $\sim 10^{-4}$ and the polynomial degree grows linearly with $\kappa$. For benchmarking, start with well-conditioned matrices ($\kappa < 10$).
2. **epsilon not directly controlling error**: The `epsilon` parameter is passed to `QSVTSolver` and influences the polynomial degree, but the achieved solution error also depends on matrix conditioning and block-encoding quality. Compare `result["Solution vector"]` against `np.linalg.solve(A, b)` for ground truth.
3. **Non-power-of-2 dimensions**: The implementation may zero-pad $A$ to the next power of 2. Verify that the solution vector length matches the original $b$ dimension — padded entries are artifacts.
4. **Scaling factor interpretation**: `result["Scaling factor applied"]` is the normalization constant from the QSVT solver. The raw solution vector must be multiplied by this factor to recover the unnormalized solution.
5. **Block-encoding constraints**: The public algorithm delegates block encoding and solving to `QSVTSolver(A, b, epsilon, backend=..., device=..., dtype=...)` from `unitarylab.library.linear_solver`; the wrapper itself does not expose a `method` parameter.
