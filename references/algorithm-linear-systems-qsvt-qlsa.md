# QSVT QLSA

## Scope and Reference Use

This reference documents `QSVTLinearSolverAlgorithm` in `unitarylab_algorithms/linear_algebra/qsvt_qlsa` and its delegation to the SDK solver.

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

Wrapper source: `unitarylab_algorithms/linear_algebra/qsvt_qlsa/algorithm.py`.

Reference scripts: [qsvt_qlsa_algorithm.py](../scripts/qsvt_qlsa_algorithm.py) (API usage example); [qsvt_qlsa_qsvt_implementation.py](../scripts/qsvt_qlsa_qsvt_implementation.py) (independent teaching implementation).

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
| `A` | required | Coefficient matrix | Pass a 2D array such as `[[0.8, 0], [0, 0.4]]`; the example matrix is not a `run()` default. |
| `b` | required | Right-hand side | Pass a matching NumPy vector such as `np.array([1, 2])`; the example vector is not a `run()` default. |
| `epsilon` | required | Target approximation accuracy | Forwarded to the SDK; the wrapper adds no range validation or default. The SDK function itself defaults to `1e-3`. |
| `backend`, `device`, `dtype` | `'torch'`, `'cpu'`, `np.complex128` | Execution settings | Forwarded to `QSVTSolver`. |

- `A`: Coefficient matrix of the linear system.
- `b`: Right-hand-side vector.
- `epsilon`: Target approximation accuracy passed to the underlying QSVT linear solver.

> **Summary**: This algorithm receives matrix `A`, vector `b`, and target accuracy `epsilon`, then calls `QSVTSolver(A, b, epsilon)` to perform the QSVT-based linear solve. The returned results include the solution vector, the scaling factor used by the solver, runtime, and generated circuit files.

---

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on normal return; does not certify a residual or achieved `epsilon`. |
| `Solution vector` | array-like | Solution vector returned by `QSVTSolver(A, b, epsilon)`. |
| `Scaling factor applied` | numeric | Scaling factor returned by the QSVT linear solver. |
| `Simulation time (s)` | `float` | Wall-clock time for the solver call. |
| `circuit_path` | `str` | Saved SVG circuit path. |
| `plot` | `list` | Saved output file metadata from `save_txt()`. |
| `circuit` | `Circuit` | Circuit returned by the underlying QSVT solver. |

## Implementation Architecture

- Main class: `QSVTLinearSolverAlgorithm`
- Run signature: `run(A, b, epsilon, backend='torch', device='cpu', dtype=np.complex128)`

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

These are theoretical requirements and cost models, not guarantees checked or reported by this wrapper. UnitaryLab 1.5.1 exports `QSVTSolver` as a compiled extension.

- **Block encoding:** `A` must admit an $(\alpha, a, 0)$-block encoding with subnormalization factor $\alpha \ge \|A\|$. The effective QSVT condition parameter is $K = \alpha / \sigma_{\min}(A)$, not simply the ordinary matrix condition number $\kappa(A)$.
- **Bounded polynomial:** The inverse polynomial approximation $P(x) \approx 1/(2Kx)$ must obey $|P(x)| \le 1$ for every $x \in [-1, 1]$, not only on the spectral interval $[1/K, 1]$. A polynomial that fits the singular values well but exceeds 1 in the spectral gap is not physically realizable by QSVT.
- **Post-selection probability:** The linear-solver success probability scales as $\Omega(1/\kappa_A^2)$ for the relevant scaled system. Ill-conditioned inputs may require amplitude amplification, adding an $O(\kappa_A)$ factor.
- **Query accounting:** A degree-$d$ QSVT transformation requires order $d$ block-encoding queries. An ideal inverse-polynomial construction can have degree $O(K\log(K/\epsilon))$, with further state-preparation, post-selection, amplification, and readout costs. This is not the degree formula or an end-to-end bound established by the current implementation.

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

The inspected SDK constructs odd Chebyshev coefficients with `_indicator(kappa, alpha, epsilon, beta=2)`, uses `QSP(gs_be.dagger(), ...)`, extracts the ancilla-zero statevector block, and applies the solution scale before returning it. Its degree follows its own `_b_value` / `_j_0_value` formulas, not a direct assignment from the ideal big-O expression above. It does not implement amplitude amplification or return a certified residual.

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
4. **Post-selection**: The inspected SDK takes `state_arr[:2**n_sys]`, fixing its high-index ancillas to zero. It does not return the norm-squared of this block as a success-probability field.
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
| Polynomial degree | Logged as `2*len(c_chebyshev)-1`; determined by SDK coefficient helpers. |
| Post-selection probability | Not returned by this wrapper. |
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
    d = int(np.ceil(K * np.log(K / epsilon)))  # Illustrative heuristic, not the SDK formula

    #    c. Build QSVT circuit with phase sequence for 1/x approximation
    #    d. Post-select on ancilla = success
    p_success_scale = 1.0 / (kappa ** 2)  # Ideal worst-case scale, not a measured probability

    return {
        "classical_solution": x_classical,
        "condition_number": kappa,
        "polynomial_degree": d,
        "illustrative_success_scale": p_success_scale,
    }
```

Note: This skeleton illustrates the conceptual QSVT pipeline. The actual `QSVTSolver` handles block encoding, phase-sequence optimization, circuit construction, and statevector extraction.

## Debugging Tips

1. **Ill-conditioned matrices**: Conditioning can increase approximation cost and reduce success probability, but `1/kappa**2` is not the measured probability for a particular input. Start with a small well-conditioned matrix and inspect residual and direction.
2. **epsilon not directly controlling error**: The `epsilon` parameter is passed to `QSVTSolver` and influences the polynomial degree, but the achieved solution error also depends on matrix conditioning and block-encoding quality. Compare `result["Solution vector"]` against `np.linalg.solve(A, b)` for ground truth.
3. **Dimensions and singularity**: The inspected SDK infers `ceil(log2(A.shape[0]))`, but does not pad `b` to that state dimension. Do not assume arbitrary dimensions work, or zero-pad an inversion problem: that creates a singular enlarged matrix. Use a square nonsingular power-of-two problem for the documented example.
4. **Scaling factor interpretation**: The SDK first divides both `A` and `b` by `||A||_2`, then returns `final_solution = raw_solution * (norm_b * kappa * beta2)`. The wrapper's `Solution vector` is already scaled. Multiplying it by `Scaling factor applied` again would double-apply the scale.
5. **Block-encoding constraints**: The public algorithm delegates block encoding and solving to `QSVTSolver(A, b, epsilon, backend=..., device=..., dtype=...)` from `unitarylab.library.linear_solver`; the wrapper itself does not expose a `method` parameter.
6. **Polynomial validity and special conditioning**: The inspected `_indicator()` samples an interval at 100 points and rescales coefficients when its sampled maximum exceeds one; that is not a proof of boundedness on all of `[-1,1]`. Its endpoint formula includes `1/(kappa*alpha-1)`, so the `kappa*alpha≈1` regime also needs attention. Do not describe ideal polynomial feasibility as a source-enforced guarantee.
