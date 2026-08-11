---
name: "vqls"
description: "Variational Quantum Linear Solver for caller-provided power-of-two linear systems. Documents the current A-and-b interface, three implemented cost paths, source-verified return contract, execution limits, and runnable examples."
---

# VQLS

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the VQLS code in `unitarylab_algorithms/linear_algebra/vqls`.

When using this skill:
- **Explanation:** Describe only behavior implemented by the current source. Do not claim convergence, accuracy, Hermiticity, invertibility, or finite outputs that the code does not validate.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Start with a small NumPy `A` that has a power-of-two dimension and a nonzero matching `b`. Compare the observed output with the actual return keys and exception paths below.
- **Modification or reimplementation:** Follow `VQLSAlgorithm.run()` in `unitarylab_algorithms/linear_algebra/vqls/algorithm.py`. Do not restore the removed `n_qubits`, `coefficients`, `max_iterations`, `tolerance`, or `initial_spread` interface.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/vqls_implementation.py` as reference-only material. If either conflicts with `algorithm.py`, follow `algorithm.py`.
- **Validation:** Record the selected cost path, matrix dimension, inferred qubit count, optimizer budget, SciPy success flag from logs, returned `status`, possible `NaN` values, and generated files.

## Overview

`VQLSAlgorithm.run()` receives a caller-provided coefficient matrix `A` and right-hand-side vector `b`. It converts `A` to complex dtype without normalizing it, flattens and normalizes `b`, infers `n_qubits` from the matrix dimension, Pauli-decomposes `A`, builds a hardware-efficient Ansatz, and minimizes one of three implemented cost functions with SciPy COBYLA.

The returned quantum solution is the normalized state produced by the optimized Ansatz. The classical reference, when `np.linalg.solve()` succeeds, is computed from `A` and normalized `b`, then normalized again. The source does not guarantee optimizer convergence or agreement between the quantum and classical states.

---

## Reference Implementation Example

```python
import numpy as np

from unitarylab_algorithms.linear_algebra.vqls.algorithm import VQLSAlgorithm

A = np.array(
    [
        [1.5, 0.2],
        [0.2, 1.8],
    ],
    dtype=complex,
)
b = np.array([1.0, 0.5], dtype=complex)

algo = VQLSAlgorithm(text_mode="plain")
result = algo.run(
    A=A,
    b=b,
    cost_function="local_classical",
    n_layers=4,
    maxiter=500,
    tol=1e-6,
    seed=42,
)

print(result["status"])
print(result["Fidelity"])
print(result["Ax Fidelity"])
print(result["Solution State (Quantum)"])
```

Adjust only parameters present in the current source signature:

```python
run(
    A,
    b,
    cost_function="local_ht",
    n_layers=4,
    maxiter=500,
    tol=1e-6,
    seed=42,
    epsilon=None,
    backend="torch",
    device="cpu",
    dtype=np.complex128,
)
```

> **Do not** call the removed fixed-structure interface. The current `run()` has no `n_qubits`, `coefficients`, `max_iterations`, `tolerance`, or `initial_spread` parameters.

## Core Parameters Explained

| Parameter | Default | Description | Input Info |
|---|---|---|---|
| `A` | required | Caller-provided coefficient matrix. `run()` reads `A.shape` before converting it to a complex NumPy array. It must then be square and have power-of-two dimension. It is not normalized. | Pass a NumPy array or another object that already has `.shape`; a plain nested list fails before conversion. |
| `b` | required | Caller-provided right-hand side. It is converted to a flat complex array and normalized as `b_state = b / ||b||`. | Its length must equal `A.shape[0]`, and its norm must be at least `1e-12`. |
| `cost_function` | `"local_ht"` | Selects `"local_ht"`, `"local_classical"`, or `"global"`. | Any other value raises `ValueError`. |
| `n_layers` | `4` | Controls the number of Ansatz layers and the parameter count `2 * n_qubits * n_layers`. | `run()` does not validate its type or range. |
| `maxiter` | `500` | Passed to SciPy as COBYLA option `maxiter`. | `run()` does not validate its type or range. |
| `tol` | `1e-6` | Passed to SciPy as COBYLA option `tol`. | `run()` does not validate its type or range. |
| `seed` | `42` | Seeds `np.random.default_rng()` for initialization in `[-0.5, 0.5]`. | Must be accepted by NumPy's RNG constructor; `run()` adds no separate validation. |
| `epsilon` | `None` | For a local cost, computes `gamma_stop = (1 / n_qubits) * (float(epsilon) / kappa)**2`. | No range validation; ignored by the global-cost threshold path. |
| `backend` | `"torch"` | Passed to explicit Hadamard-test circuit execution. | Used by `"local_ht"`; not forwarded to Ansatz or `U_b` calls to `get_matrix()`. |
| `device` | `"cpu"` | Passed to explicit Hadamard-test circuit execution. | Used by `"local_ht"` only. |
| `dtype` | `np.complex128` | Passed to explicit Hadamard-test circuit execution. | Used by `"local_ht"` only. |

- `A` must be two-dimensional and square after conversion.
- `A.shape[0]` must satisfy `2**int(log2(A.shape[0])) == A.shape[0]`.
- An empty `0 × 0` matrix fails while converting `log2(0)` to `int`.
- A `1 × 1` matrix passes the power-of-two check with `n_qubits=0`, but all three cost paths fail later; the implementation does not successfully support this case.
- The code does not validate finite values, Hermiticity, invertibility, condition-number bounds, or parameter ranges.

> **Summary**: The current algorithm receives `A` and `b`, normalizes only `b`, Pauli-decomposes `A`, optimizes an Ansatz with one of three cost paths, computes post-processing fidelities, exports one circuit and one text file, and returns the exact fields listed below. Any stronger correctness or convergence claim is not enforced by the source.

---

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` on every normal return because VQLS calls `_build_return_dict(True, ...)`, regardless of SciPy's `result.success`. |
| `Fidelity` | Python `float` or `None` | `_fidelity(x_cl, x_quantum)`. It can be `NaN` because `_fidelity()` has no zero-norm or finite-value guard. It is `None` when the caught classical solve raises `np.linalg.LinAlgError`. |
| `Ax Fidelity` | Python `float` | `_fidelity(b_state, Ax_norm)`. It can be `NaN` when `A @ x_quantum` is zero or non-finite. |
| `Cost Function` | `str` | The selected cost-function name. |
| `Condition Number` | Python `float` | `float(np.linalg.cond(A))`; not guaranteed finite. |
| `Solution State (Quantum)` | `np.ndarray` | Ansatz state divided by its norm without a zero/finite guard. |
| `Solution State (Classical)` | `np.ndarray` or `None` | `np.linalg.solve(A, b_state)` divided by its norm; `None` on the caught `LinAlgError`. |
| `Computation Time (s)` | Python `float` | Time from after input logging through the return of `minimize()`; post-processing and export are excluded. |
| `Cost History` | `list` of Python `float` | One value per optimizer objective call. The separate pre-optimization `c0` call is not appended. |
| `Early Stopped` | `bool` | Whether a tracked local-cost call reached `gamma_stop`; it does not mean COBYLA was stopped. |
| `circuit_path` | `str` | Path returned by `save_circuit()`. |
| `plot` | `list[dict[str, str]]` | Text-file metadata, currently `[{"format": "txt", "filename": "vqls_algorithm_result.txt"}]`. |
| `circuit` | `Circuit` | The selected circuit after `decompose(n=2)`. |

There is no `file_path`, `Relative Error`, or `Residual Norm` key in the current return dictionary.

## Implementation Architecture

- Main class: `VQLSAlgorithm`
- Source signature: `run(A, b, cost_function="local_ht", n_layers=4, maxiter=500, tol=1e-6, seed=42, epsilon=None, backend="torch", device="cpu", dtype=np.complex128)`
- Base return builder: `unitarylab_algorithms/algo_base.py::_build_return_dict`
- Pauli decomposition: `unitarylab.library.pauli_operator.pauli_string_decomposition`
- Optimizer call: `minimize(..., method="COBYLA", options={"maxiter": maxiter, "tol": tol, "rhobeg": 0.5})`

1. **Input recording and validation**: Read `A.shape`, log the original `b` norm, convert inputs, validate shapes and cost name, normalize only `b`, and compute `kappa = cond(A)`.
2. **Pauli decomposition**: Call `pauli_string_decomposition(A, partition_commuting=True, real_symmetric_hint=is_real_sym)`, where `is_real_sym = np.allclose(A, A.T) and np.allclose(A.imag, 0)`.
3. **Ansatz and cost construction**: Build `2 * n_qubits * n_layers` parameters and select the local Hadamard-test, local matrix, or global path.
4. **COBYLA execution**: Evaluate an untracked initial `c0`, then record every objective call in `Cost History`. The `epsilon` flag does not interrupt the optimizer.
5. **Post-processing and export**: Build the normalized Ansatz state, attempt the classical solve, calculate two fidelities, select and decompose a representative circuit, save SVG and text output, and build the result dictionary.

---

1. Convert `A` and `b` to complex arrays after input logging.
2. Infer `n_qubits = int(log2(A.shape[0]))`; no `n_qubits` argument is accepted.
3. Normalize `b` but not `A`.
4. Decompose `A` into retained Pauli strings and coefficients.
5. Optimize the selected source-defined cost with COBYLA.
6. Compute normalized quantum and classical state outputs.
7. Export a representative circuit and text result.
8. Return base metadata plus the nine VQLS-specific output fields.

---

## Mathematical Deep Dive

The global cost path computes:

```text
Ax = A @ x(theta)
C_global = 1.0,                                           if ||Ax|| < 1e-12
C_global = 1 - |<b_state| Ax / ||Ax||>|^2,               otherwise
```

For both local paths, let `P_l` be the matrix associated with retained Pauli label `l` and let `cp = coeffs[l] * conj(coeffs[lp])`. The classical local path accumulates:

```text
psi_norm += cp * <x| P_lp U_b U_b_dagger P_l |x>
mu_sum   += cp * <x| P_lp U_b Z_j U_b_dagger P_l |x>
```

The second expression is summed over every system qubit `j`. The returned local cost is:

```text
C_local = 1.0,                                                         if abs(psi_norm) < 1e-12
C_local = real(0.5 - 0.5*abs(mu_sum)/(n_qubits*abs(psi_norm))),       otherwise
```

The Hadamard-test path obtains complex `psi_norm` and `mu_j` terms from separate real and imaginary ancilla-`Z` circuit executions. For `psi_norm`, it skips `U_b_dagger`, controlled `Z`, and `U_b`. For each active `j`, it applies `U_b_dagger`, `CZ(ancilla, j)`, and `U_b` between controlled `P_l` and controlled `P_lp`.

The source does not test that the two local paths return equal values.

---

| Source Operation | Implemented Behavior |
|---|---|
| Pauli coefficient cutoff | Retains coefficients with magnitude greater than `1e-10`. |
| Pauli label order | Leftmost character is qubit 0, the least-significant qubit. |
| Classical Pauli matrix | Reverses the label before Kronecker products to preserve that qubit convention. |
| Ansatz | Initial `H` layer, then `RY`, `RZ`, and a CNOT ring for every layer when `n_qubits >= 2`. |
| Classical optimizer | COBYLA with `rhobeg=0.5`, caller-supplied `maxiter`, and caller-supplied `tol`. |

---

For every retained Pauli pair `(l, lp)`, one `"local_ht"` objective call rebuilds and executes two `psi_norm` circuits plus two circuits for every system qubit `j`. The two zero-parameter `psi_norm` circuits created during setup are not reused by the execution closures. The source contains no runtime or memory bound.

---

## Hands-On Example

- Use a NumPy `A`; a plain nested list has no `.shape` when input metadata is recorded.
- Use at least a `2 × 2` matrix; `1 × 1` reaches unsupported zero-qubit paths.
- Use `"local_classical"` when the task specifically needs the dense-matrix local implementation.
- Use `"global"` when the task specifically needs the source's global-overlap objective.
- Use `"local_ht"` when the task specifically needs the explicit ancilla-circuit path and its representative Hadamard-test export.
- Inspect both the logged SciPy `success` value and returned `status`; they represent different source variables.

## Maintenance Checklist

When updating this skill after algorithm changes:

1. Re-read `algorithm.py`, especially `run()`, the three cost builders, `_build_Ub()`, post-processing, representative-circuit selection, and `test()`.
2. Update parameter defaults, validation boundaries, cost formulas, return fields, and examples from executable source behavior.
3. Use `parameters.json` only to discover mismatches; never let it override `run()`.
4. Verify whether `epsilon` now interrupts COBYLA before describing it as true early stopping.
5. Keep this leaf skill's existing format while removing stale fixed-structure claims.

## Prerequisites

- Python objects used as `A` must expose `.shape` before `run()` converts them.
- The implementation imports `numpy`, `scipy.optimize.minimize`, and the project `unitarylab` circuit, transpiler, and Pauli-decomposition modules.
- Circuit export requires the project circuit drawer invoked by `BaseAlgorithm.save_circuit()`.
- The source adds no dependency preflight checks; import, backend, device, dtype, draw, and file-write errors can propagate.

## Understanding the Key Quantum Components

1. **Caller-provided problem**: `A` and `b` are required. `n_qubits` is inferred from `A.shape[0]`.
2. **Pauli decomposition**: Terms are retained above the `1e-10` cutoff and reordered through the commuting-group partition path. Hermiticity is not checked.
3. **Parameterized Ansatz**: The parameter count is exactly `2 * n_qubits * n_layers`; short supplied theta arrays would be zero-padded by `_build_ansatz()`, and long arrays truncated, although `run()` creates the exact count.
4. **Three cost paths**: `"local_ht"` executes ancilla circuits, `"local_classical"` evaluates dense matrices, and `"global"` directly evaluates the normalized `A @ x` overlap.
5. **COBYLA and epsilon**: `epsilon` can set a Boolean flag for local costs, but the implemented branch contains only `pass`, so optimization is not interrupted.

## Theory-to-Code Mapping

| Documented Object | Code Object or Location |
|---|---|
| Caller matrix and vector | `VQLSAlgorithm.run(A, b, ...)` |
| Number of system qubits | `int(np.log2(A.shape[0]))` |
| Pauli labels and coefficients | `pauli_string_decomposition()` result split into `pauli_labels` and `coeffs` |
| Ansatz parameter count | `_ansatz_num_params(n_qubits, n_layers)` |
| Ansatz state | `_ansatz_state(theta, n_qubits, n_layers, entangle=True)` |
| Uniform-state comparison | `np.allclose(b_state, uniform, atol=1e-10)` |
| General `U_b` | `_build_Ub()` Gram–Schmidt completion and `unitary` insertion |
| Local circuit cost | `_make_cost_local_ht()` |
| Local dense-matrix cost | `_make_cost_local_classical()` |
| Global cost | `_cost_global()` |
| Optimizer | `scipy.optimize.minimize(method="COBYLA")` |
| Quantum solution | `result["Solution State (Quantum)"]` |
| Classical reference | normalized `np.linalg.solve(A, b_state)` |
| Returned status | `_build_return_dict(True, ...)` produces `"ok"` |

## Minimal Manual Implementation

```python
import numpy as np

from unitarylab_algorithms.linear_algebra.vqls.algorithm import VQLSAlgorithm


def run_vqls(A, b, *, cost_function="local_classical"):
    A = np.asarray(A, dtype=complex)
    b = np.asarray(b, dtype=complex)

    algo = VQLSAlgorithm(text_mode="plain")
    return algo.run(
        A=A,
        b=b,
        cost_function=cost_function,
        n_layers=4,
        maxiter=500,
        tol=1e-6,
        seed=42,
    )


A = np.array([[1.5, 0.2], [0.2, 1.8]], dtype=complex)
b = np.array([1.0, 0.5], dtype=complex)
result = run_vqls(A, b)

print(result["status"])
print(result["Fidelity"])
print(result["Ax Fidelity"])
```

Note: This wrapper calls the project implementation directly. It does not recreate or idealize the VQLS algorithm, and it does not add validation guarantees absent from `VQLSAlgorithm.run()`.

## Debugging Tips

1. **Unexpected keyword errors**: Remove old arguments such as `n_qubits`, `coefficients`, `max_iterations`, `tolerance`, and `initial_spread`; supply `A` and `b`.
2. **Nested-list failure**: `run()` reads `A.shape` before `np.asarray(A)`. Convert `A` to a NumPy array before calling.
3. **Zero or small dimensions**: A zero-norm `b` raises `ValueError`; `0 × 0` and `1 × 1` matrix paths fail for the reasons documented above.
4. **Low cost but low fidelity**: The optimizer minimizes the selected cost, not either post-processing fidelity field. The source does not enforce a numerical relationship between them.
5. **`success=False` but `status="ok"`**: SciPy's flag sets `self.status` for text output, while the returned dictionary is built with a hard-coded success argument.
6. **`NaN` fidelity**: `_fidelity()` normalizes without checking zero or non-finite norms. Inspect `A @ x_quantum`, the classical solve, and input finiteness.
7. **No true early stop**: `Early Stopped=True` records threshold attainment only; the current branch does not stop COBYLA.
8. **Missing output keys**: Use `Ax Fidelity`, `Cost Function`, `Condition Number`, `Cost History`, and `Early Stopped`; do not expect `Relative Error`, `Residual Norm`, or `file_path`.
