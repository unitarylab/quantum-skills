---
name: "vqls"
description: Solve user-provided linear systems Ax = b with the UnitaryLab Variational Quantum Linear Solver, using a hardware-efficient ansatz and selectable local Hadamard-test, local classical, or global cost functions.
---

# VQLS

## Purpose

Use this skill for the `VQLSAlgorithm` implementation in
`unitarylab_algorithms/linear_algebra/vqls`.

The current implementation accepts a caller-provided square matrix `A` and
right-hand-side vector `b`. For dimensions that reach the main execution path,
an `N × N` system uses `n_qubits = int(log2(N))`, and the power-of-two check
requires `2**n_qubits == N`. Although `N=1` passes that check, the resulting
zero-qubit circuits fail later, so the implementation does not successfully
handle `1 × 1` systems.

`A` is converted to complex dtype but is not normalized. `b` is flattened,
converted to complex dtype, and normalized to `b_state = b / ||b||`. The
returned quantum solution is the normalized state produced by the optimized
Ansatz. The source does not guarantee that optimization converges or that this
state approximates the exact solution.

## Source Of Truth

- Runtime behavior and public signature:
  `unitarylab_algorithms/linear_algebra/vqls/algorithm.py`
- Public exports:
  `unitarylab_algorithms/linear_algebra/vqls/__init__.py`
- UI metadata:
  `unitarylab_algorithms/linear_algebra/vqls/parameters.json`

When these files disagree, follow the `VQLSAlgorithm.run()` implementation in
`algorithm.py`.

## Quick Start

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

The example uses `"local_classical"` to avoid the explicit Hadamard-test
execution path. The default `"local_ht"` mode constructs and executes
statevector circuits for every required real and imaginary expectation value.

## Run Signature

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

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `A` | required | Object with a usable `.shape` before validation. It is later converted to a complex NumPy array and must be square with power-of-two dimension. It is not normalized. |
| `b` | required | Nonzero right-hand-side vector with length equal to `A.shape[0]`. It is flattened, converted to complex dtype, and normalized internally. |
| `cost_function` | `"local_ht"` | Cost evaluation mode: `"local_ht"`, `"local_classical"`, or `"global"`. |
| `n_layers` | `4` | Used by Python loops and by the parameter-count expression `2 * n_qubits * n_layers`. `run()` does not validate its type or range. |
| `maxiter` | `500` | Forwarded to SciPy as COBYLA option `maxiter`. `run()` does not validate it. |
| `tol` | `1e-6` | Forwarded to SciPy as COBYLA option `tol`. `run()` does not validate it. |
| `seed` | `42` | Seed used to initialize Ansatz parameters uniformly in `[-0.5, 0.5]`. |
| `epsilon` | `None` | For a local cost, converted with `float(epsilon)` and used in `gamma_stop = (1 / n_qubits) * (epsilon / kappa)**2`. It is not range-checked. No threshold is built for `"global"`. |
| `backend` | `"torch"` | Passed to the explicit circuit executions inside `"local_ht"`. |
| `device` | `"cpu"` | Passed to the explicit circuit executions inside `"local_ht"`. |
| `dtype` | `np.complex128` | Passed to the explicit circuit executions inside `"local_ht"`. Ansatz and `U_b` matrix construction call `get_matrix()` without forwarding these three arguments. |

Use the source parameter names `maxiter` and `tol`. Do not use the obsolete
names `max_iterations` or `tolerance` when calling `run()`.

## Input Validation

The implementation enforces the following:

- `A` must be a two-dimensional square matrix.
- `A.shape[0]` must be a power of two.
- `len(b)` must equal `A.shape[0]`.
- `b` must have nonzero norm.
- `cost_function` must be one of `"local_ht"`, `"local_classical"`, and
  `"global"`.

Before converting `A` with `np.asarray`, `run()` records `A.shape`; a plain
Python nested list therefore fails with `AttributeError` before the documented
matrix validation.

The implementation does not validate that:

- `A` or `b` contains only finite values;
- `A` is Hermitian, invertible, or well-conditioned;
- `n_layers`, `maxiter`, `tol`, or `epsilon` has a valid type or range.

An empty `0 × 0` matrix fails while converting `log2(0)` to `int`. A `1 × 1`
matrix passes the power-of-two test with `n_qubits=0`, but the global and local
classical paths later call `Circuit.get_matrix()` with zero qubits, while the
local Hadamard-test cost has a zero-qubit divisor. No cost mode successfully
supports this case.

A singular matrix is not rejected before cost construction or optimization.
If execution reaches `np.linalg.solve(A, b_state)` and that call raises
`np.linalg.LinAlgError`, only that exception is caught:
`Solution State (Classical)` and `Fidelity` are then set to `None`. Failures
earlier in the cost path, condition-number calculation, or export are not
converted into a result dictionary.

## Pauli Decomposition And Endianness

`run()` computes:

```python
is_real_sym = np.allclose(A, A.T) and np.allclose(A.imag, 0)
terms = pauli_string_decomposition(
    A,
    partition_commuting=True,
    real_symmetric_hint=is_real_sym,
)
```

The called project function does not check Hermiticity. It computes Pauli
coefficients, discards terms whose coefficient magnitude is not greater than
`1e-10`, and greedily reorders retained terms into commuting groups before
flattening them back into one list.

Its labels are little-endian by character position: the leftmost character is
qubit 0, the least-significant qubit. `_apply_controlled_pauli_string()`
therefore maps `label[pos]` directly to system qubit `pos`.
`_pauli_string_to_matrix()` reverses the label before applying Kronecker
products, preserving the same qubit convention in the classical local-cost
path.

## Cost-Function Modes

### `local_ht`

For every retained Pauli-term pair `(l, lp)`, this path executes two circuits
for `psi_norm` and two circuits for every system-qubit index `j`: one real
circuit and one imaginary circuit. Each circuit is rebuilt at the current
`theta`; the two zero-parameter `psi_norm` circuits created during setup are
stored in local variables but are not used by the execution closures.

Each circuit applies, in source order:

1. `H` to the ancilla and, for the imaginary part, `S_dagger`;
2. the Ansatz on the system qubits;
3. controlled `P_l`;
4. for a `mu_j` circuit only, `U_b_dagger`, `CZ(ancilla, j)`, then `U_b`;
5. controlled `P_lp`;
6. a final ancilla `H`.

The code represents the active `j` with `z_angles[j] = pi`; that exact value
takes the `qc.cz(...)` branch. For `psi_norm`, all angles are zero, so both
`U_b` segments and controlled `Z` are skipped.

Ancilla `Z` expectation values form complex values
`re + 1j * im`. With `cp = coeffs[l] * conj(coeffs[lp])`, the implementation
accumulates `psi_norm` over `(l, lp)` and `mu_sum` over `(l, lp, j)`, then
returns:

```text
1.0,                                             if abs(psi_norm) < 1e-12
real(0.5 - 0.5*abs(mu_sum)/(n_qubits*abs(psi_norm))), otherwise
```

### `local_classical`

This path builds dense matrices `P_l`, `U_b`, `U_b_dagger`, and one `Z_j` per
system qubit. For `x = x(theta)` and
`cp = coeffs[l] * conj(coeffs[lp])`, its exact accumulations are:

```text
psi_norm += cp * <x| P_lp U_b U_b_dagger P_l |x>
mu_sum   += cp * <x| P_lp U_b Z_j U_b_dagger P_l |x>
```

The second expression is summed over every `j`. It then uses the same
piecewise return expression shown for `"local_ht"`. No code checks that the
two local implementations produce equal numerical values.

### `global`

Uses

```text
C_G = 1 - |<b_state| (A|x> / ||A|x>||) |^2
```

If `||A|x>|| < 1e-12`, the function returns `1.0`. The `epsilon` threshold is
not constructed for this mode. This path does not use the Pauli terms after
they have been computed.

## Ansatz And State Preparation

The Ansatz starts with Hadamard gates on all system qubits. Each of its
`n_layers` layers then applies:

1. one `RY` rotation to every qubit;
2. one `RZ` rotation to every qubit;
3. a ring of CNOT gates when at least two qubits are present.

The number of variational parameters is:

```text
2 * n_qubits * n_layers
```

`_build_Ub()` receives the normalized `b_state`. It uses only Hadamard gates
when this exact source condition is true:

```python
uniform = np.ones(1 << n_qubits, complex) / np.sqrt(1 << n_qubits)
np.allclose(b_state, uniform, atol=1e-10)
```

The call does not override `np.allclose`'s other defaults. Consequently, the
fast path is specifically determined by this comparison to the positive,
real-valued `uniform` array; the source does not separately recognize vectors
that differ from it by a global phase.

Otherwise, the code normalizes the received vector again, places it in the
first column of a matrix, and performs Gram–Schmidt completion using
computational-basis vectors in index order. Failure to complete all columns
raises `RuntimeError`. It inserts `U` for `U_b` or `U.conj().T` for
`U_b_dagger` as a `unitary` gate and calls the project `Unroll`. Only
`NotImplementedError` from unrolling is caught; in that case the original
unitary block is returned. Other construction or unrolling exceptions
propagate.

## Algorithm Flow

1. Record input metadata, convert `A` and `b` to complex arrays, validate their
   shapes, and normalize only `b`.
2. Infer `n_qubits = log2(A.shape[0])` and compute `kappa = cond(A)`.
3. Decompose `A` into Pauli strings and complex coefficients.
4. Build the hardware-efficient Ansatz and selected cost function.
5. Initialize `2 * n_qubits * n_layers` parameters from the seeded RNG.
6. Evaluate the initial cost once as `c0`, then call:
   `minimize(..., method="COBYLA", options={"maxiter": maxiter, "tol": tol,
   "rhobeg": 0.5})`.
7. Construct the normalized variational solution state.
8. Compare it with `np.linalg.solve(A, b_state)` when the classical solve
   succeeds, then normalize that classical result.
9. Calculate `Ax Fidelity`, export a representative circuit, save text output,
   and return the standard algorithm result dictionary.

The first `c0` evaluation is logged but is not appended to `Cost History`.
During SciPy objective calls, each cost is appended. For a local cost with
non-`None` `epsilon`, `Early Stopped` becomes `True` when a tracked cost is at
most `gamma_stop`. The branch that is intended to stop contains only `pass`;
there is no callback, exception, or other interruption, so COBYLA continues.

## Return Fields

The VQLS-specific output fields are:

| Key | Type | Description |
|---|---|---|
| `Fidelity` | Python `float` or `None` | `_fidelity(x_cl, x_quantum)` after normalizing both arguments again. The helper has no zero-norm or finite-value guard, so the float can be `NaN`. It is `None` only when the caught classical solve raises `LinAlgError`. |
| `Ax Fidelity` | Python `float` | `_fidelity(b_state, Ax_norm)`. If `||A @ x_quantum|| <= 1e-12`, `Ax_norm` is left unnormalized; a zero vector causes `_fidelity` to divide by zero and can produce `NaN`. |
| `Cost Function` | `str` | Selected cost-function mode. |
| `Condition Number` | Python `float` | `float(np.linalg.cond(A))`; it is not required to be finite. |
| `Solution State (Quantum)` | `np.ndarray` | `_ansatz_state()` divides the simulated state by its norm without a zero/finite guard. |
| `Solution State (Classical)` | `np.ndarray` or `None` | `np.linalg.solve(A, b_state)`, divided by its norm without a zero/finite guard; `None` on the caught `LinAlgError`. |
| `Computation Time (s)` | Python `float` | Time from immediately after input logging to immediately after `minimize`; post-processing and file export are excluded. |
| `Cost History` | `list` of Python `float` | Cost from every optimizer objective call; the separate initial `c0` evaluation is excluded. |
| `Early Stopped` | `bool` | Whether a tracked local-cost call reached `gamma_stop`; it does not mean optimization was stopped. |

The actual base return fields are:

| Key | Actual type and value |
|---|---|
| `status` | `str`; `"ok"` on every normal return because VQLS calls `_build_return_dict(True, ...)`, regardless of `result.success`. |
| `circuit_path` | `str`; the path returned by `save_circuit()`. |
| `plot` | `list[dict[str, str]]`; for the current text export it is `[{"format": "txt", "filename": "vqls_algorithm_result.txt"}]`. |
| `circuit` | project `Circuit`; the decomposed circuit passed to `save_circuit()`. |

There is no `file_path` key in the actual return dictionary.

For `"local_ht"`, the code builds several representative circuits after
optimization but exports only `hadamard_test_re`. It chooses representative
indices from retained Pauli labels as follows: the first two non-identity
terms when at least two exist; a mixed `rep_l=0`/single-non-identity selection
when exactly one exists; otherwise indices zero. It chooses the first `Z` in
the selected `rep_l` label, otherwise the first non-identity character,
otherwise qubit zero. Empty Pauli decompositions can fail while indexing this
selection.

For `"local_classical"` and `"global"`, the exported circuit is the optimized
Ansatz. In every mode the selected circuit is transformed with
`decompose(n=2)` before saving. With the default output directory, the files
are `results/vqls/vqls_algorithm_circuit.svg` and
`results/vqls/vqls_algorithm_result.txt`. Saving or drawing failures
propagate.

## Important Implementation Notes

- `run()` accepts caller-provided `A` and `b`; it has no `n_qubits` or
  `coefficients` parameters.
- `A` is Pauli-decomposed internally with
  `pauli_string_decomposition(..., partition_commuting=True,
  real_symmetric_hint=is_real_sym)`.
- `A` is not normalized. `b`, the Ansatz state, and a successfully computed
  classical state are divided by their respective norms, but the latter two
  divisions do not guard against zero or non-finite norms.
- `epsilon` can set `Early Stopped=True` but does not interrupt COBYLA.
- SciPy's `result.success` controls `self.status` (`"success"` or `"failed"`)
  and the text file content, but the returned `status` remains `"ok"` on every
  normal return.
- The `parameters.json` defaults and names may lag behind the Python API.
  Follow the `run()` signature in `algorithm.py`.

## Maintenance Checklist

When the VQLS source changes:

1. Re-read `algorithm.py`, especially `run()`, validation, cost construction,
   output assembly, and the convenience `test()` function.
2. Update examples, parameter names/defaults, supported cost modes, and return
   fields from executable source behavior.
3. Use `parameters.json` only to discover mismatches; do not let it override
   executable behavior in `algorithm.py`.
4. Verify whether `epsilon` now truly stops optimization before describing it
   as functional early stopping.
5. Keep this leaf skill focused on using and understanding the current VQLS
   implementation.
