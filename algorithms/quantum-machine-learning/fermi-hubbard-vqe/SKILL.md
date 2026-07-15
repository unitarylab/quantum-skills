---
name: fermi-hubbard-vqe
description: Guides small open-chain Fermi-Hubbard ground-state calculations with UnitaryLab's VQE workflow, including its Jordan-Wigner mapping, Ry-Rz COBYLA ansatz, exact-energy reference, endianness checks, output contract, and paired-mode spin measurements.
---

# Fermi-Hubbard VQE

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab Fermi-Hubbard VQE workflow.

This implementation estimates the ground-state energy of a small open one-dimensional Fermi-Hubbard model. It maps the `2L` fermionic modes to qubits with Jordan-Wigner, optimizes an Ry-Rz ansatz with CX entanglement using COBYLA, and compares the variational result with an exact ground-state reference.

Use this skill when you need to:
- Estimate the ground-state energy of a small open-chain Fermi-Hubbard model.
- Demonstrate Jordan-Wigner mapping, dense exact diagonalization, and VQE optimization.
- Validate qubit ordering, endianness, optimizer status, and paired-mode measurements.

When using this skill:
- **Explanation:** Explain the Hamiltonian, mode order, full-Fock-space assumption, variational bound, and dense-matrix scale limitation. Do not generate code unless the user requests it.
- **Run or reuse:** Use `FermiHubbardVQEAlgorithm().run(...)` as the public entry point. Start with `L=2`, `seed=7`, and `measure_shots=0`; do not import internal adapter code as a second public workflow.
- **Debugging:** Run the documented small example first. Check the mode order, `q0` least-significant-bit convention, bit-reversal round trip, returned energies, and optimizer fields before changing implementation code.
- **Modification or reimplementation:** Preserve the documented parameter schema, Hamiltonian sign convention, output contract, best-observed-parameter behavior, and paired-mode measurement definition.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/fermi_hubbard_vqe_implementation.py` as reference material for troubleshooting, API comparison, and validation; keep standalone reimplementations independent of the skill directory.
- **Validation:** Compare `VQE Energy` with `Exact Energy`, check `VQE Energy >= Exact Energy - 1e-8`, and report optimizer convergence separately from absolute energy error. Mention backend, dependency, and exponential scaling limitations when relevant.

## Physical Scope and Assumptions

- The model has open boundary conditions: hopping connects sites `j` and `j+1` only. The mode order is `(1↑, 1↓, 2↑, 2↓, ...)`, mapped by Jordan–Wigner to `2L` qubits.
- The exact diagonalization and VQE act on the complete Fock space of these `2L` modes (dimension `2^(2L)`), not a fixed-particle-number sector. No particle-number constraint or number-preserving initialization is imposed.
- The Ry-Rz ansatz with CX gates does not generally preserve particle number. Its state may leave every fixed-`N` sector during optimization.
- The Hamiltonian is `H=-t Σ(c†jσ c(j+1)σ+h.c.) + U Σ n(j↑)n(j↓) - B Σ(n(j↑)-n(j↓))`; the Zeeman sign is therefore fixed by `-B(n↑-n↓)`.
- UnitaryLab uses `q0` as the least-significant bit. The adapter bit-reverses the dense Hamiltonian before VQE and checks a round trip. Equal spectra prove only that this is a legal basis permutation; they do not alone prove that every qubit/bitstring convention is correct.
- The optional measurement routine is documented by its API as a total-spin measurement and uses paired-mode correlators (`XX`, `YY`, `XY`, `YX`, and `Z`), combined in the implementation as `(XX+YY)/4`, `(XY-YX)/4`, and `Z/4`. Do not replace this with the incorrect physical-spin expression `1/2 Σ X/Y/Z` over all occupation-mode qubits. The public field retains the source name `Measured Magnetic Moment`; interpret it according to this paired-mode mapping and do not invent a different observable in a manual implementation.

## Implementation Architecture

`algorithm.py` is the public workflow. It builds the fermionic and Pauli Hamiltonians, obtains the formal exact reference, calls `run_pauli_vqe`, writes the circuit/plot/parameter files, optionally measures the final circuit, and assembles the public dictionary. `vqe_adapter.py` is an internal bridge to the generic `VQEAlgorithm`: it performs matrix conversion and bit reversal, validates spectra, records objective evaluations, tracks the lowest observed parameters, rebuilds that circuit, and checks its energy. It is not a second public algorithm entry point.

## Reference Implementation

```python
from unitarylab_algorithms import FermiHubbardVQEAlgorithm

result = FermiHubbardVQEAlgorithm().run(
    L=2, t=1.0, U=4.0, B=1.5, layers=5,
    max_iter=1000, seed=7, measure_shots=10000,
    backend="torch", device="cpu",
)
print(result["VQE Energy"], result["Exact Energy"], result["status"])
```

## Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `params` | `None` | Dict or JSON string; aliases `vqe_layers`, `vqe_max_iter`, `measurement_shots` are accepted. |
| `L`, `t`, `U`, `B` | `2`, `1.0`, `4.0`, `1.5` | Sites and Hamiltonian coefficients. |
| `layers` | `5` | Ry-Rz layers with nearest-neighbor and ring-closing CX gates. |
| `max_iter` | `1000` | Passed to SciPy COBYLA as `options={"maxiter": max_iter}`: the optimizer's maximum function-evaluation budget (SciPy's `maxiter` option), not a guaranteed number of ansatz layers or recorded history points. |
| `seed` | `7` | Initial parameter seed. |
| `measure_shots` | `10000` | Shots per measurement setting; `0` skips measurement. Values below the measurement routine's minimum are invalid when enabled. |
| `backend`, `device`, `dtype` | `"torch"`, `"cpu"`, `np.complex128` | Circuit execution settings forwarded to the optimized circuit. |

## Optimization and Energy Checks

The adapter records every objective evaluation and tracks `best_energy` and `best_parameters`. It reconstructs the final circuit from those tracked parameters (the lowest-energy circuit observed during optimization), checks its energy against the tracked value, and checks `E_VQE >= E_exact - 1e-8`.

Do not infer convergence from `nfev < max_iter`. The adapter exposes `Optimizer Converged`, derived from the optimizer's returned message; use that field (and `Optimizer Message`) for optimizer termination. `Absolute Error = |E_VQE-E_exact|` is an accuracy metric, not a convergence flag. The exact energy is a reference, and the variational inequality is an upper-bound check; neither by itself proves optimizer convergence.

The current adapter derives `Optimizer Converged` from message text rather than directly returning SciPy's `opt.success`; treat it as the source's termination-status approximation. Direct `opt.success` propagation would require a source change.

## Exact Energy and Endianness

The formal workflow obtains `Exact Energy` through `pauli_ground_state(pauli_expression)`. The adapter additionally uses dense `np.linalg.eigvalsh` internally to validate the parsed matrix, spectrum invariance under bit reversal, and the exact reference. A standalone/manual verification may use `np.linalg.eigvalsh`; that is a validation step, not a separate feature returned by `FermiHubbardVQEAlgorithm.run()`.

## Return Contract

The returned dict contains the base fields `status`, `circuit_path`, `plot`, and `circuit`, plus:

`Exact Energy`, `VQE Energy`, `Absolute Error`, `Circuit Energy`, `Number of Qubits`, `Optimizer Evaluations`, `Optimizer Converged`, `Optimizer Message`, `VQE Runtime`, `Total Runtime`, `Qubit Mapping`, `Fermionic Hamiltonian`, and `Pauli Hamiltonian`.

`circuit_path` is the SVG circuit path. `plot` is a list of file descriptors for the convergence SVG and optimized-parameter `.npy` file (each has `format` and `filename`). The circuit is returned as `circuit`. Optimized parameters and convergence history are not separate dict fields: parameters are saved in the `.npy` file, while history is used to create the convergence plot. When `measure_shots>0`, the additional fields are `Measured Magnetic Moment`, `Magnetic Moment Standard Errors`, and `Measurement Total Shots`; interpret the measurement cautiously as noted above.

## Minimal Manual Implementation

Use this only as a transparent energy-validation sketch. It deliberately does not implement magnetic measurements or fabricate output paths.

```python
import numpy as np
from scipy.optimize import minimize
from unitarylab import Circuit
from unitarylab.library.fermi_hubbard.fermi_hubbard_pauli import fermi_hubbard_pauli
from unitarylab.library.fermi_hubbard.pauli_ground_state import pauli_string_to_matrix

def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")

def _reverse(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)

def _ansatz(theta, n, layers, *, backend, device, dtype):
    _positive_int("n", n); _positive_int("layers", layers)
    theta = np.asarray(theta, dtype=float)
    if theta.size != 2 * n * layers:
        raise ValueError("theta has the wrong size")
    qc = Circuit(n)
    for layer in theta.reshape(layers, n, 2):
        for q in range(n):
            qc.ry(layer[q, 0], q); qc.rz(layer[q, 1], q)
        for q in range(n - 1): qc.cx(q, q + 1)
        if n > 1: qc.cx(n - 1, 0)
    return qc, np.asarray(qc.execute(backend=backend, device=device, dtype=dtype).state)

def manual_fh_vqe(L=2, t=1.0, U=4.0, B=1.5, layers=3, max_iter=100,
                  seed=7, backend="torch", device="cpu", dtype=np.complex128):
    _positive_int("L", L); _positive_int("layers", layers); _positive_int("max_iter", max_iter)
    if isinstance(seed, bool) or not isinstance(seed, int): raise TypeError("seed must be an integer")
    if not isinstance(backend, str) or not isinstance(device, str):
        raise TypeError("backend and device must be strings")
    try:
        np.dtype(dtype)
    except TypeError as error:
        raise TypeError("dtype must be a valid NumPy dtype") from error
    n = 2 * L
    expression = fermi_hubbard_pauli(L, t, U, B)
    h_be = np.asarray(pauli_string_to_matrix(expression), dtype=np.complex128)
    if h_be.shape != (2 ** n, 2 ** n):
        raise ValueError("Hamiltonian shape does not match the qubit count")
    if not np.allclose(h_be, h_be.conj().T, atol=1e-12):
        raise ValueError("Hamiltonian must be Hermitian")
    p = np.array([_reverse(i, n) for i in range(2 ** n)])
    H = h_be[np.ix_(p, p)]
    exact = float(np.linalg.eigvalsh(H)[0]); history = []
    def energy(theta):
        _, state = _ansatz(theta, n, layers, backend=backend, device=device, dtype=dtype)
        value = float(np.vdot(state, H @ state).real); history.append(value); return value
    x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, 2 * n * layers)
    opt = minimize(energy, x0, method="COBYLA", options={"maxiter": max_iter})
    circuit, state = _ansatz(opt.x, n, layers, backend=backend, device=device, dtype=dtype)
    final_energy = float(np.vdot(state, H @ state).real)
    if state.shape != (2 ** n,) or not np.isfinite(state).all():
        raise RuntimeError("final state has an invalid shape or value")
    if not opt.success: raise RuntimeError(f"optimizer did not converge: {opt.message}")
    if abs(final_energy - float(opt.fun)) > 1e-8: raise RuntimeError("final circuit energy mismatch")
    if final_energy < exact - 1e-8: raise RuntimeError("variational upper bound violated")
    return {"circuit": circuit, "energy": final_energy, "exact_energy": exact,
            "absolute_error": abs(final_energy - exact), "history": history,
            "parameters": np.asarray(opt.x).copy(), "optimizer": opt}
```

## Debugging

1. Start with `L=2` and remember dense memory scales as `2^(4L)`.
2. Check the mode order, `q0` least-significant-bit convention, and bit-reversal round trip together; spectrum equality alone is insufficient.
3. Use `Optimizer Converged`/`Optimizer Message`, then separately inspect absolute error and the variational upper bound.
4. Set `measure_shots=0` for energy-only debugging.
