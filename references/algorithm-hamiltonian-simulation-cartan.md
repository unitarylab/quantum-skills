# Cartan Decomposition Hamiltonian Simulation Reference

## Scope and Reference Use

The reference focuses on the real-symmetric matrix path, Lax-flow settings, exported circuit, and numerical comparison with `expm`.

Scope:
- The Hamiltonian is a real symmetric matrix suitable for Cartan-Lax decomposition
- High-precision Hamiltonian simulation is needed with controllable error tolerance
- Lie-algebraic structure of the Hamiltonian is important

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

Wrapper source: `unitarylab_algorithms/hamiltonian_simulation/cartan/algorithm.py`.

Reference scripts: [cartan_algorithm.py](../scripts/cartan_algorithm.py) (API usage example); [cartan_implementation.py](../scripts/cartan_implementation.py) (independent teaching implementation).

## Overview

**Category:** Hamiltonian Simulation — Structural Decomposition

Cartan decomposition computes the time-evolution operator

$$
U(t) = e^{-iHt}
$$

through a Cartan split $\mathfrak{g}=\mathfrak{k}\oplus\mathfrak{m}$ and a Lax flow, targeting the form $K e^{-i\eta}K^\dagger$. Here $\mathfrak{k}$ is a subalgebra and $\mathfrak{m}$ is a complementary subspace, generally not a subalgebra. The wrapper delegates the decomposition to `unitarylab.library.hamiltonian.hamiltonian_simulation(method='cartan-lax')`; a symmetric/antisymmetric matrix split alone is not the algorithm.

---

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import CartanDecompositionAlgorithm

# Define a 2x2 real symmetric Hamiltonian
H = np.array([[2.0, 1.0],
              [1.0, 2.0]])

algo = CartanDecompositionAlgorithm()

result = algo.run(
    H=H,
    t=1.0,
    error=1e-3,
    lr=1e-3,
    max_steps=100000,
    reps=5000,
)

print("status      :", result['status'])
print("circuit_path:", result['circuit_path'])
print("plot        :", result['plot'])          # e.g. [{'format': 'txt', 'filename': '/path/to/result.txt'}]
```

### Accessing Detailed Output

All `self.output` fields are merged into the returned dict, so they can be accessed either from `result` directly or from `algo.output`:

```python
# Approximate evolution unitary
U_approx = result["Evolution result"]        # same as algo.output["Evolution result"]

# Exact reference unitary
U_exact  = result["Exact evolution"]

# Achieved error
error_val = result["Final total error"]

# Saved files
for f in result["plot"]:
    print(f"Saved {f['format']} file: {f['filename']}")

print(f"Total error: {error_val:.2e}")
```

## Core Parameters Explained

### `CartanDecompositionAlgorithm.__init__()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output text format mode. |
| `algo_dir` | `str\|None` | `None` | Directory for saving result files. Auto-generated if `None`. |

### `CartanDecompositionAlgorithm.run()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `H` | `np.ndarray \| list` | required | Hermitian Hamiltonian matrix (2D array). Must be real symmetric in the current implementation. |
| `t` | `float` | required | Total evolution time. Also used as default for `evol_time`. |
| `error` | `float` | required | Positive stopping tolerance for the off-$\mathfrak{h}$ component norm; not itself the final unitary-error bound. |
| `evol_time` | `float` | `t` | Override for the evolution time passed to the simulator. |
| `lr` | `float` | `1e-3` | Positive base integration step size; the SDK adapts the effective value. |
| `max_steps` | `int` | `100000` | Positive hard cap on Lax updates. |
| `reps` | `int` | `5000` | Positive baseline budget before adaptive scaling. |
| `backend`, `device`, `dtype` | execution settings | `'torch'`, `'cpu'`, `np.complex128` | Forwarded to `hamiltonian_simulation`. |

The wrapper forwards these values to the SDK. The SDK checks positivity, real symmetry, finite time, and a square matrix of positive dimension. Supply `H` as a NumPy array: although the signature also names `list`, the wrapper's exact-reference expression multiplies `H` by a complex scalar without converting it first.

---

## Inputs and Outputs

### Inputs

| Name | Type | Constraints |
|---|---|---|
| `H` | `np.ndarray` | Real-symmetric square matrix; the SDK pads a non-power-of-two dimension and crops returned matrices to the original subspace. |
| `t` | `float` | Finite real number. |
| `error` | `float` | Positive off-$\mathfrak h$ stopping tolerance. |

### Outputs

The `run()` method returns a dictionary built by `_build_return_dict(success, circuit_path, filepath, circuit)`. The `self.output` fields are merged into the result via `result.update(self.output)`, so all keys below are accessible directly on the returned dict:

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on normal return; inspect `Final total error` separately. Exceptions propagate. |
| `circuit_path` | `str` | Local path to the saved circuit diagram (SVG) |
| `plot` | `list` | List of saved result files, each as `{"format": str, "filename": str}` (format is the 3-char file extension) |
| `circuit` | `object` | Raw circuit object from `runable.circuit` |
| `Evolution result` | `np.ndarray` | Approximate unitary from the Cartan-Lax flow (`runable.evolution_result`) |
| `Final total error` | `float` | SDK `total_error`: spectral norm of exact-minus-approximate evolution on the original subspace; distinct from the off-$\mathfrak h$ stopping norm. |
| `Computation time (s)` | `float` | Wall-clock runtime in seconds |
| `Exact evolution` | `np.ndarray` | Exact matrix $U_{\text{exact}} = e^{-iHt_{\text{evol}}}$ computed via `scipy.linalg.expm` |

---

## Implementation Architecture

### Execution Flow

1. **Input recording** — Store `H`, `t`, and `error` via `self.update_input(...)`.
2. **Parameter expansion** — Read `evol_time`, `lr`, `max_steps`, `reps` from `**kwargs`; fall back to defaults when absent.
3. **Cartan-Lax simulation** — Call `hamiltonian_simulation(H, evol_time, method='cartan-lax', target_error=error, lr=lr, max_steps=max_steps, reps=reps)`.
4. **Exact reference** — Compute $U_{\text{exact}} = \text{expm}(-i H\, t_{\text{evol}})$ via `scipy.linalg.expm`.
5. **Result export** — Save the circuit diagram and text report; return the unified result dictionary via `self._build_return_dict(...)`.

### Engineering Constraints

- The current implementation handles Hamiltonians through the **matrix path** only; Pauli-string input is not yet supported.
- Computational cost scales with `lr`, `max_steps`, and `reps`. Tight `error` thresholds combined with small `lr` will increase runtime significantly.
- `max_steps` acts as a hard termination guard; convergence is not guaranteed if the budget is exhausted before `error` is met.
- Classical overhead from `expm` grows cubically with Hilbert-space dimension; this implementation is suited for small-to-medium systems.

---

## Mathematical Deep Dive

The Lie algebra $\mathfrak{g}$ is decomposed as a direct sum:

$$
\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}
$$

with the canonical closure relations:

$$
[\mathfrak{k},\, \mathfrak{k}] \subseteq \mathfrak{k}, \qquad
[\mathfrak{k},\, \mathfrak{m}] \subseteq \mathfrak{m}, \qquad
[\mathfrak{m},\, \mathfrak{m}] \subseteq \mathfrak{k}
$$

The target unitary is factored as:

$$
U(t) = K \cdot e^{-i\eta} \cdot K^\dagger, \qquad K \in e^{\mathfrak{k}},\quad \eta \in \mathfrak{m}
$$

The `cartan-lax` flow iteratively updates $K$ via gradient-descent-like steps until the off-$\mathfrak{h}$ component norm falls below the target error $\epsilon$. The exact reference matrix

$$
U_{\text{exact}} = e^{-iH\,t_{\text{evol}}}
$$

is computed via `scipy.linalg.expm` for error benchmarking.

---

## Prerequisites

- Lie algebra basics: decomposition $\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}$ with canonical closure relations
- Hamiltonian simulation fundamentals: $U(t) = e^{-iHt}$
- Python: `numpy`, `scipy.linalg.expm`
- Understanding of gradient-descent-like iterative flows on Lie algebras

## Understanding the Key Quantum Components

1. **Lie algebra decomposition**: A Cartan involution defines a subalgebra $\mathfrak{k}$ and complementary subspace $\mathfrak{m}$ with the displayed commutator closure relations. Symmetric/antisymmetric matrix labels depend on representation and should not replace that construction.
2. **Lax flow iteration**: `cartan-lax` iteratively updates $K \in e^{\mathfrak{k}}$ via gradient-descent-like steps, driving the off-$\mathfrak{h}$ component norm below the target error $\epsilon$. The `lr` parameter controls step size; `max_steps` acts as a hard termination guard.
3. **Target factorization**: $U(t) = K \cdot e^{-i\eta} \cdot K^\dagger$ with $K \in e^{\mathfrak{k}}$ and $\eta \in \mathfrak{m}$. The circuit construction follows from this factored form.
4. **Error control**: `error` sets the stopping tolerance for the off-$\mathfrak{h}$ norm. Tight tolerances combined with small `lr` increase iteration count significantly.
5. **Matrix-only path**: The current implementation operates on dense matrices — Pauli-string input is not yet supported. The classical `expm` overhead grows cubically with Hilbert-space dimension.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Hamiltonian $H$ | `H` — 2D `np.ndarray`, real symmetric |
| Evolution time $t$ | `t` parameter; also `evol_time` override |
| Stopping tolerance $\epsilon$ | Positive `error`, applied to the off-$\mathfrak h$ norm. |
| Lax step size | Positive `lr`, adapted inside the SDK. |
| Maximum iterations | Positive `max_steps`, hard cap. |
| Iteration budget | Positive `reps`, baseline before SDK adaptation. |
| Cartan-Lax engine | `hamiltonian_simulation(H, evol_time, method='cartan-lax', ...)` from `unitarylab.library` |
| Exact reference $e^{-iHt}$ | `scipy.linalg.expm(-1j * H * t_evol)` |
| Approximate unitary $U_{\text{approx}}$ | `result["Evolution result"]` |
| Achieved error | `result["Final total error"]` |

## Hands-On Example

Sweep error tolerance to observe accuracy-vs-runtime trade-offs:

```python
import numpy as np
from unitarylab_algorithms import CartanDecompositionAlgorithm

H = np.array([[2.0, 1.0],
              [1.0, 2.0]])

for error in [1e-2, 1e-3, 1e-5]:
    algo = CartanDecompositionAlgorithm()
    result = algo.run(H=H, t=1.0, error=error, lr=1e-3, max_steps=100000)
    print(f"error_tol={error:.0e}, achieved={result['Final total error']:.2e}, "
          f"runtime={result['Computation time (s)']:.3f}s, status={result['status']}")
```

Compare achieved error and runtime empirically; a tighter requested tolerance does not guarantee convergence or monotonic improvement under a fixed iteration budget.

## Minimal Manual Implementation

```python
import numpy as np
from scipy.linalg import expm

def cartan_simulation_skeleton(H, t, lr=1e-3, max_steps=100000, error=1e-3):
    """Delegate to the SDK; this is not an independent Cartan-Lax implementation."""
    from unitarylab.library.hamiltonian import hamiltonian_simulation
    H = np.asarray(H, dtype=complex)
    result = hamiltonian_simulation(
        H, t, method='cartan-lax', target_error=error,
        lr=lr, max_steps=max_steps, reps=5000,
    )
    U_approx = result.evolution_result
    U_exact = expm(-1j * H * t)
    return U_approx, np.linalg.norm(U_approx - U_exact, 'fro')
```

This example preserves delegation to the SDK. A loop based only on `(H-H.T)/2` would have a zero update for every real-symmetric input and is not a valid Cartan-Lax reimplementation.

## Debugging Tips

1. **Convergence failure**: If `max_steps` is exhausted before `error` is met, increase `max_steps` or relax `error`. Small `lr` with tight `error` is the most common cause of slow convergence.
2. **lr too large**: The Lax flow may oscillate or diverge. If the final error is worse than expected, halve `lr` and re-run. Start with `1e-3` and adjust.
3. **Non-symmetric H**: The current implementation expects real symmetric Hamiltonians. If `H` has significant imaginary components, the `k`/`m` split may produce unexpected results — verify symmetry with `np.allclose(H, H.T)`.
4. **Dimension scaling**: Dense matrix storage and `expm` become expensive as dimension grows. There is no established universal qubit threshold at which the exact reference dominates the Lax-flow runtime.
5. **reps vs max_steps**: `reps` sets the baseline iteration budget before adaptive scaling; `max_steps` is the absolute hard cap. If the flow converges near `reps`, increasing it can reduce total iterations by enabling better adaptive scaling.
