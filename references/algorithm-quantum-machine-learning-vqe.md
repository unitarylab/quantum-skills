# Variational Quantum Eigensolver (VQE)

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

VQE estimates ground-state energy by optimizing a parameterized circuit. This implementation accepts a small Hermitian matrix of dimension `2**n` or generates a random one for benchmarking.

The `VQEAlgorithm` source contract (`unitarylab_algorithms/quantum_machine_learning/vqe/algorithm.py`) fixes an Ry-Rz/ring-CX ansatz and COBYLA. Its parameters support Hamiltonian selection, depth, initialization, and optimization budget; they do not expose custom ansätze or optimizers. `../scripts/vqe_algorithm.py` shows the wrapper call. The manual and Qiskit examples below are separate implementations.

## Overview

1. Validate or randomly generate a Hermitian Hamiltonian matrix $H$.
2. Initialize variational parameters $\theta \in \mathbb{R}^{2 \times n \times \text{layers}}$ uniformly at random in $[-\pi, \pi]$.
3. For each optimization step (COBYLA), build the ansatz circuit with `Ry`+`Rz` rotations and ring CX entanglement, compute $\langle\psi(\theta)|H|\psi(\theta)\rangle$.
4. Return the optimized VQE energy, exact energy, and absolute error.

## Prerequisites

SDK documentation: [UnitaryLab](simulator-unitarylab.md#sources-and-version).

- Variational principle: $E_0 \leq \langle\psi(\theta)|H|\psi(\theta)\rangle$ for all $\theta$.
- Pauli operator measurement; COBYLA gradient-free optimization.
- `torch`, `numpy`, `scipy.optimize`, `Circuit`.

## Reference Implementation Example

```python
from unitarylab_algorithms.quantum_machine_learning.vqe.algorithm import VQEAlgorithm

algo = VQEAlgorithm(text_mode="plain")
result = algo.run(
    n=2,
    layers=3,
    max_iter=150,
    seed=7,
    backend='torch'
)

print(f"Ground energy (VQE): {result['VQE Energy']:.4f}")
print(f"Exact ground energy: {result['Exact Energy']:.4f}")
print(f"Absolute error: {result['Absolute Error']:.2e}")
print(result['plot'])  # list of saved file dicts
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output text rendering mode passed to `BaseAlgorithm`. |
| `algo_dir` | `str\|None` | `None` | Directory to save outputs. Auto-derived from `cwd/results/...` if `None`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `2` | Number of qubits (used when generating a random Hamiltonian). |
| `layers` | `int` | `2` | Number of variational Ry+Rz+ring-CX layers. More layers = more expressive ansatz. |
| `max_iter` | `int` | `150` | COBYLA function-evaluation budget passed as `options['maxiter']`. |
| `seed` | `int` | `7` | Random seed for parameter initialization and random Hamiltonian generation. |
| `hamiltonian` | `np.ndarray\|None` | `None` | User-supplied Hermitian matrix. If `None`, a random Hermitian is generated. |
| `normalize` | `bool` | `True` | Whether to normalize the random Hamiltonian by its spectral norm. |
| `backend` | `str` | `'torch'` | Simulation backend for `Circuit.execute()`. |
| `device` | `str` | `'cpu'` | Device string passed to `Circuit.execute()`. |
| `dtype` | | `np.complex128` | Numeric dtype passed to `Circuit.execute()`. |

## Return Fields

Returned by `_build_return_dict(success, circuit_path, filepath, circuit)` merged with `self.output`:

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` when the workflow returns; it does not test `opt_res.success`. Exceptions propagate. |
| `circuit_path` | `str` | Path to the initial ansatz circuit SVG (`VQE_Circuit.svg`). |
| `plot` | `List[dict]` | List of saved file dicts, each `{"format": "svg", "filename": "<path>"}`. Contains the convergence plot (`VQE_Convergence.svg`). |
| `circuit` | `Circuit` | Initial ansatz `Circuit` object (at initial parameters). |
| `Exact Energy` | `float` | Exact ground-state energy from `np.linalg.eigvalsh`. |
| `VQE Energy` | `float` | VQE optimized energy estimate. |
| `Absolute Error` | `float` | `\|VQE Energy − Exact Energy\|`. |
| `Optimizer Message` | `str` | COBYLA termination message. |
| `Quantum Comp Time` | `float` | Wall-clock seconds for the COBYLA optimization loop. |

## Implementation Architecture

`VQEAlgorithm` in `algorithm.py` implements the VQE hybrid loop in four stages.

**`run(n, layers, max_iter, seed, hamiltonian, normalize, backend, device, dtype)` — Four Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Validation & Init | `_validate_hamiltonian(hamiltonian)` checks shape/Hermiticity; or `_random_hermitian(n, seed, normalize)` generates one; `rng.uniform(-π, π, 2*num_qubits*layers)` initializes parameters | Prepares $H$ and starting point $\theta_0$ |
| 2 — Ansatz Preview | `_build_circuit(initial_theta, num_qubits, layers)` builds the parameterized circuit for visualization | Architecture preview saved as `VQE_Circuit.svg` |
| 3 — Optimization Loop | `minimize(_expectation, x0=initial_theta, method='COBYLA')` — each call rebuilds the circuit via `_build_circuit`, executes it, and returns $\langle\psi|H|\psi\rangle$ | Core VQE hybrid quantum-classical loop |
| 4 — Export | `save_circuit(qc_draw, 'VQE_Circuit.svg')` + matplotlib convergence plot saved as `VQE_Convergence.svg` | Persists outputs; returns result dict via `_build_return_dict` |

**Helper Methods:**

- **`_validate_hamiltonian(hamiltonian)`** — Casts to `complex128`, checks square power-of-2 dimension and Hermiticity. Returns `(h, num_qubits)`.
- **`_random_hermitian(num_qubits, seed, normalize)`** — Generates a random Hermitian via $(A + A^\dagger)/2$; optionally divides by spectral norm. Returns `np.ndarray`.
- **`_build_ansatz(layer_parameters, num_qubits)`** — Applies `ry` then `rz` to each qubit, followed by ring CX entanglement (`cx(q, q+1)` for all $q$, then `cx(num_qubits-1, 0)`). `layer_parameters` has shape `(num_qubits, 2)`.
- **`_build_circuit(parameters_flat, num_qubits, layers)`** — Reshapes `parameters_flat` to `(layers, num_qubits, 2)`, creates a `Circuit(num_qubits)`, and appends one `_build_ansatz` sub-circuit per layer.
- **`_expectation(parameters_flat, hamiltonian, num_qubits, layers, history)`** — Called by COBYLA at each iteration. Calls `_build_circuit(...).execute(backend=backend, device=device, dtype=dtype).state`, computes `state.conj().T @ hamiltonian @ state`, appends to `history`, returns real part.

**Key execution pattern:** `_expectation` rebuilds and executes the circuit for every objective evaluation. COBYLA uses function values; the history records evaluations, and termination can occur before the `max_iter` budget is exhausted.

**Data flow:** `(n, layers)` → `_validate_hamiltonian` / `_random_hermitian` → `initial_theta` → COBYLA(`_expectation`) → `opt_res` → `save_circuit` + convergence plot → `_build_return_dict`.

## Understanding the Key Quantum Components

The ansatz consists of alternating layers:
```
Layer l: Ry(θ[l,0,0]) Rz(θ[l,0,1]) ⊗ Ry(θ[l,1,0]) Rz(θ[l,1,1]) ⊗ ... → CX(0→1) → CX(1→2) → ... → CX(n-1→0)
```
Each layer applies 2 rotation angles per qubit (Ry + Rz), then a ring of CX gates that fully entangles adjacent qubits. Total parameters per run: $2 \times n \times \text{layers}$.

### 2. Hamiltonian Expectation Value
$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle = \text{Tr}[H \cdot |\psi(\theta)\rangle\langle\psi(\theta)|]$$
Computed in the implementation as a matrix-vector product on the state vector.

### 3. COBYLA Optimizer (Gradient-Free)
COBYLA minimizes $E(\theta)$ without computing gradients. Well-suited for noisy quantum evaluations where gradients may be unreliable.

### 4. Variational Principle
For a normalized state and Hermitian Hamiltonian, the variational principle gives $E(\theta)\geq E_0$, up to numerical error. It does not guarantee monotonic objective values, global optimization, or convergence to the ground state.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Hermitian Hamiltonian $H$ | `_validate_hamiltonian(hamiltonian)` / `_random_hermitian(n, seed, normalize)` |
| Exact ground energy $E_0$ | `_exact_ground_energy(hamiltonian)` — `np.linalg.eigvalsh(hamiltonian).min()` |
| Ansatz layer $U_l(\theta)$ | `_build_ansatz(layer_parameters, num_qubits)` — `ry` + `rz` per qubit, then ring CX |
| Full circuit $U(\theta)$ | `_build_circuit(parameters_flat, num_qubits, layers)` — appends `layers` ansatz sub-circuits |
| Parameter vector $\theta \in \mathbb{R}^{2 n L}$ | `parameters_flat`; reshaped to `(layers, num_qubits, 2)` inside `_build_circuit` |
| Energy expectation $\langle\psi|H|\psi\rangle$ | `_expectation` — `(state.conj().T @ hamiltonian @ state).real` |
| COBYLA optimizer (gradient-free) | `minimize(_expectation, x0=initial_theta, method='COBYLA')` |
| Convergence history | `history` list appended inside every `_expectation` call |
| Result dict | `_build_return_dict(True, circuit_path, filename, qc_draw)` merged with `self.output` |

**Notes on implementation:** The Hamiltonian works for any $2^n \times 2^n$ Hermitian matrix; `n` only matters when generating a random one. The COBYLA optimizer never computes gradients. `.execute(backend=backend, device=device, dtype=dtype).state` returns the full statevector.

## Mathematical Deep Dive

**Ansatz state:** $|\psi(\theta)\rangle = U(\theta)|0^{\otimes n}\rangle$ where $U(\theta) = \prod_{l=1}^{L}[\text{RingCX} \cdot \bigotimes_q R_z(\theta_{l,q,1}) R_y(\theta_{l,q,0})]$.

**Parameter count:** $2 \times n \times \text{layers}$ real-valued angles, initialized uniformly in $[-\pi, \pi]$ via `rng.uniform(-np.pi, np.pi, size=2 * num_qubits * layers)`.

**Convergence:** Accuracy depends on ansatz expressibility, initialization, and optimizer termination. More layers or a larger budget can help particular cases, but neither guarantees the exact ground energy. Inspect `Optimizer Message` and `Absolute Error` separately; the returned `circuit` remains the initial preview circuit.

## Hands-On Example

```python
from unitarylab_algorithms.quantum_machine_learning.vqe.algorithm import VQEAlgorithm

algo = VQEAlgorithm(text_mode="plain")
result = algo.run(n=2, layers=4, max_iter=200, seed=0)

print(f"VQE energy:   {result['VQE Energy']:.6f}")
print(f"Exact energy: {result['Exact Energy']:.6f}")
print(f"Error:        {result['Absolute Error']:.2e}")
print(f"Status:       {result['status']}")
print(f"Circuit SVG:  {result['circuit_path']}")
print(f"Plot files:   {result['plot']}")
```

## Minimal Manual Implementation

The following Python skeleton mirrors the core methods of `VQEAlgorithm`: `_random_hermitian`, `_build_ansatz`, `_build_circuit`, `_expectation`, and the COBYLA loop in `run`.

```python
# Mirrors VQEAlgorithm._random_hermitian, _build_ansatz, _build_circuit, _expectation, run()
import numpy as np
from scipy.optimize import minimize
from unitarylab import Circuit

def random_hermitian(num_qubits: int, seed: int = 7, normalize: bool = True) -> np.ndarray:
    """Random Hermitian via (A + A†)/2, optionally normalized by spectral norm."""
    dim = 2 ** num_qubits
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (a + a.conj().T) / 2.0
    if normalize:
        spec_norm = np.linalg.norm(h, ord=2)
        if spec_norm > 0:
            h = h / spec_norm
    return h

def build_ansatz(layer_parameters: np.ndarray, num_qubits: int) -> Circuit:
    """One ansatz layer: Ry+Rz per qubit, then ring CX entanglement."""
    ansatz = Circuit(num_qubits)
    for q in range(num_qubits):
        ansatz.ry(float(layer_parameters[q, 0]), q)
        ansatz.rz(float(layer_parameters[q, 1]), q)
    for q in range(num_qubits - 1):
        ansatz.cx(q, q + 1)
    if num_qubits > 1:
        ansatz.cx(num_qubits - 1, 0)
    return ansatz

def build_circuit(parameters_flat: np.ndarray, num_qubits: int, layers: int) -> Circuit:
    """Stack `layers` ansatz sub-circuits into one Circuit."""
    parameters = np.asarray(parameters_flat, dtype=float).reshape(layers, num_qubits, 2)
    qc = Circuit(num_qubits)
    for layer in range(layers):
        layer_qc = build_ansatz(parameters[layer], num_qubits)
        qc.append(layer_qc, range(num_qubits))
    return qc

def expectation(parameters_flat: np.ndarray, hamiltonian: np.ndarray,
                num_qubits: int, layers: int, history: list) -> float:
    """Compute ⟨ψ(θ)|H|ψ(θ)⟩ via statevector simulation."""
    state = build_circuit(parameters_flat, num_qubits, layers).execute(
        backend='torch', device='cpu', dtype=np.complex128
    ).state
    energy = float(np.real((state.conj().T @ hamiltonian @ state).item()))
    history.append(energy)
    return energy

def run_vqe(n: int = 2, layers: int = 2, max_iter: int = 150, seed: int = 7):
    """Full VQE hybrid loop matching VQEAlgorithm.run()."""
    hamiltonian = random_hermitian(n, seed=seed, normalize=True)
    exact_energy = float(np.min(np.real(np.linalg.eigvalsh(hamiltonian))))
    rng = np.random.default_rng(seed)
    initial_theta = rng.uniform(-np.pi, np.pi, size=2 * n * layers)
    history = []

    opt_res = minimize(
        fun=expectation,
        x0=initial_theta,
        args=(hamiltonian, n, layers, history),
        method='COBYLA',
        options={'maxiter': max_iter},
    )
    return {
        'VQE Energy': float(opt_res.fun),
        'Exact Energy': exact_energy,
        'Absolute Error': abs(float(opt_res.fun) - exact_energy),
        'history': history,
    }
```

## Debugging Tips

1. **VQE energy above exact**: This is consistent with the variational bound. Inspect ansatz limitations, initialization, objective history, and `Optimizer Message` before changing depth or budget.
2. **Initialization comparisons**: With an explicit fixed `hamiltonian`, changing `seed` changes initial parameters. With `hamiltonian=None`, the same seed also generates the random Hamiltonian; hold that matrix fixed before comparing optimizer starts on the same problem.
3. **Invalid Hamiltonian**: `_validate_hamiltonian` raises `ValueError` if the matrix is not square, not a power-of-2 dimension, or not Hermitian. Verify with `np.allclose(H, H.conj().T)`.
4. **Slow optimization**: There are `2*n*layers` parameters and every objective evaluation executes a full statevector circuit. A gradient optimizer would require a separate implementation change; it is not a `run()` option.
5. **Energy below exact beyond numerical tolerance**: Check state normalization, Hamiltonian ordering, Hermiticity, and energy calculation. A successful workflow status does not validate the variational inequality.

## Reference Implementation

The wrapper contract above is based on `VQEAlgorithm`.
Qiskit is included here only as a concise reference example showing the standard VQE workflow with its estimator + ansatz + optimizer abstraction.
The comparison uses the official [`VQE` interface](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.VQE.html) and [`n_local` circuit builder](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.n_local). Its ansatz and optimizer differ from the UnitaryLab wrapper; their flexibility does not add arguments to `VQEAlgorithm.run()`.
### Minimal Qiskit Example
```python
from qiskit.circuit.library import n_local
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP

# Example Hamiltonian
hamiltonian = SparsePauliOp.from_list([
    ("ZZ", 1.0),
    ("XI", -0.5),
    ("IX", -0.5),
])

# Parameterized ansatz
ansatz = n_local(
    num_qubits=2,
    rotation_blocks="ry",
    entanglement_blocks="cx",
    reps=2,
)

# Estimator + optimizer
estimator = StatevectorEstimator()
optimizer = SLSQP(maxiter=300)

# Standard VQE object
vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
)

result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

print("Ground-state energy:", result.eigenvalue)
print("Optimal parameters:", result.optimal_parameters)
print("Cost function evaluations:", result.cost_function_evals)
```

### Qiskit Example with Callback
```python
import numpy as np
from qiskit.circuit.library import n_local
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SLSQP

def my_callback(eval_count, params, mean, metadata):
    if eval_count % 10 == 0:
        print(f"Iter {eval_count}: energy = {mean:.6f}")

hamiltonian = SparsePauliOp.from_list([
    ("ZZ", 1.0),
    ("XI", -0.5),
    ("IX", -0.5),
])

ansatz = n_local(
    num_qubits=2,
    rotation_blocks="ry",
    entanglement_blocks="cx",
    reps=2,
)

estimator = StatevectorEstimator()
optimizer = SLSQP(maxiter=300)

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
    initial_point=np.zeros(ansatz.num_parameters),
    callback=my_callback,
)

result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

print("Final energy:", result.eigenvalue)
print("Optimal point:", result.optimal_point)
```
### Other Qiskit VQE Variants (Reference)

Qiskit also includes several VQE-related variants beyond the standard
`VQE` workflow:

- **`AdaptVQE`**
  An adaptive VQE variant that iteratively builds a compact ansatz from a pool of
  candidate operators. In each iteration, it selects the operator associated with
  the largest gradient and appends the corresponding evolution block, making the
  ansatz increasingly tailored to the target Hamiltonian. It relies on an internal
  `VQE` solver and is commonly used with operator pools such as excitation-based
  ansätze in quantum chemistry.
  [Official AdaptVQE reference](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.AdaptVQE.html).
