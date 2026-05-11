---
name: vqe
description: Skill for understanding, using, and implementing the Variational Quantum Eigensolver (VQE) for finding the ground state energy of the 2-qubit Ising Hamiltonian via the VQEAlgorithm class.
---

# Variational Quantum Eigensolver (VQE)

## Purpose

VQE is a hybrid quantum-classical algorithm that finds the ground-state energy of a Hamiltonian by variationally optimizing a parameterized quantum circuit (ansatz). This implementation targets the 2-qubit Ising model:
$$H = Z_0 Z_1 - 0.5 X_0 - 0.5 X_1$$

Use this skill when you need to:
- Find ground-state energies of small quantum systems.
- Learn the variational quantum eigensolver workflow: ansatz → expectation → optimize.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Construct the 2-qubit Ising Hamiltonian matrix $H$.
2. Initialize variational parameters $\theta \in \mathbb{R}^{n\_qubits \times n\_layers}$ randomly.
3. For each optimization step (COBYLA), build the ansatz circuit with `Ry` layers + CX entanglement, compute $\langle\psi(\theta)|H|\psi(\theta)\rangle$.
4. Return the optimized ground state energy and loss history.

## Prerequisites

- Variational principle: $E_0 \leq \langle\psi(\theta)|H|\psi(\theta)\rangle$ for all $\theta$.
- Pauli operator measurement; COBYLA gradient-free optimization.
- `torch`, `numpy`, `scipy.optimize`, `Circuit`.

## Using the Provided Implementation

```python
from unitarylab_algorithms import VQEAlgorithm

algo = VQEAlgorithm(seed=42)
result = algo.run(
    n_qubits=2,
    n_layers=3,
    max_iter=150,
    backend='torch'
)

print(f"Ground energy (VQE): {result['energy']:.4f}")
print(f"Exact ground energy: {result['exact_energy']:.4f}")
print(result['plot'])
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `42` | Random seed for `numpy` and `torch`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_qubits` | `int` | `2` | Number of qubits. Fixed Hamiltonian uses 2; changing this changes the circuit but not the Hamiltonian. |
| `n_layers` | `int` | `3` | Number of variational Ry+CX layers. More layers = more expressive ansatz. |
| `max_iter` | `int` | `150` | Maximum COBYLA iterations. |
| `backend` | `str` | `'torch'` | Simulation backend for `Circuit`. |
| `algo_dir` | `str\|None` | `None` | Directory to save loss curve PNG and circuit SVG. |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `energy` | `float` | VQE optimized ground-state energy estimate. |
| `exact_energy` | `float` | Exact ground state energy (computed by `torch.linalg.eigvalsh`). |
| `loss_history` | `List[float]` | Energy at each COBYLA iteration. |
| `circuit` | `Circuit` | Final ansatz circuit at optimal parameters. |
| `circuit_path` | `str` | Path to circuit SVG diagram. |
| `plot_path` | `str` | Path to loss curve PNG. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`VQEAlgorithm` in `algorithm.py` implements the VQE hybrid loop in five stages using a Hamiltonian builder, an ansatz circuit builder, and SciPy classical optimization.

**`run(n_qubits, n_layers, max_iter, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | `_get_hamiltonian()` builds $H$ as a PyTorch tensor; `torch.linalg.eigvalsh(h_mat)[0]` gives exact energy; `np.random.rand` initializes parameters | Creates Hamiltonian and parameter starting point |
| 2 — Circuit Mapping | `_build_circuit(initial_params, ...)` builds the initial ansatz for visualization | Architecture preview (not used for optimization) |
| 3 — Optimization Loop | `minimize(obj_func, x0, method='COBYLA', maxiter=max_iter)` — each call to `obj_func` rebuilds the circuit, executes it, and computes $\langle\psi|H|\psi\rangle$ | Core VQE hybrid quantum-classical loop |
| 4 — Observable Evaluation | Builds final circuit with `opt_res.x`; computes $\langle Z_0\rangle$, $\langle Z_1\rangle$, $\langle Z_0Z_1\rangle$ | Post-optimization physical observables |
| 5 — Export | `_generate_outputs(history, qc_draw, z0, z1, z0z1, algo_dir)` | Saves circuit PNG, loss curve PNG, observables bar chart |

**Helper Methods:**

- **`_get_hamiltonian()`** — Constructs Ising $H = ZZ - 0.5XI - 0.5IX$ using `torch.kron()`. Returns a `(4,4)` complex128 tensor. Hardcoded for 2-qubit Ising.
- **`_build_circuit(params_flat, n_qubits, n_layers, backend)`** — Reshapes `params_flat` to `(n_qubits, n_layers)`, creates `Circuit(n_qubits, backend=backend)`, applies `ry(params[q,l], q)` per qubit per layer, then `cx(0, 1)` per layer.
- **`obj_func(p_flat)` (closure in `run`)** — Called by COBYLA at each iteration. Rebuilds circuit, calls `qc.execute(initial_state=|0⟩)`, converts result to PyTorch tensor, computes `(psi† H psi).real`.
- **`_generate_outputs(history, ...)`** — Saves three files: circuit diagram, energy convergence plot (matplotlib), and observables bar chart.

**Key execution pattern:** `obj_func` calls `_build_circuit` and `execute` on every optimizer iteration. Each circuit evaluation uses the full statevector simulation. No gradient computation — COBYLA uses only function values.

**Data flow:** `(n_qubits, n_layers)` → `_get_hamiltonian()` → `initial_params` → COBYLA(`obj_func`) → `opt_res.x` → observable evaluation → `_generate_outputs()` → result dict.

## Understanding the Key Quantum Components
The ansatz consists of alternating layers:
```
Layer l: Ry(θ[0,l])⊗Ry(θ[1,l]) → CX(0→1)
```
Each layer has $n\_qubits$ rotation angles $\theta_{q,l}$. The `CX` gate entangles the qubits, creating correlations necessary to capture the ground state.

### 2. Hamiltonian Expectation Value
$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle = \text{Tr}[H \cdot |\psi(\theta)\rangle\langle\psi(\theta)|]$$
Computed in the implementation as a matrix-vector product on the state vector.

### 3. Observables
Three local observables are measured: $Z_0$, $Z_1$, $Z_0Z_1$. Their weighted sum gives $\langle H\rangle = \langle Z_0Z_1\rangle - 0.5\langle X_0\rangle - 0.5\langle X_1\rangle$.

### 4. COBYLA Optimizer (Gradient-Free)
COBYLA minimizes $E(\theta)$ without computing gradients. Well-suited for noisy quantum evaluations where gradients may be unreliable.

### 5. Variational Principle
The variational principle guarantees $E(\theta) \geq E_0$ for all $\theta$. As optimization proceeds, $E(\theta)$ approaches the true ground state energy $E_0$ from above.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Hamiltonian $H = ZZ - 0.5XI - 0.5IX$ | `_get_hamiltonian()` — `torch.kron(Z,Z) - 0.5*torch.kron(X,I) - 0.5*torch.kron(I,X)` |
| Exact ground energy $E_0$ | `torch.linalg.eigvalsh(h_mat)[0]` |
| Ansatz $U(\theta)$ per layer | `_build_circuit()` — `ry(params[q,l], q)` + `cx(0, 1)` |
| Parameter vector $\theta \in \mathbb{R}^{n \cdot L}$ | `params_flat` in `obj_func`; reshaped to `(n_qubits, n_layers)` |
| Energy expectation $\langle\psi|H|\psi\rangle$ | `(torch.conj(psi.t()) @ h_mat @ psi).real.item()` |
| COBYLA optimizer (gradient-free) | `minimize(obj_func, x0, method='COBYLA')` |
| Convergence history | `history` list appended in every `obj_func` call |
| Observable $\langle Z_0\rangle$ | `(psi† ⊗ kron(Z,I) ⊗ psi).real.item()` in Stage 4 |

**Notes on implementation:** The Hamiltonian is hardcoded for the 2-qubit Ising model in `_get_hamiltonian()` — the `n_qubits` parameter affects the ansatz circuit but not the Hamiltonian. The COBYLA optimizer never computes gradients; circuit reconstruction at every step means `n_layers * max_iter` `execute()` calls total. The `execute(initial_state=|0⟩)` call is used for VQE; a `np.eye(2**n, 1)[:, 0]` column vector is passed as the initial state.

## Mathematical Deep Dive

**Ansatz state:** $|\psi(\theta)\rangle = U(\theta)|0^{\otimes n}\rangle$ where $U(\theta) = \prod_{l=1}^{L}[CX_{01}\cdot \bigotimes_q R_y(\theta_{q,l})]$.

**Exact ground energy** for the 2-qubit Ising model: $E_0 \approx -1.2808$.

**Convergence:** With $n\_layers \geq 2$, the ansatz has sufficient expressibility to reach $E_0$ exactly for this 2-qubit Hamiltonian.

## Hands-On Example (UnitaryLab)

```python
from unitarylab_algorithms import VQEAlgorithm

algo = VQEAlgorithm(seed=0)
result = algo.run(n_qubits=2, n_layers=4, max_iter=200)

print(f"VQE energy:   {result['energy']:.6f}")
print(f"Exact energy: {result['exact_energy']:.6f}")
print(f"Error:        {abs(result['energy'] - result['exact_energy']):.2e}")
```

## Reference Implementation (Qiskit)
The main implementation in this skill is based on the project’s own VQEAlgorithm.
Qiskit is included here only as a concise reference example showing the standard VQE workflow with its estimator + ansatz + optimizer abstraction.
### Minimal Qiskit Example
```python
from qiskit.circuit.library import TwoLocal
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
ansatz = TwoLocal(
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
from qiskit.circuit.library import TwoLocal
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

ansatz = TwoLocal(
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
  Official reference:  
  `https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.AdaptVQE.html#qiskit_algorithms.AdaptVQE`
  
## Minimal Manual Implementation (UnitaryLab) 

The following Python skeleton reconstructs the VQE core components — the Hamiltonian builder, the ansatz circuit, and the hybrid optimization loop.

```python
# Simplified reconstruction — mirrors VQEAlgorithm._get_hamiltonian(), _build_circuit(), and obj_func()
import numpy as np
import torch
from scipy.optimize import minimize
from unitarylab.core import Circuit

def build_ising_hamiltonian() -> torch.Tensor:
    """2-qubit Ising H = ZZ - 0.5*XI - 0.5*IX."""
    I = torch.eye(2, dtype=torch.complex128)
    X = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
    Z = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
    return torch.kron(Z, Z) - 0.5*torch.kron(X, I) - 0.5*torch.kron(I, X)

def build_vqe_ansatz(params_flat: np.ndarray, n_qubits: int, n_layers: int,
                     backend: str = 'torch') -> Circuit:
    """Ry+CX ansatz: for each layer, Ry per qubit then CX(0,1)."""
    params = params_flat.reshape(n_qubits, n_layers)
    qc = Circuit(n_qubits, backend=backend)
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(params[q, l]), q)   # variational rotation
        qc.cx(0, 1)                          # entanglement
    return qc

def vqe_energy(params_flat: np.ndarray, n_qubits: int, n_layers: int,
               H: torch.Tensor, backend: str = 'torch') -> float:
    """Compute <psi(params)|H|psi(params)> by statevector simulation."""
    qc = build_vqe_ansatz(params_flat, n_qubits, n_layers, backend)
    psi0 = np.zeros((2**n_qubits, 1), dtype=np.complex128);  psi0[0, 0] = 1.0
    psi_out = qc.execute(initial_state=psi0)
    psi = torch.as_tensor(psi_out, dtype=torch.complex128)
    return float((psi.conj().T @ H @ psi).real.item())

def run_vqe(n_qubits: int = 2, n_layers: int = 3, max_iter: int = 150,
            backend: str = 'torch', seed: int = 42):
    """Full VQE hybrid loop."""
    np.random.seed(seed)
    H = build_ising_hamiltonian()
    exact_E = float(torch.linalg.eigvalsh(H)[0].real)
    theta_0 = np.random.rand(n_qubits * n_layers) * 2 * np.pi
    history = []

    def obj(p):
        e = vqe_energy(p, n_qubits, n_layers, H, backend)
        history.append(e)
        return e

    res = minimize(obj, x0=theta_0, method='COBYLA', options={'maxiter': max_iter})
    return {'energy': res.fun, 'exact_energy': exact_E, 'loss_history': history}
```

## Debugging Tips

1. **VQE energy above exact**: Insufficient layers (`n_layers`) or iterations (`max_iter`). Increase both.
2. **Local minima**: COBYLA may converge to a local minimum. Re-run with different `seed`.
3. **`n_qubits != 2`**: The Hamiltonian is hard-coded for 2 qubits. Changing `n_qubits` changes the ansatz but not the Hamiltonian — use only `n_qubits=2` for a meaningful result.
4. **Slow convergence**: COBYLA can be slow for large parameter counts ($n\_qubits \times n\_layers$). For $\geq 10$ parameters, consider gradient-based methods with Parameter Shift Rule.
5. **Loss not decreasing**: Check that `max_iter` is large enough. For 6 parameters, 150 iterations is usually sufficient; for more parameters, increase to 500+.
