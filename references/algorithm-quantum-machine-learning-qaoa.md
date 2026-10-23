# Quantum Approximate Optimization Algorithm (QAOA)

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

QAOA is a hybrid quantum-classical algorithm for combinatorial optimization. This implementation solves the Max-Cut problem: partition graph vertices into two sets to maximize the number of cut edges. It alternates between cost-Hamiltonian and mixer-Hamiltonian evolution layers.

The `QAOAAlgorithm` source contract (`unitarylab_algorithms/quantum_machine_learning/qaoa/algorithm.py`) accepts unweighted `(u, v)` edges. It exposes layer count and COBYLA budget, with fixed cost and mixer constructions; it has no weight, custom-ansatz, or optimizer argument. `../scripts/qaoa_algorithm.py` is the wrapper-call example. The manual and Qiskit examples below are separate reconstructions/comparisons.

## Overview

1. Encode the source cost Hamiltonian: $H_C = \sum_{(u,v)\in E} Z_u Z_v$.
2. Build the QAOA circuit: $p$ alternating layers of $e^{-i\gamma_k H_C}$ and $e^{-i\beta_k H_{\text{mix}}}$.
3. Optimize $\gamma, \beta$ using COBYLA to minimize $\langle H_C\rangle$.
4. Extract the bit string with maximum probability as the cut solution.

## Prerequisites

SDK documentation: [UnitaryLab](simulator-unitarylab.md#sources-and-version).

- Graph theory (Max-Cut); Ising Hamiltonians.
- QAOA variational ansatz; Trotterized cost/mixer gates.
- COBYLA gradient-free optimizer.
- `numpy`, `torch`, `networkx`, `scipy.optimize`, `Circuit`.

## Reference Implementation Example

```python
from unitarylab_algorithms.quantum_machine_learning.qaoa.algorithm import QAOAAlgorithm

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
n = 6

algo = QAOAAlgorithm(text_mode="plain")
result = algo.run(
    edges=edges,
    n=n,
    layers=4,
    max_iter=100,
    backend='torch'
)

print(f"Best cut value: {result['Max-Cut Value']}")
print(f"Best partition: {result['Optimal bitstring']}")
print(result['plot'])  # list of {"format": "svg", "filename": "..."} dicts
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output text formatting mode (`"plain"` or `"legacy"`). |
| `algo_dir` | `str\|None` | `None` | Output directory for results. Auto-generated from cwd if `None`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `edges` | `List[Tuple[int,int]]` | default graph | Edge list of the graph. Defaults to a 6-node example. |
| `n` | `int` | `6` | Number of qubits = number of graph vertices. |
| `layers` | `int` | `4` | Number of QAOA layers $p$; increasing it does not guarantee a better optimized result. |
| `max_iter` | `int` | `100` | COBYLA function-evaluation budget passed as `options['maxiter']`. |
| `backend` | `str` | `'torch'` | Simulation backend (`'torch'` or `'numpy'`). |
| `device` | `str` | `'cpu'` | Compute device for the backend. |
| `dtype` | `type` | `np.complex128` | Data type for statevector computation. |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` when the workflow returns; optimizer success is not checked, and exceptions propagate. |
| `circuit_path` | `str` | Path to saved circuit diagram SVG (`QAOA_Circuit.svg`). |
| `plot` | `List[Dict]` | List of `{"format": "svg", "filename": "..."}` dicts for output plots (convergence + Max-Cut graph). |
| `circuit` | `Circuit` | QAOA circuit object built from initial parameters (used for diagram export). |
| `Optimal bitstring` | `str` | Source field name for the most-probable final basis string; global cut optimality is not certified. |
| `Max-Cut Value` | `int` | Cut-edge count of that decoded string. |
| `Optimized Energy` | `float` | Final COBYLA-optimized energy $\langle H_C \rangle$. |
| `Quantum Computation Time` | `float` | Wall-clock time (seconds) for the quantum optimization stage. |

## Implementation Architecture

`QAOAAlgorithm` in `algorithm.py` implements the QAOA hybrid loop in five stages using a Hamiltonian builder, a circuit builder, and COBYLA classical optimization.

**`run(edges, n, layers, max_iter, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | `_get_h_cost(edges, n_qubits)` builds $H_C$ as a `(2^n × 2^n)` NumPy array; `np.linalg.eigvalsh(h_cost)[0]` gives exact ground energy | Creates cost Hamiltonian and initial parameters |
| 2 — Circuit Mapping | `_build_circuit(initial_params, n_qubits, edges)` builds example circuit for visualization | Architecture preview only |
| 3 — Optimization Loop | `minimize(obj_func, x0=initial_params, method='COBYLA', options={'maxiter': max_iter})` — `obj_func` rebuilds circuit, executes, computes `psi† H_C psi` | Core QAOA hybrid quantum-classical loop |
| 4 — Solution Decoding | Builds final circuit with `opt_res.x`; extracts `best_idx = argmax(|psi|²)`; decodes to bitstring; counts Max-Cut value | Greedy Max-Cut solution extraction |
| 5 — Export | `_generate_outputs(edges, best_bits, history, qc_draw, ...)` | Saves circuit SVG, loss curve, Max-Cut graph visualization |

**Helper Methods:**

- **`_get_h_cost(edges, n_qubits)`** — For each edge `(u,v)`: builds $Z_u \otimes Z_v$ via `np.kron` with identity padding; accumulates into `h_cost` matrix. Returns `(2^n, 2^n)` complex128 array.
- **`_build_circuit(params, n_qubits, edges)`** — Creates `Circuit(n_qubits)`. Applies `H` to all qubits (initial $|+\rangle^{\otimes n}$). Loops `p` QAOA layers: for each edge, applies `CX(u,v) → Rz(2γ, v) → CX(u,v)`; for each qubit, applies `Rx(2β, j)`. Backend is passed to `execute()`, not to `Circuit()`.
- **`obj_func(p_flat)` (closure)** — Called by COBYLA. Calls `_build_circuit`, then `qc.execute(initial_state=|0⟩)`; `np.real(psi.conj().T @ h_cost @ psi)`.
- **`_generate_outputs`** — Saves three plots: circuit, energy convergence, and NetworkX Max-Cut graph.

**Data flow:** `edges` → `_get_h_cost()` → COBYLA(`obj_func`) → `opt_res.x` → final circuit → `argmax(|psi|²)` → bitstring → Max-Cut count → result dict.

## Understanding the Key Quantum Components

### 1. Cost Hamiltonian $H_C$
$$H_C = \sum_{(u,v)\in E} Z_u Z_v$$
Implemented as: for each edge $(u,v)$, embed the corresponding $Z_u Z_v$ interaction into the full $n$-qubit space using Kronecker products and add it to the cost matrix. The source does not add the identity offset or the negative one-half Max-Cut normalization.

### 2. Cost Layer $e^{-i\gamma H_C}$
For each edge $(u,v)$: `CX(u, v) → Rz(2γ, v) → CX(u, v)`. This implements $e^{-i\gamma Z_u Z_v}$ via phase kickback.

### 3. Mixer Layer $e^{-i\beta H_{\text{mix}}}$
$$H_{\text{mix}} = \sum_j X_j$$
Implemented as: `Rx(2β, j)` on each qubit $j$. This generates superpositions to explore the cut space.

### 4. Initial State $|+\rangle^{\otimes n}$
Each qubit starts in $|+\rangle = H|0\rangle$, an equal superposition of all $2^n$ bit strings — all possible partitions.

### 5. Approximation and Decoding Limits
Theoretical approximation guarantees require specified graph families, suitable parameters, and an appropriate expectation or sampling criterion. This finite-budget COBYLA workflow and its most-probable-string decoding do not establish such a guarantee. It does not return an approximation ratio or exact maximum-cut certificate.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Initial state $|+\rangle^{\otimes n}$ | `for i in range(n_qubits): qc.h(i)` in `_build_circuit` |
| Cost Hamiltonian $H_C = \sum_{(u,v)\in E} Z_u Z_v$ | `_get_h_cost(edges, n_qubits)` — builds the source cost Hamiltonian in matrix form |
| Cost layer $e^{-i\gamma Z_uZ_v}$ | `cx(u,v) → rz(2*gamma, v) → cx(u,v)` per edge per layer |
| Mixer layer $e^{-i\beta X_j}$ | `rx(2*beta, j)` for each qubit per layer |
| Parameters $(\gamma_1,\ldots,\gamma_p, \beta_1,\ldots,\beta_p)$ | `params[:p]` = gammas, `params[p:]` = betas (flat array) |
| Energy $\langle\psi|H_C|\psi\rangle$ | `np.real(psi.conj().T @ h_cost @ psi).item()` |
| COBYLA optimization | `minimize(obj_func, x0, method='COBYLA')` |
| Bitstring decoding | `best_idx = argmax(|psi|²)` → `format(best_idx, '0nb')` |
| Max-Cut count | `len([(u,v) for (u,v) in edges if best_bits[u] != best_bits[v]])` |
| Exact minimum Ising energy $\lambda_{\min}(H_C)$ | `np.linalg.eigvalsh(h_cost)[0]`; computed for the text summary, not returned as a separate metric |

**Notes on implementation:** This reference uses the source cost Hamiltonian defined in the overview,
$$
H_C = \sum_{(u,v)\in E} Z_u Z_v.
$$
For an unweighted graph, `cut(z)=(|E|-energy(z))/2` in the source's string convention. Its dense builder places vertex 0 in the leftmost Kronecker factor, and decoding indexes the leftmost printed bit as vertex 0. With the SDK's qubit-0-least-significant-bit convention, a dense edge `(u,v)` therefore acts on physical qubits `(n-1-u,n-1-v)`, while the circuit cost layer uses `(u,v)` directly. These may describe different physical edge sets for a nonsymmetric graph. Preserve the distinction when reproducing the source and report the ordering mismatch; repairing the wrapper is a separate code change.

## Mathematical Deep Dive

**QAOA circuit:**
$$|\psi(\gamma, \beta)\rangle = e^{-i\beta_p H_{\text{mix}}} e^{-i\gamma_p H_C} \cdots e^{-i\beta_1 H_{\text{mix}}} e^{-i\gamma_1 H_C} |+\rangle^{\otimes n}$$

**Objective:**
$$
\min_{\gamma,\beta} \langle\psi(\gamma,\beta)|H_C|\psi(\gamma,\beta)\rangle
\quad \text{with} \quad
H_C = \sum_{(u,v)\in E} Z_u Z_v
$$


The source does not compute a Max-Cut approximation ratio; it reports the optimized energy of this Ising cost and the most-probable bit string.

**Total parameters:** $2p$ real numbers ($p$ gammas + $p$ betas).

## Hands-On Example

```python
from unitarylab_algorithms.quantum_machine_learning.qaoa.algorithm import QAOAAlgorithm

# Petersen graph-like structure
edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,5), (1,6), (2,7), (3,8), (4,9)]
n = 10

algo = QAOAAlgorithm()
result = algo.run(edges=edges, n=n, layers=2, max_iter=80)

print(f"Cut value: {result['Max-Cut Value']}")
print(f"Partition: {result['Optimal bitstring']}")
```

## Minimal Manual Implementation

The following Python skeleton reconstructs the core QAOA components — the cost Hamiltonian builder, the QAOA circuit, and the hybrid optimization loop.

```python
# Simplified reconstruction — mirrors QAOAAlgorithm._get_h_cost(), _build_circuit(), and obj_func()
import numpy as np
from scipy.optimize import minimize
from unitarylab.core import Circuit

def build_cost_hamiltonian(edges, n_qubits: int) -> np.ndarray:
    """Build H_C = sum_{(u,v) in E} Z_u Z_v as a (2^n x 2^n) matrix."""
    dim = 2**n_qubits
    H_c = np.zeros((dim, dim), dtype=np.complex128)
    Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    for u, v in edges:
        ops = [I] * n_qubits
        ops[u] = Z; ops[v] = Z
        ZuZv = ops[0]
        for k in range(1, n_qubits): ZuZv = np.kron(ZuZv, ops[k])
        H_c += ZuZv
    return H_c

def build_qaoa_circuit(params: np.ndarray, edges, n_qubits: int) -> Circuit:
    """QAOA circuit: H^n initial state → p layers of cost + mixer gates."""
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    qc = Circuit(n_qubits)                       # backend passed to execute(), not Circuit()
    for i in range(n_qubits): qc.h(i)           # uniform superposition |+>^n
    for layer in range(p):
        # Cost layer: e^{-i*gamma*Z_u*Z_v} via CX-Rz-CX
        for u, v in edges:
            qc.cx(u, v)
            qc.rz(2 * gammas[layer], v)
            qc.cx(u, v)
        # Mixer layer: e^{-i*beta*X_j} via Rx
        for j in range(n_qubits):
            qc.rx(2 * betas[layer], j)
    return qc

def run_qaoa(edges, n: int = 6, layers: int = 2,
             max_iter: int = 100, backend: str = 'torch', device: str = 'cpu'):
    """Full QAOA hybrid loop for Max-Cut."""
    np.random.seed(42)
    H_c = build_cost_hamiltonian(edges, n)
    params0 = np.random.uniform(0, np.pi, 2 * layers)
    history = []

    def obj(params):
        qc = build_qaoa_circuit(params, edges, n)
        psi = np.asarray(qc.execute(
            initial_state=np.eye(2**n, 1, dtype=np.complex128),
            backend=backend, device=device, dtype=np.complex128
        ).state)
        energy = float(np.real(psi.conj().T @ H_c @ psi))
        history.append(energy); return energy

    res = minimize(obj, x0=params0, method='COBYLA', options={'maxiter': max_iter})

    # Decode: most probable basis state as partition
    qc_final = build_qaoa_circuit(res.x, edges, n)
    psi_final = np.asarray(qc_final.execute(
        initial_state=np.eye(2**n, 1, dtype=np.complex128),
        backend=backend, device=device, dtype=np.complex128
    ).state)
    best_idx = int(np.argmax(np.abs(psi_final.flatten())**2))
    best_bits = format(best_idx, f'0{n}b')
    cut_val = len([(u, v) for u, v in edges if best_bits[u] != best_bits[v]])
    return {'status': 'ok', 'Optimal bitstring': best_bits, 'Max-Cut Value': cut_val, 'Optimized Energy': res.fun}
```

## Debugging Tips

1. **`edges` contains invalid vertex indices**: Vertex indices must be in `[0, n)`. Out-of-range indices cause circuit construction errors.
2. **Low cut value**: Inspect the objective history, decoded edge count, graph ordering, and optimization budget. More layers add parameters but do not guarantee an improvement from COBYLA.
3. **COBYLA termination or poor solution**: Construction seeds NumPy and Torch at 42; `run()` exposes no seed or initial-parameter argument. Hold the graph fixed and inspect objective history and budget. A different initialization requires an explicit implementation change.
4. **Large `n`**: The state has `2**n` amplitudes and the dense cost matrix has `4**n` entries; the feasible size depends on memory and backend.
5. **`edges=None`**: Uses the built-in default 6-node graph. Always pass explicit `edges` and matching `n`.
6. **Decoding interpretation**: Count edges using the source string positions to verify `Max-Cut Value`; this field does not certify the graph's maximum cut.

## Reference Implementation

The wrapper contract above is based on `QAOAAlgorithm`.
The following Qiskit example is provided only as a **reference implementation** for users who want to compare the workflow with the standard Qiskit QAOA interface.

Qiskit provides a built-in [`QAOA` class](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.QAOA.html) in `qiskit_algorithms.minimum_eigensolvers`, with a separate sampler/optimizer contract. The example below uses a different diagonal Hamiltonian from the wrapper's Max-Cut example. Its workflow is:

1. Define a diagonal cost Hamiltonian.
2. Instantiate `QAOA` with a sampler and a classical optimizer.
3. Call `compute_minimum_eigenvalue(...)` to solve the optimization problem.

### Minimal Qiskit Example

```python
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA

# Example 2-qubit diagonal cost Hamiltonian
cost_op = SparsePauliOp.from_list([
    ("ZZ", 1.0),
    ("ZI", 0.5),
    ("IZ", 0.5),
])

qaoa = QAOA(
    sampler=StatevectorSampler(),
    optimizer=COBYLA(),
    reps=2,
)

result = qaoa.compute_minimum_eigenvalue(operator=cost_op)

print("Eigenvalue:", result.eigenvalue)
print("Best measurement:", result.best_measurement)
print("Eigenstate:", result.eigenstate)
```
