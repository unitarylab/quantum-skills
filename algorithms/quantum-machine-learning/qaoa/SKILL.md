---
name: qaoa
description: Skill for understanding, using, and implementing the Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut problems via the QAOAAlgorithm class.
---

# Quantum Approximate Optimization Algorithm (QAOA)

## Purpose

QAOA is a hybrid quantum-classical algorithm for combinatorial optimization. This implementation solves the Max-Cut problem: partition graph vertices into two sets to maximize the number of cut edges. It alternates between cost-Hamiltonian and mixer-Hamiltonian evolution layers.

Use this skill when you need to:
- Solve Max-Cut or weighted graph partitioning with a quantum algorithm.
- Learn the QAOA workflow: cost encoding → variational layers → COBYLA optimization.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Encode the Max-Cut cost Hamiltonian: $H_C = -\frac{1}{2}\sum_{(u,v)\in E}(I - Z_u Z_v)$.
2. Build the QAOA circuit: $p$ alternating layers of $e^{-i\gamma_k H_C}$ and $e^{-i\beta_k H_{\text{mix}}}$.
3. Optimize $\gamma, \beta$ using COBYLA to minimize $\langle H_C\rangle$.
4. Extract the bit string with maximum probability as the cut solution.

## Prerequisites

- Graph theory (Max-Cut); Ising Hamiltonians.
- QAOA variational ansatz; Trotterized cost/mixer gates.
- COBYLA gradient-free optimizer.
- `numpy`, `torch`, `networkx`, `scipy.optimize`, `GateSequence`.

## Using the Provided Implementation

```python
from unitarylab.algorithms import QAOAAlgorithm

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
n_qubits = 6

algo = QAOAAlgorithm(seed=42)
result = algo.run(
    edges=edges,
    n_qubits=n_qubits,
    p_layers=4,
    max_iter=100,
    backend='torch'
)

print(f"Best cut value: {result['maxcut']}")
print(f"Best partition: {result['best_partition']}")
print(result['plot'])
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `42` | Random seed for initialization. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `edges` | `List[Tuple[int,int]]` | default graph | Edge list of the graph. Defaults to a 6-node example. |
| `n_qubits` | `int` | `6` | Number of qubits = number of graph vertices. |
| `p_layers` | `int` | `4` | Number of QAOA layers $p$. More layers = better approximation. |
| `max_iter` | `int` | `100` | COBYLA maximum iterations. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str\|None` | `None` | Output directory for plots and circuit. |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `maxcut` | `int` | the `maxcut_val` in running, Maximum cut value found. |
| `best_partition` | `str` | Bit string encoding the optimal partition. |
| `optimal_params` | `np.ndarray` | Optimized $[\gamma_1,\ldots,\gamma_p,\beta_1,\ldots,\beta_p]$. |
| `loss_history` | `List[float]` | $\langle H_C\rangle$ per iteration. |
| `circuit` | `GateSequence` | QAOA circuit at optimal parameters. |
| `circuit_path` | `str` | Path to circuit SVG. |
| `plot_path` | `str` | Path to training curve PNG. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`QAOAAlgorithm` in `algorithm.py` implements the QAOA hybrid loop in five stages using a Hamiltonian builder, a circuit builder, and COBYLA classical optimization.

**`run(edges, n_qubits, p_layers, max_iter, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | `_get_h_cost(edges, n_qubits)` builds $H_C$ as a `(2^n × 2^n)` NumPy array; `np.linalg.eigvalsh(h_cost)[0]` gives exact ground energy | Creates cost Hamiltonian and initial parameters |
| 2 — Circuit Mapping | `_build_circuit(initial_params, ..., backend)` builds example circuit for visualization | Architecture preview only |
| 3 — Optimization Loop | `minimize(obj_func, x0, method='COBYLA', maxiter=max_iter)` — `obj_func` rebuilds circuit, executes, computes `psi† H_C psi` | Core QAOA hybrid quantum-classical loop |
| 4 — Solution Decoding | Builds final circuit with `opt_res.x`; extracts `best_idx = argmax(|psi|²)`; decodes to bitstring; counts Max-Cut value | Greedy Max-Cut solution extraction |
| 5 — Export | `_generate_outputs(edges, best_bits, history, qc_draw, ...)` | Saves circuit PNG, loss curve, Max-Cut graph visualization |

**Helper Methods:**

- **`_get_h_cost(edges, n_qubits)`** — For each edge `(u,v)`: builds $Z_u \otimes Z_v$ via `np.kron` with identity padding; accumulates into `h_cost` matrix. Returns `(2^n, 2^n)` complex128 array.
- **`_build_circuit(params, n_qubits, edges, backend)`** — Creates `GateSequence(n_qubits, backend=backend)`. Applies `H` to all qubits (initial $|+\rangle^{\otimes n}$). Loops `p` QAOA layers: for each edge, applies `CX(u,v) → Rz(2γ, v) → CX(u,v)`; for each qubit, applies `Rx(2β, j)`.
- **`obj_func(p_flat)` (closure)** — Called by COBYLA. Calls `_build_circuit`, then `qc.execute(initial_state=|0⟩)`; `np.real(psi.conj().T @ h_cost @ psi)`.
- **`_generate_outputs`** — Saves three plots: circuit, energy convergence, and NetworkX Max-Cut graph.

**Data flow:** `edges` → `_get_h_cost()` → COBYLA(`obj_func`) → `opt_res.x` → final circuit → `argmax(|psi|²)` → bitstring → Max-Cut count → result dict.

## Understanding the Key Quantum Components
$$H_C = \frac{1}{2}\sum_{(u,v)\in E}(Z_u Z_v - I)$$
Implemented as: for each edge $(u,v)$, compute $ZZ + II$ Kronecker product applied to the $n$-qubit state. The ground state of $-H_C$ corresponds to the maximum cut.

### 2. Cost Layer $e^{-i\gamma H_C}$
For each edge $(u,v)$: `CX(u, v) → Rz(2γ, v) → CX(u, v)`. This implements $e^{-i\gamma Z_u Z_v}$ via phase kickback.

### 3. Mixer Layer $e^{-i\beta H_{\text{mix}}}$
$$H_{\text{mix}} = \sum_j X_j$$
Implemented as: `Rx(2β, j)` on each qubit $j$. This generates superpositions to explore the cut space.

### 4. Initial State $|+\rangle^{\otimes n}$
Each qubit starts in $|+\rangle = H|0\rangle$, an equal superposition of all $2^n$ bit strings — all possible partitions.

### 5. Approximation Ratio
QAOA at depth $p$ achieves approximation ratio $\geq \alpha_p$ for Max-Cut, where $\alpha_1 \approx 0.692$ and $\alpha_p \to 1$ as $p \to \infty$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Initial state $|+\rangle^{\otimes n}$ | `for i in range(n_qubits): qc.h(i)` in `_build_circuit` |
| Cost Hamiltonian $H_C = \sum_{(u,v)}Z_uZ_v$ | `_get_h_cost(edges, n_qubits)` — NumPy Kronecker sum |
| Cost layer $e^{-i\gamma Z_uZ_v}$ | `cx(u,v) → rz(2*gamma, v) → cx(u,v)` per edge per layer |
| Mixer layer $e^{-i\beta X_j}$ | `rx(2*beta, j)` for each qubit per layer |
| Parameters $(\gamma_1,\ldots,\gamma_p, \beta_1,\ldots,\beta_p)$ | `params[:p]` = gammas, `params[p:]` = betas (flat array) |
| Energy $\langle\psi|H_C|\psi\rangle$ | `np.real(psi.conj().T @ h_cost @ psi).item()` |
| COBYLA optimization | `minimize(obj_func, x0, method='COBYLA')` |
| Bitstring decoding | `best_idx = argmax(|psi|²)` → `format(best_idx, '0nb')` |
| Max-Cut count | `len([(u,v) for (u,v) in edges if best_bits[u] != best_bits[v]])` |
| Exact maximum energy $\lambda_\min(-H_C)$ | `np.linalg.eigvalsh(h_cost)[0]` (note: ground state of $H_C$ = minimum energy) |

**Notes on implementation:** The cost Hamiltonian here is $H_C = \sum_{(u,v)} Z_u Z_v$ (without the $1/2(Z_uZ_v - I)$ normalization used in some formulations). The Max-Cut solution is found by taking the most-probable basis state from the final statevector — this is a greedy extraction, not a full measurement simulation.

## Mathematical Deep Dive

**QAOA circuit:**
$$|\psi(\gamma, \beta)\rangle = e^{-i\beta_p H_{\text{mix}}} e^{-i\gamma_p H_C} \cdots e^{-i\beta_1 H_{\text{mix}}} e^{-i\gamma_1 H_C} |+\rangle^{\otimes n}$$

**Objective:**
$$\min_{\gamma,\beta} \langle\psi(\gamma,\beta)|H_C|\psi(\gamma,\beta)\rangle = -\frac{1}{2}\sum_{(u,v)\in E}(1 - \langle Z_u Z_v\rangle)$$

**Approximation ratio:** $r = \langle H_C\rangle_{\text{QAOA}} / C_{\text{max}}$ where $C_{\text{max}}$ is the true Max-Cut value.

**Total parameters:** $2p$ real numbers ($p$ gammas + $p$ betas).

## Hands-On Example

```python
from unitarylab.algorithms import QAOAAlgorithm

# Petersen graph-like structure
edges = [(0,1), (1,2), (2,3), (3,4), (4,0), (0,5), (1,6), (2,7), (3,8), (4,9)]
n_qubits = 10

algo = QAOAAlgorithm(seed=0)
result = algo.run(edges=edges, n_qubits=n_qubits, p_layers=2, max_iter=80)

print(f"Cut value: {result['maxcut']}")
print(f"Partition: {result['plots']['best_partition']}")
```

## Implementing Your Own Version

The following Python skeleton reconstructs the core QAOA components — the cost Hamiltonian builder, the QAOA circuit, and the hybrid optimization loop.

```python
# Simplified reconstruction — mirrors QAOAAlgorithm._get_h_cost(), _build_circuit(), and obj_func()
import numpy as np
from scipy.optimize import minimize
from unitarylab.core import GateSequence

def build_cost_hamiltonian(edges, n_qubits: int) -> np.ndarray:
    """Build H_C = sum_{(u,v) in E} Z_u Z_u as a (2^n x 2^n) matrix."""
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

def build_qaoa_circuit(params: np.ndarray, edges, n_qubits: int,
                        backend: str = 'torch') -> GateSequence:
    """QAOA circuit: H^n initial state → p layers of cost + mixer gates."""
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    qc = GateSequence(n_qubits, backend=backend)
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

def run_qaoa(edges, n_qubits: int, p_layers: int = 2,
             max_iter: int = 100, backend: str = 'torch', seed: int = 42):
    """Full QAOA hybrid loop for Max-Cut."""
    np.random.seed(seed)
    H_c = build_cost_hamiltonian(edges, n_qubits)
    params0 = np.random.uniform(0, np.pi, 2 * p_layers)
    history = []

    def obj(params):
        qc = build_qaoa_circuit(params, edges, n_qubits, backend)
        psi = np.asarray(qc.execute(
            initial_state=np.eye(2**n_qubits, 1, dtype=np.complex128)
        ))
        energy = float(np.real(psi.conj().T @ H_c @ psi))
        history.append(energy); return energy

    res = minimize(obj, x0=params0, method='COBYLA', options={'maxiter': max_iter})

    # Decode: most probable basis state as partition
    qc_final = build_qaoa_circuit(res.x, edges, n_qubits, backend)
    psi_final = np.asarray(qc_final.execute(
        initial_state=np.eye(2**n_qubits, 1, dtype=np.complex128)
    ))
    best_idx = int(np.argmax(np.abs(psi_final.flatten())**2))
    best_bits = format(best_idx, f'0{n_qubits}b')
    cut_val = sum(1 for u, v in edges if best_bits[u] != best_bits[v])
    return {'energy': res.fun, 'best_partition': best_bits, 'best_cut': cut_val}
```

## Debugging Tips

1. **`edges` contains invalid vertex indices**: Vertex indices must be in `[0, n_qubits)`. Out-of-range indices cause circuit construction errors.
2. **Low cut value with small `p_layers`**: QAOA quality improves with $p$. Use `p_layers=4+` for better approximation.
3. **COBYLA stuck in local minimum**: Re-run with different `seed`. COBYLA is sensitive to initialization.
4. **Large `n_qubits`**: Circuit simulation grows exponentially as $2^{n\_qubits}$. Practical limit is ~20 qubits.
5. **`edges=None`**: Uses the built-in default 6-node graph. Always pass explicit `edges` and matching `n_qubits`.
6. **Approximation ratio interpretation**: The bit string `best_partition` maps qubit $q$ to side 0 or 1. Count edges between different sides to verify `best_cut`.
