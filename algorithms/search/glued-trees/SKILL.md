---
name: glued-trees
description: "Use when users ask about the Glued Trees quantum walk, coupled-oscillator graph embedding, Pauli decomposition and cropping, Suzuki-Trotter evolution, exit-state sampling, or debugging UnitaryLab's GluedTreesAlgorithm. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and reference-guided reimplementation."
---

# Glued Trees Quantum Walk Algorithm

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement `GluedTreesAlgorithm`.

The implementation builds two randomly glued binary trees, maps their adjacency structure to a coupled-oscillator Hamiltonian, decomposes that dense Hamiltonian into Pauli strings, approximates its time evolution with a first- or second-order product formula, and estimates the probability of reaching the exit vertex.

When using this skill:

- **Explanation:** Explain the graph problem, oscillator embedding, Hamiltonian simulation, theoretical oracle separation, and implementation limitations. Do not generate code unless requested.
- **Run or reuse:** Generate standalone user code that imports `GluedTreesAlgorithm` from `unitarylab_algorithms`. Do not import from this skill's `scripts/` directory.
- **Debugging:** Start with `qubits=4`, a short explicit time list, low shots, and `max_terms=None`. Separate model, Pauli-truncation, Trotter, sampling, and backend errors.
- **Modification or reimplementation:** Preserve the graph labels, padded Cholesky factor, block Hamiltonian, source cropping rule, time semantics, exit-state convention, and return contract.
- **Reference scripts:** `scripts/algorithm.py` calls only the installed `unitarylab-algorithms` public API and never injects a workspace source path. `scripts/glued_trees_implementation.py` is a separate independent reconstruction and verification aid.
- **Validation:** Compare a small full-term Trotter circuit with dense `exp(-iHt)` and vary one approximation control at a time.

The theoretical glued-trees oracle problem has an exponential quantum/classical query separation. This package is a dense classical simulator demonstration and does not establish wall-clock quantum advantage.

## Overview

1. Construct two balanced binary trees and reverse the labels of the right tree.
2. Randomly connect every left leaf to two right leaves while keeping every right-leaf glue degree at most two.
3. Form adjacency `A`, stiffness `K = 3I - A`, Cholesky factor `B`, and the Hermitian oscillator embedding.
4. Decompose the Hamiltonian into real Pauli terms, sorted by coefficient magnitude.
5. Optionally crop the term list with the implemented 60% diversity / 40% magnitude rule.
6. For every requested time, build a fixed-step Suzuki product formula, execute it, and sample the exit basis state.
7. Save the circuit corresponding to the first occurrence of the largest sampled probability, a probability plot, and a text report.

## Prerequisites

- Binary trees, adjacency matrices, Cholesky factorization, and coupled oscillators.
- Pauli-string decomposition and Hamiltonian simulation.
- First- and second-order Suzuki-Trotter product formulas.
- Python, NumPy, NetworkX, Matplotlib, and UnitaryLab `Circuit`, `Register`, and `ClassicalRegister`.
- SciPy only for the independent script's dense `expm_multiply` validation, not for the public algorithm.

## Reference Implementation Example

This example resolves `unitarylab_algorithms` from the active Python environment's installed package. Do not prepend the repository workspace to `sys.path`.

```python
from unitarylab_algorithms import GluedTreesAlgorithm

algorithm = GluedTreesAlgorithm(text_mode="plain")
result = algorithm.run(
    qubits=4,
    seed=0,
    shots=256,
    repetitions=4,
    order=1,
    max_terms=None,
    backend="numpy",
    times="0,2,4",
)

print(result["status"])
print(result["Nodes"], result["Exit state"])
print(result["Pauli terms"])
print(result["Peak exit probability"], result["Peak time"])
print(result["Rows"])
print(result["circuit_path"])
print(result.get("plot", []))
```

## Core Parameters Explained

**Constructor `GluedTreesAlgorithm(text_mode, algo_dir)`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Base result-report formatting mode. |
| `algo_dir` | `str` or `None` | `None` | Output directory. When omitted, uses `results/search/glued_trees` under the current working directory. |

**`run(qubits, seed, shots, repetitions, order, max_terms, backend, times)`:**

| Parameter | Type | Default | Source contract |
|---|---|---:|---|
| `qubits` | integer-like | `6` | Converted with `int()`; `build_glued_trees()` requires 4 through 8. |
| `seed` | integer-like | `0` | Non-negative; controls graph glue and per-time execution/sampling. |
| `shots` | integer-like | `8192` | Positive samples per time point. |
| `repetitions` | integer-like | `10` | Positive product-formula repetitions for each time. |
| `order` | integer-like | `1` | Exactly 1 or 2; checked by `make_trotter_step()`. |
| `max_terms` | integer-like, blank, or `None` | `None` | Blank, `None`, or `"None"` retains all terms; otherwise must be positive. |
| `backend` | `str` | `"numpy"` | Exactly `"numpy"` or `"torch"`; device is fixed to CPU. |
| `times` | `None`, blank, CSV, scalar, or sequence | `None` | Must normalize to a non-empty finite one-dimensional float array. |

Default times are

```text
2 * (qubits - 2) + [-12, -10, ..., 10, 12].
```

Negative times are accepted by the callable implementation. `parameters.json` constrains UI text entry but `algorithm.py` defines runtime behavior.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` whenever the workflow completes; no minimum success-probability threshold is applied. |
| `circuit_path` | `str` | Saved SVG for the representative sampled-peak circuit. |
| `plot` | `list[dict]` | Text-report and PNG descriptors. Paths are filenames as supplied to the base return builder. |
| `circuit` | `Circuit` | Representative circuit. |
| `Nodes` | `int` | Number of graph vertices. |
| `Exit state` | `str` | `qubits`-bit representation of `node_count - 1`. |
| `Pauli terms` | `str` | Selected and full term counts in `selected/full` form. |
| `Peak exit probability` | `float` | Maximum sampled exit frequency. |
| `Peak time` | `float` | First time achieving the maximum sampled frequency. |
| `Rows` | `list[dict]` | One timing/probability record per time point. |
| `Model construction (s)` | `float` | Model construction wall time. |
| `Pauli decomposition (s)` | `float` | Pauli decomposition wall time. |
| `Total time (s)` | `float` | Total wall time. |

Each row contains `time`, `sampled_probability`, `exit_count`, `build_seconds`, `execute_seconds`, and `sample_seconds`.

## Implementation Architecture

`GluedTreesAlgorithm.run()` has four logged stages and several public module-level helpers.

| Stage | Code action | Algorithmic role |
|---|---|---|
| 1 — Model | `build_glued_trees(qubits, seed)` | Construct graph, `A`, `K`, factor `B`, and `H`. |
| 2 — Decomposition | `decompose_hamiltonian()` then `crop_pauli_list()` | Produce the complete or cropped real Pauli expansion. |
| 3 — Scan | Build, execute, and sample one circuit per parsed time | Estimate entrance-to-exit transport and track the sampled peak. |
| 4 — Export | Save representative circuit, PNG plot, and text report | Produce the standardized artifact and result contract. |

**Key helpers:**

- `build_glued_trees()` validates the dense demonstration size, uses a local `random.Random(seed)`, and returns `GluedTreesModel`.
- `decompose_hamiltonian()` calls `pauli_string_decomposition(..., partition_commuting=False, real_symmetric_hint=True)` and sorts by `(-abs(coefficient), word)`.
- `crop_pauli_list()` does not approximate by magnitude alone: it first attempts a rounded 60% quota across `I/X/Y/Z` letters and qubit positions, then fills remaining slots in sorted order.
- `make_trotter_step()` caches repeated `(word, coefficient * dt)` evolution blocks. Order 2 uses half-strength copies of every term except the last, the last full term, then the reversed half sequence.
- `make_glued_trees_circuit()` prepares the default all-zero state and applies the step `repetitions` times unless `time == 0`.
- `parse_times()` implements all accepted input forms and the default scan.
- `plot_results()` plots sampled probability rather than an exact probability.

**Data flow:** `qubits, seed` → graph/model → full Pauli list → optional crop → time array → circuits → samples → rows/peak → artifacts/result.

## Understanding the Key Quantum Components

### 1. Glued graph and entrance/exit

For `columns = qubits - 2`, each tree has depth `columns - 1` and the union has

```text
N = 2 ** (columns + 1) - 2 = 2 ** (qubits - 1) - 2
```

vertices. Vertex 0 is the entrance. The oscillator velocity-block exit basis index is `N - 1`, not the final computational basis state `2**qubits - 1`.

### 2. Coupled-oscillator embedding

With adjacency `A`, the source constructs

```text
K = 3I - A = B B†
H = -[[0, B], [B†, 0]].
```

`B` is the Cholesky factor padded with four zero columns. This padding is exact for the implemented node/qubit relation and embeds the oscillator state in a `2**qubits`-dimensional Hilbert space.

### 3. Pauli decomposition and cropping

The dense real symmetric `H` is expanded as `H = sum_j c_j P_j`. The optional `max_terms` crop changes `H`; it is a Hamiltonian approximation whose error is independent of product-formula error.

### 4. Product-formula evolution

For first order and `dt = time / repetitions`, the circuit approximates

```text
exp(-i H time) ~= (product_j exp(-i c_j P_j dt)) ** repetitions.
```

Second order uses the source's symmetric half/full/half ordering. Increasing repetitions reduces product-formula error but cannot restore omitted Pauli terms.

### 5. Exit-state sampling

The circuit is executed once per time and `result.sample()` produces `shots` bit strings. The source uses `counts.get(exit_bits, 0) / shots` directly and does not reverse sample strings in this algorithm.

## Theory-to-Code Mapping

| Theory or contract | Source object or expression |
|---|---|
| Two balanced trees | `nx.balanced_tree(2, columns - 1)` and reversed relabeling |
| Random degree-two glue | `rng.sample(candidates, 2)` with the right-leaf degree map |
| Adjacency matrix | `nx.to_numpy_array(..., nodelist=range(nodes))` |
| Stiffness `3I-A` | `stiffness = 3 * np.eye(nodes) - adjacency` |
| Factor `B` | Padded `np.linalg.cholesky(stiffness)` |
| Hermitian embedding | Off-diagonal assignments to `hamiltonian` |
| Pauli expansion | `pauli_string_decomposition()` |
| Optional truncation | `crop_pauli_list(full_terms, max_terms)` |
| One product-formula step | `make_trotter_step()` |
| Full time circuit | `make_glued_trees_circuit()` |
| Exit string | `format(model.exit_state, f"0{qubits}b")` |
| Sampled probability | `counts.get(exit_bits, 0) / shots` |
| Representative circuit | Circuit retained when a strictly higher sampled probability is observed |

## Mathematical Deep Dive

The glued-trees problem hides a path between roots behind exponentially many locally similar branches. The continuous-time quantum walk exploits interference in the column-symmetric subspace. This implementation reaches a related unitary evolution through a coupled-oscillator formulation.

The stiffness matrix is positive definite for the implemented finite graph, allowing `K = BB†`. If oscillator displacement and velocity coordinates are combined into a first-order state, the block matrix formed from `B` generates Hermitian quantum evolution. The algorithm embeds this matrix into a power-of-two dimension, decomposes it into Pauli strings, and simulates the exponential by product formulas.

For the dense implementation:

- Hamiltonian storage is `O(4**qubits)`.
- A complete Pauli basis contains up to `4**qubits` strings.
- With `L` times, `r` repetitions, and `m` retained terms, a first-order circuit contains approximately `L * r * m` Pauli-evolution blocks.
- State-vector simulation remains exponential in `qubits`.

The sampled peak is affected by finite shots, Pauli cropping, product-formula error, and the selected time grid. Report those controls with any numerical conclusion.

## Hands-On Example

```python
from unitarylab_algorithms import GluedTreesAlgorithm

result = GluedTreesAlgorithm().run(
    qubits=4,
    seed=0,
    shots=512,
    repetitions=8,
    order=2,
    max_terms=None,
    backend="numpy",
    times=[0.0, 2.0, 4.0],
)

assert result["status"] == "ok"
assert result["Nodes"] == 6
assert result["Exit state"] == "0101"
assert len(result["Rows"]) == 3
assert all(0.0 <= row["sampled_probability"] <= 1.0 for row in result["Rows"])
```

For numerical validation, hold `max_terms=None` and physical time fixed while increasing `repetitions`; then hold the Trotter settings fixed while increasing `max_terms`. Do not change both controls in the same comparison.

## Minimal Manual Implementation

The following is a self-contained minimal implementation of the same Glued Trees algorithm. It does not import `GluedTreesAlgorithm`, the source module, or this skill's reference scripts. It rebuilds the graph, oscillator Hamiltonian, Pauli expansion, Suzuki circuit, execution loop, and exit-state statistic from UnitaryLab circuit primitives and the source equations.

```python
from collections import Counter
import random

import networkx as nx
import numpy as np
from unitarylab import Circuit, ClassicalRegister, Register
from unitarylab.library.pauli_operator import (
    pauli_string_decomposition,
    pauli_string_evolution,
)


def build_model(qubits, seed):
    """Rebuild A, K, B and H=-[[0,B],[B.T,0]] from the source."""
    if not 4 <= qubits <= 8:
        raise ValueError("dense demonstration requires 4 <= qubits <= 8")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    columns = qubits - 2
    nodes = 2 ** (columns + 1) - 2

    left = nx.balanced_tree(2, columns - 1)
    right = nx.relabel_nodes(left, lambda vertex: nodes - 1 - vertex)
    graph = nx.union(left, right)

    leaf_count = 2 ** (columns - 1)
    left_leaves = range(leaf_count - 1, 2 * leaf_count - 1)
    right_leaves = range(2 * leaf_count - 1, 3 * leaf_count - 1)
    right_degrees = dict.fromkeys(right_leaves, 0)
    rng = random.Random(seed)

    for left_leaf in left_leaves:
        candidates = [v for v in right_leaves if right_degrees[v] == 0]
        if not candidates:
            candidates = [v for v in right_leaves if right_degrees[v] < 2]
        for right_leaf in rng.sample(candidates, 2):
            graph.add_edge(left_leaf, right_leaf)
            right_degrees[right_leaf] += 1

    adjacency = nx.to_numpy_array(graph, nodelist=range(nodes), dtype=float)
    stiffness = 3 * np.eye(nodes) - adjacency
    factor = np.pad(np.linalg.cholesky(stiffness), ((0, 0), (0, 4)))

    hamiltonian = np.zeros((2**qubits, 2**qubits))
    hamiltonian[:nodes, nodes:] = -factor
    hamiltonian[nodes:, :nodes] = -factor.T
    return graph, hamiltonian


def pauli_terms(hamiltonian):
    """Perform the same complete, real, magnitude-sorted Pauli expansion."""
    raw_terms = pauli_string_decomposition(
        hamiltonian,
        partition_commuting=False,
        real_symmetric_hint=True,
    )
    terms = [(word, float(np.real(coeff))) for word, coeff in raw_terms]
    return sorted(terms, key=lambda term: (-abs(term[1]), term[0]))


def suzuki_step(qubits, terms, dt, order=1):
    """Build the source first- or second-order Pauli product formula."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    step = Circuit(qubits, name=f"Suzuki order {order}, dt={dt:g}")
    if dt == 0:
        return step

    sequence = list(terms)
    if order == 2 and sequence:
        half = [(word, coeff / 2) for word, coeff in sequence[:-1]]
        sequence = half + sequence[-1:] + list(reversed(half))

    cached_blocks = {}
    for word, coefficient in sequence:
        key = (word, coefficient * dt)
        if key not in cached_blocks:
            cached_blocks[key] = pauli_string_evolution(*key)
        step.append(cached_blocks[key], target=list(range(qubits)))
    return step


def evolution_circuit(qubits, terms, time, repetitions, order):
    """Prepare entrance |0...0> and evolve it for physical time t."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    quantum = Register("state", qubits)
    classical = ClassicalRegister("result", qubits)
    circuit = Circuit(quantum, classical, name=f"Glued Trees t={time:g}")
    step = suzuki_step(qubits, terms, time / repetitions, order)
    if time != 0:
        for _ in range(repetitions):
            circuit.append(step, target=quantum[:])
    return circuit


def run_glued_trees(
    qubits=4,
    seed=0,
    shots=256,
    repetitions=4,
    order=1,
    times=(0.0, 2.0, 4.0),
    backend="numpy",
):
    """Execute the minimal full-term time scan and return the sampled peak."""
    if shots <= 0 or repetitions <= 0 or seed < 0:
        raise ValueError("shots/repetitions must be positive; seed non-negative")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)):
        raise ValueError("times must be a non-empty finite 1D sequence")

    graph, hamiltonian = build_model(qubits, seed)
    terms = pauli_terms(hamiltonian)  # full expansion: no truncation error
    exit_bits = format(graph.number_of_nodes() - 1, f"0{qubits}b")
    rows = []

    for index, time in enumerate(times):
        circuit = evolution_circuit(
            qubits, terms, float(time), repetitions, order
        )
        result = circuit.execute(
            backend=backend, device="cpu", seed=seed + index
        )
        counts = Counter(result.sample(shots=shots, seed=seed + index))
        exit_count = counts.get(exit_bits, 0)
        rows.append(
            {
                "time": float(time),
                "exit_count": exit_count,
                "sampled_probability": exit_count / shots,
                "circuit": circuit,
            }
        )

    peak = max(rows, key=lambda row: row["sampled_probability"])
    return {
        "nodes": graph.number_of_nodes(),
        "exit_state": exit_bits,
        "pauli_terms": len(terms),
        "rows": rows,
        "peak_time": peak["time"],
        "peak_exit_probability": peak["sampled_probability"],
        "representative_circuit": peak["circuit"],
    }


result = run_glued_trees()
assert result["nodes"] == 6
assert result["exit_state"] == "0101"
print(result["peak_time"], result["peak_exit_probability"])
```

This minimal version intentionally retains all Pauli terms and uses explicit times so that the algorithmic path is visible. The full independent reference additionally implements the source `max_terms` diversity crop, default-time parser, stage timings, dense `expm_multiply` comparison, and CLI checks:

```bash
python scripts/glued_trees_implementation.py
python scripts/glued_trees_implementation.py --execute --order 2
```

The default command prints a structural report rather than a single pass/fail marker: graph nodes and edges, entrance and exit states, matrix shapes, symmetry/Hermiticity residuals, the minimum stiffness eigenvalue, full and demonstration-cropped Pauli counts, leading Pauli terms, product-formula block counts, parsed time grids, and the dense `t=0` reference. With `--execute`, it additionally prints every time point's exit count, sampled probability, exact dense probability, absolute difference, per-stage timings, and sampled peak.

## Debugging Tips

1. **`qubits` rejected:** The callable dense demonstration supports only 4 through 8.
2. **Decomposition or simulation is slow:** Start at 4 qubits and a short explicit time list. Full terms are the correctness baseline; use cropping only after establishing it.
3. **Peak changes between runs:** Sampling uses `seed + time_index`; confirm the time order, seed, and shots before diagnosing physics.
4. **Trotter convergence stalls:** Check whether `max_terms` cropped the Hamiltonian. Repetitions do not reduce truncation error.
5. **Unexpected exit string:** Use `format(node_count - 1, f"0{qubits}b")`; do not use the all-ones basis state.
6. **`status == "ok"` but probability is small:** Status indicates successful completion only.
7. **Current environment rejects `seed` in `Circuit.execute()`:** The installed UnitaryLab is older than the algorithm source contract. Use a compatible version; do not remove the documented `seed` behavior.

The authoritative callable is `unitarylab_algorithms/search/glued_trees/algorithm.py`. `parameters.json` describes the UI and the bilingual READMEs explain intent, but neither overrides implemented Python behavior.
