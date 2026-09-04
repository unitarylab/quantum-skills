---
name: ising
description: Simulate open-boundary two-dimensional transverse-field Ising dynamics with local Trotter-Suzuki circuits and the UnitaryLab TensorNet backend.
---

# 2D Transverse-Field Ising Dynamics

## How to Use This Skill

Use this skill when explaining, running, debugging, or modifying `IsingAlgorithm`. The implementation builds local circuits for an open-boundary two-dimensional transverse-field Ising model, evolves a TensorNet backend state over Trotter steps, computes global and optional local observables, and saves plots or GIFs.

For user-facing code, import `IsingAlgorithm` from `new.ising.algorithm` and use its public `run()` method. Do not import reference/example scripts as runtime dependencies. Theory-only explanations do not require execution; numerical verification requires the UnitaryLab TensorNet backend.

## Scope and Limitations

The implementation supports an open-boundary rectangular lattice with one qubit per site and Hamiltonian `H = -J sum(ZZ) - h sum(X)`. It supports first-, second-, and fourth-order Trotter-Suzuki steps, TensorNet execution with configurable bond dimension and cutoff, global observables, optional local configuration maps, and optional final-state sampling.

The `1×1` lattice is rejected because it has no nearest-neighbor edge. The implementation uses a default `|0>^N` initial state and does not expose a general initial-state parameter. TensorNet truncation is controlled by `max_bond` and `cutoff`; results can differ from exact statevector dynamics for strongly entangled states. This is a dynamics simulator and observable workflow, not a claim of universal quantum advantage or thermodynamic-limit accuracy.

## Overview

For a `rows × cols` lattice, map sites to qubits in row-major order and enumerate horizontal and vertical nearest-neighbor edges. Decompose the Hamiltonian into `H_ZZ` and `H_X`, compile local `ZZ` and `X` evolutions into a Trotter step, and execute that step repeatedly:

1. Validate physical, Trotter, TensorNet, sampling, and output-mode inputs.
2. Build the lattice model and four edge matchings.
3. Construct a first-, second-, or fourth-order Trotter circuit.
4. Execute the empty circuit to obtain the default initial TensorNet state.
5. Reuse `result.backend_state` as the next step's `initial_state`.
6. Evaluate magnetizations, nearest-neighbor correlation, domain-wall density, energy, and optional local maps.
7. Save circuit/observable/configuration outputs and return the trajectory dictionary.

## Prerequisites

- Basic Pauli operators, real-time Hamiltonian evolution, and Trotter-Suzuki formulas.
- Familiarity with two-dimensional lattice indexing and nearest-neighbor interactions.
- Python, NumPy, Matplotlib, and the public UnitaryLab `Circuit` and result APIs; SciPy is used only by the manual implementation's dense exact reference.
- A working TensorNet backend; the algorithm passes `backend="tensornet"` and `device="cpu"` internally.
- Understanding that TensorNet bond dimension and cutoff introduce approximation controls.

## API Example

```python
from unitarylab_algorithms.new.ising.algorithm import IsingAlgorithm

algorithm = IsingAlgorithm(text_mode="plain")
result = algorithm.run(
    rows=2,
    cols=2,
    coupling_J=1.0,
    field_h=3.0,
    time_step=0.025,
    trotter_steps=2,
    trotter_order=2,
    max_bond=32,
    cutoff=1e-12,
    routing="auto",
    seed=7,
    sample_shots=0,
    local_configuration_mode="gif",
)

print(result["status"])
print(result["final_observables"])
```

The constructor also accepts `algo_dir` to control output location. The default `run()` values are `6×6`, `J=1.0`, `h=3.0`, `time_step=0.025`, `trotter_steps=8`, `trotter_order=2`, `max_bond=32`, `cutoff=1e-12`, `routing="auto"`, `seed=7`, `sample_shots=0`, and `local_configuration_mode="gif"`.

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `rows` | `int` | `6` | Number of lattice rows; must be positive. |
| `cols` | `int` | `6` | Number of lattice columns; must be positive and not form `1×1` with `rows`. |
| `coupling_J` | finite `float` | `1.0` | Nearest-neighbor coupling in `H = -J sum(ZZ) - h sum(X)`. |
| `field_h` | finite `float` | `3.0` | Transverse-field strength. |
| `time_step` | positive `float` | `0.025` | Duration of one Trotter step. |
| `trotter_steps` | non-negative `int` | `8` | Number of evolution steps; zero records only the initial state. |
| `trotter_order` | `1`, `2`, or `4` | `2` | Trotter-Suzuki order. |
| `max_bond` | positive `int` | `32` | Maximum TensorNet bond dimension. |
| `cutoff` | non-negative `float` | `1e-12` | Singular-value truncation cutoff. |
| `routing` | non-empty `str` | `"auto"` | TensorNet routing strategy passed to UnitaryLab. |
| `seed` | `int` | `7` | Execution and sampling seed. |
| `sample_shots` | non-negative `int` | `0` | Final-state samples; zero disables sampling. |
| `local_configuration_mode` | `"none"`, `"final"`, `"all"`, or `"gif"` | `"gif"` | Controls local `<X_i>`/`<Z_i>` maps and output animation. |

The input validator rejects non-finite couplings/fields, invalid Trotter orders, non-positive bond dimensions, negative cutoffs, invalid routing strings, negative sample counts, and unknown local-configuration modes.

## Return Fields

The public result is a dictionary assembled by `BaseAlgorithm` and then extended with trajectory data:

| Field | Type | Presence and meaning |
|---|---|---|
| `status` | `str` | Always present; normally `"ok"` for a completed run. |
| `circuit_path` | `str` | Always present; saved SVG path for one Trotter-step circuit. |
| `plot` | `list[dict[str, str]]` | Always present; file descriptors with `format` and `filename` for the observable plot, optional configuration image/GIF, and text summary. |
| `circuit` | `Circuit` | Always present; the public `Circuit` object for one Trotter step. |
| `key` | `str` | Always present; run identifier containing the model size, Trotter order, bond dimension, time step, and step count. |
| `model` | `str` | Always present; lattice-size label in `"{rows}x{cols}"` form. |
| `rows` | `int` | Always present; number of lattice rows. |
| `cols` | `int` | Always present; number of lattice columns. |
| `num_qubits` | `int` | Always present; `rows * cols`. |
| `max_bond` | `int` | Always present; TensorNet maximum bond dimension used for the run. |
| `trotter_order` | `int` | Always present; selected product-formula order (`1`, `2`, or `4`). |
| `time_step` | `float` | Always present; duration of one Trotter step. |
| `trotter_steps` | `int` | Always present; number of evolution steps. |
| `records` | `list[dict[str, object]]` | Always present; the initial observation followed by one record after each Trotter step, so its length is `trotter_steps + 1`. |
| `final_observables` | `dict[str, object]` | Always present; a copy of `records[-1]`. |
| `local_configuration` | `dict[str, list[list[float]]]` | Present only when `local_configuration_mode != "none"`; contains final `z_map` and `x_map`. |
| `samples` | `list` | Always present; final-state samples when `sample_shots > 0`, otherwise `[]`. |
| `wall_seconds` | `float` | Always present; measured trajectory execution time in seconds. |
| `step_circuit_summary` | implementation-defined summary object | Always present; result of the public circuit-analysis summary for the Trotter step. |

Each `records` entry contains `step: int`, `time: float`, `magnetization_z: float`, `magnetization_x: float`, `mean_zz: float`, `domain_wall_density: float`, and `energy_per_site: float`. When `local_configuration_mode="all"`, every record additionally contains `z_map` and `x_map`, each a `rows x cols` nested list of floats. `final_observables` therefore also contains `step` and `time`, and in `all` mode it includes those two local maps.

`local_configuration_records` is an internal GIF-generation field from `run_trajectory()` and is removed before `IsingAlgorithm.run()` returns. Likewise, the internal `step_circuit` field is removed and exposed through the public `circuit` field instead. Neither internal field is part of the public return contract.

## Implementation Architecture

`IsingAlgorithm.run()` performs four high-level stages:

1. Validate inputs, clear prior algorithm state, and construct `IsingModel` and `SimulationSettings`.
2. Build one Trotter step through `build_trotter_step()`, using row-major qubit IDs and four non-overlapping edge groups.
3. Run `run_trajectory()`: execute the empty circuit for `|0>^N`, observe the current result, then execute the step circuit with the previous public `backend_state` as `initial_state`.
4. Save the circuit, observable plot, optional local image/GIF, text summary, and return the merged result dictionary.

The implementation does not read or manually contract internal MPS tensors. It uses public `result.expectation()`, `result.backend_state`, and optional `result.sample()` interfaces.

## Understanding the Key Quantum Components

### Lattice mapping and edge groups

Site `(row, col)` maps to `row * cols + col`. Open-boundary horizontal and vertical edges are split into four parity groups so gates within one group share no qubits.

### `ZZ` evolution

Each nearest-neighbor term is implemented with `CNOT-RZ-CNOT`. The code uses the angle `-2 * J * duration` to represent the corresponding `exp(-i H_ZZ duration)` block.

### Transverse-field evolution

Each site receives an `RX` rotation with angle `-2 * h * duration`, implementing the `H_X = -h sum(X)` contribution.

### Trotter-Suzuki composition

Order 1 applies `X` then `ZZ`. Order 2 uses a symmetric `X/2 -> ZZ -> X/2` block. Order 4 composes five second-order blocks with the Suzuki-Yoshida coefficient `p = 1/(4 - 4**(1/3))`, including the negative central substep.

### TensorNet continuation and observables

The result from one step supplies `backend_state` to the next step. Global Pauli sums are evaluated through `expectation()`, while local maps are evaluated one site at a time only when requested.

## Theory-to-Code Mapping

| Theory | Implementation |
|---|---|
| `H = -J sum_<i,j> Z_i Z_j - h sum_i X_i` | `IsingModel`, `append_zz_evolution()`, and `append_x_evolution()`. |
| Open-boundary nearest neighbors | `lattice_edges(rows, cols)`. |
| Non-overlapping gate layers | `edge_groups(rows, cols)`. |
| `exp(-i angle Z⊗Z/2)` | `append_zz()` using CNOT-RZ-CNOT. |
| `exp(-i H_X duration)` | Per-qubit `RX(-2*h*duration)`. |
| First-order product formula | `build_trotter_step(..., order=1)`. |
| Second-order Strang formula | `append_second_order_block()` and `order=2`. |
| Fourth-order Suzuki-Yoshida formula | Five weighted second-order blocks in `order=4`. |
| Magnetizations and correlations | `observable_sums()` plus `result.expectation()`. |
| Domain-wall density | `(1 - mean_zz) / 2`. |
| Energy per site | `(-J*edge_count*mean_zz - h*N*magnetization_x) / N`. |
| Tensor-network evolution | `Circuit.execute(backend="tensornet", ...)` with `initial_state=result.backend_state`. |

## Mathematical Deep Dive

For `N = rows * cols` qubits and open-boundary nearest-neighbor edges,

```text
H = H_ZZ + H_X
H_ZZ = -J sum_<i,j> Z_i Z_j
H_X  = -h sum_i X_i.
```

The first-order approximation is

```text
S1(dt) = exp(-i H_ZZ dt) exp(-i H_X dt),
```

where the rightmost operation acts first on the state. The second-order approximation is

```text
S2(dt) = exp(-i H_X dt/2) exp(-i H_ZZ dt) exp(-i H_X dt/2).
```

The fourth-order composition is

```text
S4(dt) = S2(p dt)^2 S2((1 - 4p) dt) S2(p dt)^2,
p = 1 / (4 - 4**(1/3)).
```

The reported observables are

```text
Mz = (1/N) sum_i <Zi>
Mx = (1/N) sum_i <Xi>
mean_ZZ = (1/Ne) sum_<i,j> <Zi Zj>
domain_wall_density = (1 - mean_ZZ) / 2
E/N = (-J Ne mean_ZZ - h N Mx) / N.
```

The numerical result is affected by Trotter error, TensorNet truncation (`max_bond`, `cutoff`), routing, lattice size, and evolution time. The implementation's plots and observables do not by themselves establish convergence.

## Hands-On Example

```python
from unitarylab_algorithms.new.ising.algorithm import IsingAlgorithm

result = IsingAlgorithm().run(
    rows=2,
    cols=2,
    coupling_J=1.0,
    field_h=3.0,
    time_step=0.025,
    trotter_steps=2,
    trotter_order=2,
    max_bond=32,
    cutoff=1e-12,
    routing="auto",
    seed=7,
    local_configuration_mode="gif",
)

assert result["status"] == "ok"
assert len(result["records"]) == 3
assert set(result["final_observables"]) >= {
    "magnetization_z",
    "magnetization_x",
    "mean_zz",
    "domain_wall_density",
    "energy_per_site",
}
```

Use three separate checks; changing all approximation controls at once does not identify the error source:

1. **Trotter convergence:** hold the physical final time `T = time_step * trotter_steps`, lattice, and TensorNet settings fixed. For example, compare `(time_step, trotter_steps) = (0.05, 4)`, `(0.025, 8)`, and `(0.0125, 16)` at `T=0.2`. Record the maximum absolute difference in `magnetization_z`, `magnetization_x`, `mean_zz`, and `energy_per_site` between successive refinements. The differences should decrease before TensorNet error dominates.
2. **TensorNet convergence:** hold the complete Trotter circuit fixed, then increase `max_bond` such as `16 -> 32 -> 64` and tighten `cutoff` such as `1e-8 -> 1e-10 -> 1e-12`. Compare the same observables at every recorded time. Stability under both changes is evidence that truncation is controlled for that run.
3. **Exact small-system reference:** for `2×2`, construct the same Hamiltonian and compute `exp(-iHT)|0000>` with a dense statevector or `scipy.linalg.expm`, then evaluate the five documented observables. Compare this exact evolution both with the Trotterized dense circuit, which isolates product-formula error, and with the TensorNet trajectory, which additionally tests backend continuation and truncation.

Report the lattice, total time, Trotter order, step size, bond dimension, cutoff, and observed error norm with every comparison. Do not claim convergence from `status == "ok"` or from one parameter setting, and do not impose a universal tolerance without reference-scale evidence.

## Minimal Manual Implementation

The following implementation is self-contained except for the public UnitaryLab API. It must not import `new.ising.algorithm` or use undefined pseudocode helpers. It constructs and executes the same local quantum circuit as `IsingAlgorithm`.

```python
import math
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from unitarylab import Circuit


@dataclass(frozen=True)
class IsingModel:
    rows: int
    cols: int
    coupling_J: float
    field_h: float

    @property
    def num_qubits(self):
        return self.rows * self.cols


def qubit_id(row, col, cols):
    """Row-major site-to-qubit mapping."""
    return row * cols + col


def lattice_edges(rows, cols):
    """All open-boundary horizontal and vertical nearest-neighbor edges."""
    horizontal = [
        (qubit_id(r, c, cols), qubit_id(r, c + 1, cols))
        for r in range(rows)
        for c in range(cols - 1)
    ]
    vertical = [
        (qubit_id(r, c, cols), qubit_id(r + 1, c, cols))
        for r in range(rows - 1)
        for c in range(cols)
    ]
    return horizontal + vertical


def edge_groups(rows, cols):
    """Four matchings; gates inside one group never share a qubit."""
    groups = [[], [], [], []]
    for r in range(rows):
        for c in range(cols - 1):
            first = qubit_id(r, c, cols)
            groups[c % 2].append((first, first + 1))
    for r in range(rows - 1):
        for c in range(cols):
            first = qubit_id(r, c, cols)
            groups[2 + r % 2].append((first, first + cols))
    return groups


def append_zz(circuit, first, second, angle):
    """CNOT-RZ-CNOT implements exp(-i angle*Z_i*Z_j/2)."""
    circuit.cx(first, second)
    circuit.rz(angle, second)
    circuit.cx(first, second)


def append_x_evolution(circuit, model, duration):
    """exp(-i H_X duration), H_X=-h sum_i X_i."""
    for qubit in range(model.num_qubits):
        circuit.rx(-2.0 * model.field_h * duration, qubit)


def append_zz_evolution(circuit, model, duration):
    """exp(-i H_ZZ duration), H_ZZ=-J sum_<i,j> Z_i Z_j."""
    angle = -2.0 * model.coupling_J * duration
    for group in edge_groups(model.rows, model.cols):
        for first, second in group:
            append_zz(circuit, first, second, angle)


def append_second_order_block(circuit, model, duration):
    """Strang block X(duration/2) -> ZZ(duration) -> X(duration/2)."""
    append_x_evolution(circuit, model, duration / 2.0)
    append_zz_evolution(circuit, model, duration)
    append_x_evolution(circuit, model, duration / 2.0)


def build_trotter_step(model, time_step, order=2):
    """Build one first-, second-, or fourth-order product-formula circuit."""
    if order not in (1, 2, 4):
        raise ValueError("order must be 1, 2, or 4")
    circuit = Circuit(model.num_qubits)
    if order == 1:
        # Gates execute in append order; matrix action is U_ZZ U_X |psi>.
        append_x_evolution(circuit, model, time_step)
        append_zz_evolution(circuit, model, time_step)
    elif order == 2:
        append_second_order_block(circuit, model, time_step)
    else:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        for weight in (p, p, 1.0 - 4.0 * p, p, p):
            append_second_order_block(circuit, model, weight * time_step)
    return circuit


def real_expectation(result, observable, qubits=None):
    value = (
        result.expectation(observable, qubits)
        if qubits is not None
        else result.expectation(observable)
    )
    value = complex(value)
    if abs(value.imag) > 1e-10:
        raise RuntimeError(f"expectation is not real: {value}")
    return float(value.real)


def observe(result, model):
    """Return global Mz, Mx, mean ZZ, domain walls, and energy/site."""
    edges = lattice_edges(model.rows, model.cols)
    N = model.num_qubits
    Mz_terms = [(1.0 / N, "Z", [q]) for q in range(N)]
    Mx_terms = [(1.0 / N, "X", [q]) for q in range(N)]
    ZZ_terms = [(1.0 / len(edges), "ZZ", [i, j]) for i, j in edges]
    magnetization_z = real_expectation(result, Mz_terms)
    magnetization_x = real_expectation(result, Mx_terms)
    mean_zz = real_expectation(result, ZZ_terms)
    energy = (
        -model.coupling_J * len(edges) * mean_zz
        -model.field_h * N * magnetization_x
    )
    return {
        "magnetization_z": magnetization_z,
        "magnetization_x": magnetization_x,
        "mean_zz": mean_zz,
        "domain_wall_density": (1.0 - mean_zz) / 2.0,
        "energy_per_site": energy / N,
    }


def observe_local_configuration(result, model):
    z = [real_expectation(result, "Z", [q]) for q in range(model.num_qubits)]
    x = [real_expectation(result, "X", [q]) for q in range(model.num_qubits)]
    return {
        "z_map": [z[r*model.cols:(r+1)*model.cols] for r in range(model.rows)],
        "x_map": [x[r*model.cols:(r+1)*model.cols] for r in range(model.rows)],
    }


def validate_observables(values, tolerance=1e-9):
    """Check finite values and the natural ranges of Pauli observables."""
    for key, value in values.items():
        if not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite observable {key}: {value}")
    for key in ("magnetization_z", "magnetization_x", "mean_zz"):
        if not -1.0 - tolerance <= values[key] <= 1.0 + tolerance:
            raise RuntimeError(f"{key} is outside [-1, 1]: {values[key]}")
    if not -tolerance <= values["domain_wall_density"] <= 1.0 + tolerance:
        raise RuntimeError(
            "domain_wall_density is outside [0, 1]: "
            f"{values['domain_wall_density']}"
        )


def validate_inputs(rows, cols, J, h, dt, steps, order, max_bond, cutoff,
                    routing, seed, sample_shots, save_local):
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if not isinstance(cols, int) or isinstance(cols, bool) or cols <= 0:
        raise ValueError("cols must be a positive integer")
    if rows == cols == 1:
        raise ValueError("the lattice must have at least one edge")
    if not all(math.isfinite(v) for v in (J, h, dt)) or dt <= 0:
        raise ValueError("J and h must be finite; dt must be positive")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if order not in (1, 2, 4):
        raise ValueError("order must be 1, 2, or 4")
    if not isinstance(max_bond, int) or isinstance(max_bond, bool) or max_bond <= 0:
        raise ValueError("max_bond must be positive")
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be finite and non-negative")
    if not isinstance(routing, str) or not routing:
        raise ValueError("routing must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if (not isinstance(sample_shots, int) or isinstance(sample_shots, bool)
            or sample_shots < 0):
        raise ValueError("sample_shots must be a non-negative integer")
    if not isinstance(save_local, bool):
        raise ValueError("save_local must be a boolean")


def dense_pauli_operator(n_qubits, paulis):
    """Build a little-endian dense Pauli product for the documented qubits."""
    identity = np.eye(2, dtype=complex)
    matrices = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    operator = np.array([[1.0]], dtype=complex)
    for qubit in reversed(range(n_qubits)):
        operator = np.kron(operator, matrices.get(paulis.get(qubit), identity))
    return operator


def dense_exact_observables(rows, cols, J, h, total_time):
    """Exact continuous-time reference for a small lattice such as 2x2."""
    n_qubits = rows * cols
    if n_qubits > 10:
        raise ValueError("dense reference is restricted to at most 10 qubits")
    pairs = lattice_edges(rows, cols)
    hamiltonian = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for first, second in pairs:
        hamiltonian -= J * dense_pauli_operator(
            n_qubits, {first: "Z", second: "Z"}
        )
    for qubit in range(n_qubits):
        hamiltonian -= h * dense_pauli_operator(n_qubits, {qubit: "X"})

    initial = np.zeros(2**n_qubits, dtype=complex)
    initial[0] = 1.0
    state = expm(-1j * hamiltonian * total_time) @ initial

    def expectation(operator):
        value = complex(state.conj() @ operator @ state)
        if abs(value.imag) > 1e-10:
            raise RuntimeError(f"dense expectation is not real: {value}")
        return float(value.real)

    mz = sum(
        expectation(dense_pauli_operator(n_qubits, {q: "Z"}))
        for q in range(n_qubits)
    ) / n_qubits
    mx = sum(
        expectation(dense_pauli_operator(n_qubits, {q: "X"}))
        for q in range(n_qubits)
    ) / n_qubits
    mean_zz = sum(
        expectation(dense_pauli_operator(n_qubits, {i: "Z", j: "Z"}))
        for i, j in pairs
    ) / len(pairs)
    values = {
        "magnetization_z": mz,
        "magnetization_x": mx,
        "mean_zz": mean_zz,
        "domain_wall_density": (1.0 - mean_zz) / 2.0,
        "energy_per_site": (
            -J * len(pairs) * mean_zz - h * n_qubits * mx
        ) / n_qubits,
    }
    validate_observables(values)
    return values


def max_observable_difference(first, second):
    keys = (
        "magnetization_z",
        "magnetization_x",
        "mean_zz",
        "domain_wall_density",
        "energy_per_site",
    )
    return max(abs(first[key] - second[key]) for key in keys)


def run_ising_circuit(rows=2, cols=2, J=1.0, h=3.0, dt=0.025,
                      steps=2, order=2, max_bond=32, cutoff=1e-12,
                      routing="auto", seed=7, sample_shots=0,
                      save_local=False):
    validate_inputs(
        rows, cols, J, h, dt, steps, order, max_bond, cutoff,
        routing, seed, sample_shots, save_local,
    )
    model = IsingModel(rows, cols, J, h)
    step_circuit = build_trotter_step(model, dt, order)
    execute_options = {
        "backend": "tensornet",
        "device": "cpu",
        "seed": seed,
        "backend_options": {
            "max_bond": max_bond,
            "cutoff": cutoff,
            "routing": routing,
        },
    }

    # Empty circuit produces the default |0>^N TensorNet state.
    result = Circuit(model.num_qubits).execute(**execute_options)
    records = []
    for step_index in range(steps + 1):
        values = observe(result, model)
        validate_observables(values)
        record = {
            "step": step_index,
            "time": step_index * dt,
            **values,
        }
        if save_local:
            record.update(observe_local_configuration(result, model))
        records.append(record)
        if step_index < steps:
            # Continue from the public backend state; do not inspect MPS internals.
            result = step_circuit.execute(
                initial_state=result.backend_state,
                **execute_options,
            )

    samples = result.sample(shots=sample_shots, seed=seed) if sample_shots else []
    return {
        "status": "ok",
        "circuit": step_circuit,
        "step_circuit_summary": step_circuit.analyze(show=False).get_summary(),
        "records": records,
        "final_observables": records[-1],
        "samples": samples,
    }


output = run_ising_circuit()
print(output["final_observables"])
exact = dense_exact_observables(2, 2, J=1.0, h=3.0, total_time=0.05)
print("exact-reference max error:", max_observable_difference(
    output["final_observables"], exact
))
```

The sign and angle conventions are mandatory. Because `RX(theta)=exp(-i theta X/2)`, use `theta=-2*h*duration` for `H_X=-hΣX`. Because the CNOT–RZ–CNOT block implements `exp(-i theta ZiZj/2)`, use `theta=-2*J*duration` for `H_ZZ=-JΣZZ`. Reversing either sign evolves under a different Hamiltonian.

The four edge groups are scheduling layers, not four separate Hamiltonians. Open boundaries mean no edge joins the final column to the first column or the final row to the first row. For order 4, the central coefficient `1-4p` is negative by design and must not be clipped or replaced.

The trajectory performs `steps+1` observations but only `steps` circuit executions. Each continuation must pass `result.backend_state` as `initial_state`; do not extract or mutate internal MPS tensors. For a small correctness test, run `2×2`, compare order 1 and order 2 at the same total time, then halve `dt` while doubling `steps`. The observable difference should decrease until TensorNet truncation error dominates.

`dense_exact_observables()` supplies the independent continuous-time `2×2` reference used by the example. It follows the documented little-endian qubit convention and is intentionally limited to at most 10 qubits because it materializes a `2**N × 2**N` Hamiltonian. The reported difference combines Trotter and TensorNet errors; use the three checks in `Hands-On Example` to isolate them. Preserve `validate_observables()` so finite results and Pauli-observable ranges are checked before comparison.

The manual interface uses the boolean `save_local` only to keep the reconstruction focused. The production API's `local_configuration_mode` additionally controls `none`, `final`, `all`, and `gif` output behavior; do not infer that those modes were removed from `IsingAlgorithm`.

## Debugging Tips

- A `1×1` input fails by design because the model requires at least one nearest-neighbor edge.
- Invalid Trotter orders must be exactly `1`, `2`, or `4`; no arbitrary order is accepted.
- A missing TensorNet backend or unsupported routing option is an execution-environment issue; verify the public UnitaryLab backend configuration.
- Large observable drift under smaller time steps indicates Trotter error; compare matched total times.
- Large differences under larger `max_bond` or smaller `cutoff` indicate TensorNet truncation error; perform a convergence check.
- Local configuration memory and GIF generation grow with `trotter_steps`; use `local_configuration_mode="none"` for an interface smoke test.
- If `samples` is empty, confirm that `sample_shots` is positive; zero intentionally disables sampling.
- Check `records[-1]` and `final_observables` rather than interpreting the generic status field as a physics-validity guarantee.

## Reference Implementation

- **Public API:** `new/ising/algorithm.py`, especially `IsingAlgorithm.__init__()` and `IsingAlgorithm.run()`, plus the public UnitaryLab `Circuit`, result, `expectation()`, `backend_state`, and `sample()` interfaces.
- **Authoritative implementation source:** `new/ising/algorithm.py` defines input validation, lattice mapping, Trotter construction, TensorNet continuation, observable extraction, and result assembly.
- **Reference/example material:** `README.md` and `parameters.json` provide documented defaults, examples, parameter metadata, output conventions, and explanatory interpretation. They are not runtime dependencies.
- **Theory references:** the Ising Hamiltonian and Trotter-Suzuki equations establish the mathematical model and approximation method; they do not imply support for initial states, boundaries, orders, or observables absent from the implementation.

Generated user code must import the public algorithm module and public UnitaryLab interfaces, not README files or reference scripts.
