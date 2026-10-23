"""Independent 2D transverse-field Ising circuit implementation.

The implementation is derived from the sibling ``SKILL.md`` and uses only
public UnitaryLab interfaces. It does not import ``unitarylab_algorithms.quantum_machine_learning.ising.algorithm`` or a
reference/example script. The module constructs open-boundary Trotter-Suzuki
circuits, continues a TensorNet state through time, evaluates physical
observables, and grades small systems against an exact dense evolution.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm
from unitarylab import Circuit


OBSERVABLE_KEYS = (
    "magnetization_z",
    "magnetization_x",
    "mean_zz",
    "domain_wall_density",
    "energy_per_site",
)


@dataclass(frozen=True)
class IsingConfig:
    """Physical and TensorNet settings for one evolution trajectory."""

    rows: int = 2
    cols: int = 2
    coupling: float = 1.0
    field: float = 3.0
    time_step: float = 0.025
    steps: int = 2
    order: int = 2
    max_bond: int = 32
    cutoff: float = 1e-12
    routing: str = "auto"
    seed: int = 7
    sample_shots: int = 0
    include_local: bool = False

    @property
    def num_qubits(self) -> int:
        return self.rows * self.cols

    @property
    def final_time(self) -> float:
        return self.time_step * self.steps

    def validate(self) -> None:
        if isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows <= 0:
            raise ValueError("rows must be a positive integer")
        if isinstance(self.cols, bool) or not isinstance(self.cols, int) or self.cols <= 0:
            raise ValueError("cols must be a positive integer")
        if self.num_qubits == 1:
            raise ValueError("the lattice must contain at least one edge")
        if not math.isfinite(self.coupling) or not math.isfinite(self.field):
            raise ValueError("coupling and field must be finite")
        if not math.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be positive and finite")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if self.order not in (1, 2, 4):
            raise ValueError("order must be 1, 2, or 4")
        if isinstance(self.max_bond, bool) or not isinstance(self.max_bond, int) or self.max_bond <= 0:
            raise ValueError("max_bond must be a positive integer")
        if not math.isfinite(self.cutoff) or self.cutoff < 0:
            raise ValueError("cutoff must be finite and non-negative")
        if not self.routing:
            raise ValueError("routing must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.sample_shots, bool)
            or not isinstance(self.sample_shots, int)
            or self.sample_shots < 0
        ):
            raise ValueError("sample_shots must be a non-negative integer")


@dataclass
class TrajectoryResult:
    """Circuit, records, optional samples, and exact-reference evidence."""

    circuit: Circuit
    records: list[dict[str, Any]]
    samples: Any
    exact_observables: dict[str, float] | None
    exact_error: float | None

    @property
    def final_observables(self) -> dict[str, Any]:
        return self.records[-1]

    def summary(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "time_points": len(self.records),
            "final_observables": self.final_observables,
            "exact_reference_error": self.exact_error,
            "step_circuit_summary": self.circuit.analyze(show=False).get_summary(),
        }


def site_id(row: int, col: int, cols: int) -> int:
    """Map a 2D site to a row-major qubit index."""
    return row * cols + col


def open_boundary_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Enumerate horizontal and vertical nearest-neighbor edges."""
    horizontal = [
        (site_id(row, col, cols), site_id(row, col + 1, cols))
        for row in range(rows)
        for col in range(cols - 1)
    ]
    vertical = [
        (site_id(row, col, cols), site_id(row + 1, col, cols))
        for row in range(rows - 1)
        for col in range(cols)
    ]
    return horizontal + vertical


def edge_layers(rows: int, cols: int) -> list[list[tuple[int, int]]]:
    """Split edges into four matchings with no shared qubits per layer."""
    layers: list[list[tuple[int, int]]] = [[], [], [], []]
    for row in range(rows):
        for col in range(cols - 1):
            first = site_id(row, col, cols)
            layers[col % 2].append((first, first + 1))
    for row in range(rows - 1):
        for col in range(cols):
            first = site_id(row, col, cols)
            layers[2 + row % 2].append((first, first + cols))
    return layers


def append_field_evolution(circuit: Circuit, config: IsingConfig, duration: float) -> None:
    """Append exp(-i H_X duration) for H_X=-h sum_i X_i."""
    angle = -2.0 * config.field * duration
    for qubit in range(config.num_qubits):
        circuit.rx(angle, qubit)


def append_interaction_evolution(
    circuit: Circuit, config: IsingConfig, duration: float
) -> None:
    """Append exp(-i H_ZZ duration) with CNOT-RZ-CNOT blocks."""
    angle = -2.0 * config.coupling * duration
    for layer in edge_layers(config.rows, config.cols):
        for first, second in layer:
            circuit.cx(first, second)
            circuit.rz(angle, second)
            circuit.cx(first, second)


def append_strang_block(circuit: Circuit, config: IsingConfig, duration: float) -> None:
    append_field_evolution(circuit, config, duration / 2.0)
    append_interaction_evolution(circuit, config, duration)
    append_field_evolution(circuit, config, duration / 2.0)


def build_trotter_circuit(config: IsingConfig) -> Circuit:
    """Build one first-, second-, or fourth-order Trotter step."""
    config.validate()
    circuit = Circuit(config.num_qubits)
    if config.order == 1:
        append_field_evolution(circuit, config, config.time_step)
        append_interaction_evolution(circuit, config, config.time_step)
    elif config.order == 2:
        append_strang_block(circuit, config, config.time_step)
    else:
        coefficient = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        weights = (
            coefficient,
            coefficient,
            1.0 - 4.0 * coefficient,
            coefficient,
            coefficient,
        )
        for weight in weights:
            append_strang_block(circuit, config, weight * config.time_step)
    return circuit


def real_expectation(result: Any, observable: Any, qubits=None) -> float:
    value = (
        result.expectation(observable, qubits)
        if qubits is not None
        else result.expectation(observable)
    )
    value = complex(value)
    if abs(value.imag) > 1e-10:
        raise RuntimeError(f"expectation is not real: {value}")
    return float(value.real)


def measure_global_observables(result: Any, config: IsingConfig) -> dict[str, float]:
    """Measure Mz, Mx, mean ZZ, domain-wall density, and energy/site."""
    pairs = open_boundary_edges(config.rows, config.cols)
    count = config.num_qubits
    mz = real_expectation(
        result, [(1.0 / count, "Z", [q]) for q in range(count)]
    )
    mx = real_expectation(
        result, [(1.0 / count, "X", [q]) for q in range(count)]
    )
    mean_zz = real_expectation(
        result, [(1.0 / len(pairs), "ZZ", [i, j]) for i, j in pairs]
    )
    values = {
        "magnetization_z": mz,
        "magnetization_x": mx,
        "mean_zz": mean_zz,
        "domain_wall_density": (1.0 - mean_zz) / 2.0,
        "energy_per_site": (
            -config.coupling * len(pairs) * mean_zz
            - config.field * count * mx
        ) / count,
    }
    validate_observables(values)
    return values


def measure_local_observables(result: Any, config: IsingConfig) -> dict[str, Any]:
    z_values = [
        real_expectation(result, "Z", [q]) for q in range(config.num_qubits)
    ]
    x_values = [
        real_expectation(result, "X", [q]) for q in range(config.num_qubits)
    ]
    return {
        "z_map": np.asarray(z_values).reshape(config.rows, config.cols).tolist(),
        "x_map": np.asarray(x_values).reshape(config.rows, config.cols).tolist(),
    }


def validate_observables(values: dict[str, float], tolerance: float = 1e-9) -> None:
    for key, value in values.items():
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite observable {key}: {value}")
    for key in ("magnetization_z", "magnetization_x", "mean_zz"):
        if not -1.0 - tolerance <= values[key] <= 1.0 + tolerance:
            raise RuntimeError(f"{key} is outside [-1, 1]: {values[key]}")
    domain_walls = values["domain_wall_density"]
    if not -tolerance <= domain_walls <= 1.0 + tolerance:
        raise RuntimeError(f"domain_wall_density is outside [0, 1]: {domain_walls}")


def dense_pauli_product(
    num_qubits: int, assignments: dict[int, str]
) -> np.ndarray:
    """Construct a little-endian dense Pauli product."""
    identity = np.eye(2, dtype=complex)
    matrices = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    operator = np.array([[1.0]], dtype=complex)
    for qubit in reversed(range(num_qubits)):
        label = assignments.get(qubit)
        operator = np.kron(operator, matrices[label] if label else identity)
    return operator


def exact_continuous_observables(config: IsingConfig) -> dict[str, float]:
    """Compute the exact continuous-time reference for at most 10 qubits."""
    if config.num_qubits > 10:
        raise ValueError("dense exact reference is limited to at most 10 qubits")
    pairs = open_boundary_edges(config.rows, config.cols)
    size = 2**config.num_qubits
    hamiltonian = np.zeros((size, size), dtype=complex)
    for first, second in pairs:
        hamiltonian -= config.coupling * dense_pauli_product(
            config.num_qubits, {first: "Z", second: "Z"}
        )
    for qubit in range(config.num_qubits):
        hamiltonian -= config.field * dense_pauli_product(
            config.num_qubits, {qubit: "X"}
        )

    initial = np.zeros(size, dtype=complex)
    initial[0] = 1.0
    state = expm(-1j * hamiltonian * config.final_time) @ initial

    def expectation(operator: np.ndarray) -> float:
        value = complex(state.conj() @ operator @ state)
        if abs(value.imag) > 1e-10:
            raise RuntimeError(f"dense expectation is not real: {value}")
        return float(value.real)

    mz = sum(
        expectation(dense_pauli_product(config.num_qubits, {q: "Z"}))
        for q in range(config.num_qubits)
    ) / config.num_qubits
    mx = sum(
        expectation(dense_pauli_product(config.num_qubits, {q: "X"}))
        for q in range(config.num_qubits)
    ) / config.num_qubits
    mean_zz = sum(
        expectation(
            dense_pauli_product(config.num_qubits, {i: "Z", j: "Z"})
        )
        for i, j in pairs
    ) / len(pairs)
    values = {
        "magnetization_z": mz,
        "magnetization_x": mx,
        "mean_zz": mean_zz,
        "domain_wall_density": (1.0 - mean_zz) / 2.0,
        "energy_per_site": (
            -config.coupling * len(pairs) * mean_zz
            - config.field * config.num_qubits * mx
        ) / config.num_qubits,
    }
    validate_observables(values)
    return values


def observable_error(first: dict[str, Any], second: dict[str, Any]) -> float:
    return max(abs(first[key] - second[key]) for key in OBSERVABLE_KEYS)


def execute_trajectory(config: IsingConfig) -> TrajectoryResult:
    """Execute the Trotter circuit using public TensorNet continuation."""
    config.validate()
    step_circuit = build_trotter_circuit(config)
    options = {
        "backend": "tensornet",
        "device": "cpu",
        "seed": config.seed,
        "backend_options": {
            "max_bond": config.max_bond,
            "cutoff": config.cutoff,
            "routing": config.routing,
        },
    }
    state = Circuit(config.num_qubits).execute(**options)
    records: list[dict[str, Any]] = []
    for step_index in range(config.steps + 1):
        record: dict[str, Any] = {
            "step": step_index,
            "time": step_index * config.time_step,
            **measure_global_observables(state, config),
        }
        if config.include_local:
            record.update(measure_local_observables(state, config))
        records.append(record)
        if step_index < config.steps:
            state = step_circuit.execute(
                initial_state=state.backend_state,
                **options,
            )

    samples = (
        state.sample(shots=config.sample_shots, seed=config.seed)
        if config.sample_shots
        else []
    )
    exact = exact_continuous_observables(config) if config.num_qubits <= 10 else None
    error = observable_error(records[-1], exact) if exact is not None else None
    return TrajectoryResult(step_circuit, records, samples, exact, error)


def parse_args() -> IsingConfig:
    parser = argparse.ArgumentParser(
        description="Run an independent 2D transverse-field Ising trajectory."
    )
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--field", type=float, default=3.0)
    parser.add_argument("--time-step", type=float, default=0.025)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--order", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=1e-12)
    parser.add_argument("--routing", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-shots", type=int, default=0)
    parser.add_argument("--include-local", action="store_true")
    return IsingConfig(**vars(parser.parse_args()))


def main() -> None:
    result = execute_trajectory(parse_args())
    for key, value in result.summary().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
