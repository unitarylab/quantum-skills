"""Independent, source-aligned Glued Trees reference implementation.

This module reproduces graph construction, oscillator embedding, Pauli
decomposition and cropping, Suzuki steps, time parsing, execution, sampling,
and small-system dense verification without importing the packaged algorithm.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import random
from time import perf_counter
from typing import Iterable, Optional

import networkx as nx
import numpy as np
from scipy.sparse.linalg import expm_multiply
from unitarylab import Circuit, ClassicalRegister, Register
from unitarylab.library.pauli_operator import (
    pauli_string_decomposition,
    pauli_string_evolution,
)


@dataclass
class GluedTreesModel:
    qubits: int
    graph: nx.Graph
    adjacency: np.ndarray
    stiffness: np.ndarray
    factor: np.ndarray
    hamiltonian: np.ndarray

    @property
    def columns_per_tree(self) -> int:
        return self.qubits - 2

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def exit_state(self) -> int:
        return self.node_count - 1


@dataclass
class GluedTreesRun:
    model: GluedTreesModel
    full_terms: list[tuple[str, float]]
    terms: list[tuple[str, float]]
    rows: list[dict[str, float | int]]
    representative_circuit: Circuit
    peak: dict[str, float | int]
    backend: str
    shots: int
    repetitions: int
    order: int
    model_seconds: float
    decomposition_seconds: float
    total_seconds: float


def build_glued_trees(qubits: int = 6, seed: int = 0) -> GluedTreesModel:
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
    degrees = dict.fromkeys(right_leaves, 0)
    rng = random.Random(seed)
    for vertex in left_leaves:
        candidates = [other for other in right_leaves if degrees[other] == 0]
        if not candidates:
            candidates = [other for other in right_leaves if degrees[other] < 2]
        for other in rng.sample(candidates, 2):
            graph.add_edge(vertex, other)
            degrees[other] += 1
    adjacency = nx.to_numpy_array(graph, nodelist=range(nodes), dtype=float)
    stiffness = 3 * np.eye(nodes) - adjacency
    factor = np.pad(np.linalg.cholesky(stiffness), ((0, 0), (0, 4)))
    hamiltonian = np.zeros((2**qubits, 2**qubits))
    hamiltonian[:nodes, nodes:] = -factor
    hamiltonian[nodes:, :nodes] = -factor.T
    return GluedTreesModel(
        qubits, graph, adjacency, stiffness, factor, hamiltonian
    )


def decompose_hamiltonian(model: GluedTreesModel) -> list[tuple[str, float]]:
    terms = pauli_string_decomposition(
        model.hamiltonian,
        partition_commuting=False,
        real_symmetric_hint=True,
    )
    real_terms = [
        (word, float(np.real(coefficient))) for word, coefficient in terms
    ]
    return sorted(real_terms, key=lambda term: (-abs(term[1]), term[0]))


def crop_pauli_list(
    terms: Iterable[tuple[str, float]], size: Optional[int] = None
) -> list[tuple[str, float]]:
    terms = list(terms)
    if size is not None and size <= 0:
        raise ValueError("max_terms must be positive or None")
    if size is None or len(terms) <= size:
        return terms
    selected: list[tuple[str, float]] = []
    used: set[int] = set()
    quota, cursor = round(size * 0.6), 0
    while len(selected) < quota:
        previous_size = len(selected)
        for _ in range(4):
            for qubit in range(len(terms[0][0])):
                letter = "IXYZ"[cursor % 4]
                cursor += 1
                for index, (word, _) in enumerate(terms):
                    if index not in used and word[qubit] == letter:
                        selected.append(terms[index])
                        used.add(index)
                        break
                if len(selected) >= quota:
                    break
            if len(selected) >= quota:
                break
        if len(selected) == previous_size:
            break
    for index, term in enumerate(terms):
        if len(selected) >= size:
            break
        if index not in used:
            selected.append(term)
    return selected


def make_trotter_step(
    qubits: int,
    terms: Iterable[tuple[str, float]],
    dt: float,
    order: int = 1,
) -> Circuit:
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")
    step = Circuit(qubits, name=f"Suzuki order {order}, dt={dt:g}")
    if dt == 0:
        return step
    terms = list(terms)
    if any(len(word) != qubits for word, _ in terms):
        raise ValueError("Pauli word length must equal qubits")
    sequence = terms
    if order == 2 and terms:
        half = [(word, coefficient / 2) for word, coefficient in terms[:-1]]
        sequence = half + terms[-1:] + list(reversed(half))
    blocks: dict[tuple[str, float], Circuit] = {}
    for word, coefficient in sequence:
        key = (word, coefficient * dt)
        if key not in blocks:
            blocks[key] = pauli_string_evolution(*key)
        step.append(blocks[key], target=list(range(qubits)))
    return step


def make_glued_trees_circuit(
    qubits: int,
    terms: Iterable[tuple[str, float]],
    time: float,
    *,
    repetitions: int = 10,
    order: int = 1,
) -> Circuit:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    quantum = Register("state", qubits)
    classical = ClassicalRegister("result", qubits)
    circuit = Circuit(quantum, classical, name=f"Glued Trees t={time:g}")
    step = make_trotter_step(qubits, terms, time / repetitions, order)
    if time != 0:
        for _ in range(repetitions):
            circuit.append(step, target=quantum[:])
    return circuit


def parse_times(times, model: GluedTreesModel) -> np.ndarray:
    if times is None or (isinstance(times, str) and not times.strip()):
        values = 2 * model.columns_per_tree + np.arange(-12, 13, 2)
    elif isinstance(times, str):
        values = [float(value.strip()) for value in times.split(",") if value.strip()]
    elif np.isscalar(times):
        values = [float(times)]
    else:
        values = times
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("times must be a non-empty finite one-dimensional sequence")
    return array


def run_glued_trees(
    qubits: int = 6,
    seed: int = 0,
    shots: int = 8192,
    repetitions: int = 10,
    order: int = 1,
    max_terms: Optional[int] = None,
    backend: str = "numpy",
    times=None,
) -> GluedTreesRun:
    qubits, seed = int(qubits), int(seed)
    shots, repetitions, order = int(shots), int(repetitions), int(order)
    max_terms = None if max_terms in (None, "", "None") else int(max_terms)
    backend = str(backend)
    if shots <= 0 or repetitions <= 0 or seed < 0:
        raise ValueError("shots/repetitions must be positive; seed non-negative")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    started = perf_counter()
    model = build_glued_trees(qubits, seed)
    model_end = perf_counter()
    scan_times = parse_times(times, model)
    full_terms = decompose_hamiltonian(model)
    terms = crop_pauli_list(full_terms, max_terms)
    decomposition_end = perf_counter()
    exit_bits = format(model.exit_state, f"0{qubits}b")
    rows: list[dict[str, float | int]] = []
    representative: Circuit | None = None
    best_probability = -1.0
    for index, time_value in enumerate(scan_times):
        build_start = perf_counter()
        circuit = make_glued_trees_circuit(
            qubits,
            terms,
            float(time_value),
            repetitions=repetitions,
            order=order,
        )
        built = perf_counter()
        result = circuit.execute(backend=backend, device="cpu", seed=seed + index)
        executed = perf_counter()
        counts = Counter(result.sample(shots=shots, seed=seed + index))
        sampled = perf_counter()
        exit_count = counts.get(exit_bits, 0)
        probability = exit_count / shots
        row = {
            "time": float(time_value),
            "sampled_probability": probability,
            "exit_count": exit_count,
            "build_seconds": built - build_start,
            "execute_seconds": executed - built,
            "sample_seconds": sampled - executed,
        }
        rows.append(row)
        if probability > best_probability:
            best_probability = probability
            representative = circuit
    assert representative is not None
    peak = max(rows, key=lambda row: row["sampled_probability"])
    return GluedTreesRun(
        model=model,
        full_terms=full_terms,
        terms=terms,
        rows=rows,
        representative_circuit=representative,
        peak=peak,
        backend=backend,
        shots=shots,
        repetitions=repetitions,
        order=order,
        model_seconds=model_end - started,
        decomposition_seconds=decomposition_end - model_end,
        total_seconds=perf_counter() - started,
    )


def dense_reference_probability(model: GluedTreesModel, time: float) -> float:
    initial = np.zeros(2**model.qubits, dtype=complex)
    initial[0] = 1.0
    evolved = expm_multiply(-1j * model.hamiltonian * time, initial)
    return float(abs(evolved[model.exit_state]) ** 2)


def validate_model(model: GluedTreesModel) -> None:
    expected_nodes = 2 ** (model.qubits - 1) - 2
    assert model.node_count == expected_nodes
    assert model.hamiltonian.shape == (2**model.qubits, 2**model.qubits)
    assert np.allclose(model.adjacency, model.adjacency.T)
    assert np.allclose(model.hamiltonian, model.hamiltonian.T.conj())
    assert np.all(np.linalg.eigvalsh(model.stiffness) > 0)
    assert model.exit_state == expected_nodes - 1


def build_diagnostic_report() -> dict[str, object]:
    """Validate the smallest model and return observable structural evidence."""
    model = build_glued_trees(4, 0)
    validate_model(model)
    explicit_times = parse_times("0,2,4", model)
    default_times = parse_times(None, model)
    assert np.array_equal(explicit_times, [0.0, 2.0, 4.0])
    assert np.array_equal(parse_times(2, model), [2.0])
    assert default_times.shape == (13,)
    dense_t0 = dense_reference_probability(model, 0.0)
    assert dense_t0 == 0.0
    terms = decompose_hamiltonian(model)
    cropped = crop_pauli_list(terms, min(16, len(terms)))
    for order in (1, 2):
        assert make_trotter_step(4, [("IIII", 1.0)], 0.1, order) is not None
    for invalid in (0, 3):
        try:
            make_trotter_step(4, [("IIII", 1.0)], 0.1, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Suzuki order was accepted")
    return {
        "qubits": model.qubits,
        "seed": 0,
        "columns_per_tree": model.columns_per_tree,
        "nodes": model.node_count,
        "edges": model.graph.number_of_edges(),
        "entrance_state": format(0, f"0{model.qubits}b"),
        "exit_state": format(model.exit_state, f"0{model.qubits}b"),
        "adjacency_shape": model.adjacency.shape,
        "factor_shape": model.factor.shape,
        "hamiltonian_shape": model.hamiltonian.shape,
        "adjacency_symmetry_residual": float(
            np.linalg.norm(model.adjacency - model.adjacency.T)
        ),
        "hamiltonian_hermitian_residual": float(
            np.linalg.norm(model.hamiltonian - model.hamiltonian.T.conj())
        ),
        "minimum_stiffness_eigenvalue": float(
            np.min(np.linalg.eigvalsh(model.stiffness))
        ),
        "full_pauli_terms": len(terms),
        "cropped_pauli_terms_demo": len(cropped),
        "largest_pauli_terms": terms[:5],
        "order_1_blocks_per_step": len(terms),
        "order_2_blocks_per_step": max(0, 2 * len(terms) - 1),
        "explicit_times": explicit_times.tolist(),
        "default_times": default_times.tolist(),
        "dense_exit_probability_t0": dense_t0,
    }


def print_diagnostic_report(report: dict[str, object]) -> None:
    """Print model, decomposition, circuit, and parser evidence."""
    print("Glued Trees structural report")
    print("  configuration:")
    print(f"    qubits={report['qubits']}, seed={report['seed']}")
    print(f"    columns_per_tree={report['columns_per_tree']}")
    print("  graph:")
    print(f"    nodes={report['nodes']}, edges={report['edges']}")
    print(
        f"    entrance={report['entrance_state']}, "
        f"exit={report['exit_state']}"
    )
    print("  matrices:")
    print(
        f"    adjacency={report['adjacency_shape']}, "
        f"factor={report['factor_shape']}, "
        f"hamiltonian={report['hamiltonian_shape']}"
    )
    print(
        "    adjacency_symmetry_residual="
        f"{report['adjacency_symmetry_residual']:.3e}"
    )
    print(
        "    hamiltonian_hermitian_residual="
        f"{report['hamiltonian_hermitian_residual']:.3e}"
    )
    print(
        "    minimum_stiffness_eigenvalue="
        f"{report['minimum_stiffness_eigenvalue']:.12g}"
    )
    print("  Pauli decomposition and Suzuki structure:")
    print(
        f"    full_terms={report['full_pauli_terms']}, "
        f"cropped_demo_terms={report['cropped_pauli_terms_demo']}"
    )
    print(f"    largest_terms={report['largest_pauli_terms']}")
    print(
        f"    order_1_blocks={report['order_1_blocks_per_step']}, "
        f"order_2_blocks={report['order_2_blocks_per_step']}"
    )
    print("  time parsing and reference:")
    print(f"    explicit_times={report['explicit_times']}")
    print(f"    default_times={report['default_times']}")
    print(
        "    dense_exit_probability(t=0)="
        f"{report['dense_exit_probability_t0']:.12g}"
    )


def print_execution_report(result: GluedTreesRun) -> None:
    """Print every sampled row and its exact dense reference comparison."""
    model = result.model
    exit_bits = format(model.exit_state, f"0{model.qubits}b")
    print("Glued Trees execution report")
    print(
        f"  backend={result.backend}, shots={result.shots}, "
        f"repetitions={result.repetitions}, order={result.order}"
    )
    print(
        f"  nodes={model.node_count}, exit_state={exit_bits}, "
        f"Pauli_terms={len(result.terms)}/{len(result.full_terms)}"
    )
    print("  time scan:")
    for row in result.rows:
        time_value = float(row["time"])
        sampled = float(row["sampled_probability"])
        exact = dense_reference_probability(model, time_value)
        print(
            f"    t={time_value:g}: count={row['exit_count']}/{result.shots}, "
            f"sampled={sampled:.9f}, exact={exact:.9f}, "
            f"abs_difference={abs(sampled - exact):.9f}"
        )
        print(
            f"      build={row['build_seconds']:.6f}s, "
            f"execute={row['execute_seconds']:.6f}s, "
            f"sample={row['sample_seconds']:.6f}s"
        )
    print(
        f"  sampled_peak: t={result.peak['time']:g}, "
        f"probability={result.peak['sampled_probability']:.9f}, "
        f"count={result.peak['exit_count']}"
    )
    print(
        f"  timings: model={result.model_seconds:.6f}s, "
        f"decomposition={result.decomposition_seconds:.6f}s, "
        f"total={result.total_seconds:.6f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--order", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    diagnostic = build_diagnostic_report()
    print_diagnostic_report(diagnostic)
    if args.execute:
        result = run_glued_trees(
            qubits=4,
            shots=256,
            repetitions=4,
            order=args.order,
            times="0,2,4",
        )
        print_execution_report(result)


if __name__ == "__main__":
    main()
