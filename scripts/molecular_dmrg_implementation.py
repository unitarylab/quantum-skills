"""Source-guided manual circuit reconstruction for Molecular DMRG.

This reference does not import MolecularDMRGAlgorithm, MolecularDMRGWorkflow,
or cvd_dmrg. Electronic structure and DMRG remain classical preprocessing;
the CVD preparation and QWC measurement circuits are built from basic gates.
"""

from __future__ import annotations

import itertools
import math
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from unitarylab import Circuit


PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
AXES = "XYZ"


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    operations: tuple[tuple[int, str], ...] = ()


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _state_and_qubits(state) -> tuple[np.ndarray, int]:
    vector = np.asarray(state, dtype=complex).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("state must be nonempty and finite")
    n_qubits = int(round(np.log2(vector.size)))
    if n_qubits < 1 or vector.size != 2**n_qubits:
        raise ValueError("state length must be an exact positive power of two")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("state must have nonzero norm")
    return vector / norm, n_qubits


def axis_rotation_matrix(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("axis vector must have shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(2, dtype=complex)
    nx, ny, nz = vector / angle
    sine = np.sin(angle / 2.0)
    return np.array(
        [
            [np.cos(angle / 2.0) - 1j * nz * sine, (-1j * nx - ny) * sine],
            [(ny - 1j * nx) * sine, np.cos(angle / 2.0) + 1j * nz * sine],
        ],
        dtype=complex,
    )


def entangler_matrix(parameters) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (9,):
        raise ValueError("an entangler requires exactly nine parameters")
    gate = np.kron(
        axis_rotation_matrix(parameters[3:6]),
        axis_rotation_matrix(parameters[:3]),
    )
    identity = np.eye(4, dtype=complex)
    for offset, axis in enumerate(AXES):
        angle = parameters[6 + offset]
        gate = (
            np.cos(angle) * identity
            - 1j * np.sin(angle) * np.kron(PAULI[axis], PAULI[axis])
        ) @ gate
    return gate


def apply_two_qubit(state, matrix, q0: int, q1: int, n_qubits: int):
    if not 0 <= q0 < q1 < n_qubits:
        raise ValueError("sites must satisfy 0 <= q0 < q1 < n_qubits")
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (4, 4):
        raise ValueError("two-qubit matrix must have shape (4, 4)")
    tensor = np.asarray(state, dtype=complex).reshape([2] * n_qubits)
    local_axes = [n_qubits - 1 - q1, n_qubits - 1 - q0]
    remaining = [axis for axis in range(n_qubits) if axis not in local_axes]
    permutation = local_axes + remaining
    moved = tensor.transpose(permutation).reshape(4, -1)
    moved = (matrix @ moved).reshape([2, 2] + [2] * (n_qubits - 2))
    return moved.transpose(np.argsort(permutation)).reshape(-1)


def renyi_entropy(state, n_qubits: int, bond: int, alpha: float = 1.0) -> float:
    if not 0 <= bond < n_qubits - 1:
        raise ValueError("bond must lie between adjacent qubits")
    if np.isnan(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive")
    matrix = np.asarray(state).reshape(
        2 ** (n_qubits - bond - 1), 2 ** (bond + 1)
    )
    probabilities = np.square(np.linalg.svd(matrix, compute_uv=False).real)
    probabilities = probabilities[probabilities > np.finfo(float).eps]
    probabilities /= probabilities.sum()
    if np.isinf(alpha):
        return float(-np.log(np.max(probabilities)))
    if np.isclose(alpha, 1.0):
        return float(-np.sum(probabilities * np.log(probabilities)))
    return float(np.log(np.sum(probabilities**alpha)) / (1.0 - alpha))


def append_axis_rotation(circuit: Circuit, qubit: int, vector) -> None:
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return
    nx, ny, nz = vector / angle
    sine = np.sin(angle / 2.0)
    a = np.cos(angle / 2.0) - 1j * nz * sine
    c = (ny - 1j * nx) * sine
    circuit.rz(float(-np.angle(c) - np.angle(a)), qubit)
    circuit.ry(float(2.0 * np.arctan2(abs(c), abs(a))), qubit)
    circuit.rz(float(np.angle(c) - np.angle(a)), qubit)


def append_pauli_pair_rotation(
    circuit: Circuit, q0: int, q1: int, axis: str, angle: float
) -> None:
    if axis not in AXES:
        raise ValueError("axis must be X, Y, or Z")
    if axis == "X":
        circuit.h(q0)
        circuit.h(q1)
    elif axis == "Y":
        circuit.rx(np.pi / 2.0, q0)
        circuit.rx(np.pi / 2.0, q1)
    circuit.cx(q0, q1)
    circuit.rz(float(2.0 * angle), q1)
    circuit.cx(q0, q1)
    if axis == "X":
        circuit.h(q0)
        circuit.h(q1)
    elif axis == "Y":
        circuit.rx(-np.pi / 2.0, q0)
        circuit.rx(-np.pi / 2.0, q1)


def append_inverse_entangler(
    circuit: Circuit, q0: int, q1: int, parameters
) -> None:
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (9,):
        raise ValueError("an entangler requires exactly nine parameters")
    for offset in reversed(range(3)):
        append_pauli_pair_rotation(
            circuit, q0, q1, AXES[offset], -parameters[6 + offset]
        )
    append_axis_rotation(circuit, q0, -parameters[:3])
    append_axis_rotation(circuit, q1, -parameters[3:6])


def product_factors(state, n_qubits: int) -> list[np.ndarray]:
    remainder = np.asarray(state, dtype=complex).reshape(-1)
    high_to_low = []
    for _ in range(n_qubits - 1):
        left, values, right = np.linalg.svd(
            remainder.reshape(2, -1), full_matrices=False
        )
        high_to_low.append(left[:, 0])
        remainder = values[0] * right[0]
    high_to_low.append(remainder / np.linalg.norm(remainder))
    return list(reversed(high_to_low))


def product_factor_rotation(factor) -> np.ndarray:
    factor = np.asarray(factor, dtype=complex)
    norm = float(np.linalg.norm(factor))
    if factor.shape != (2,) or norm <= 0.0:
        raise ValueError("factor must be a nonzero two-component vector")
    a, b = factor / norm
    theta = 2.0 * np.arctan2(abs(b), abs(a))
    if theta < 1e-12:
        return np.zeros(3)
    phase_a = np.angle(a) if abs(a) > 1e-15 else 0.0
    delta = np.angle(b) - phase_a
    return np.array([-theta * np.sin(delta), theta * np.cos(delta), 0.0])


def manual_cvd_circuit(
    dmrg_state,
    *,
    layers: int | None = None,
    renyi_indices: tuple[float, ...] | None = None,
    max_iterations: int = 100,
    restarts: int = 3,
    init_scale: float = 0.5,
    seed: int = 2077,
) -> tuple[Circuit, float, float]:
    target, n_qubits = _state_and_qubits(dmrg_state)
    layers = math.ceil(n_qubits / 2) + 1 if layers is None else layers
    layers = _positive_int("layers", layers)
    max_iterations = _positive_int("max_iterations", max_iterations)
    restarts = _positive_int("restarts", restarts)
    if not np.isfinite(init_scale) or init_scale < 0.0:
        raise ValueError("init_scale must be finite and nonnegative")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if renyi_indices is None:
        renyi_indices = (1.0,) * layers
    if len(renyi_indices) != layers or any(
        np.isnan(alpha) or alpha <= 0.0 for alpha in renyi_indices
    ):
        raise ValueError("renyi_indices must contain one positive value per layer")

    disentangled = target.copy()
    gate_rows: list[tuple[int, int, np.ndarray]] = []
    for layer_index, alpha in enumerate(renyi_indices):
        for start in (0, 1):
            group_input = disentangled.copy()
            optimized = []
            for q0 in range(start, n_qubits - 1, 2):
                q1 = q0 + 1

                def objective(parameters):
                    candidate = apply_two_qubit(
                        group_input,
                        entangler_matrix(parameters),
                        q0,
                        q1,
                        n_qubits,
                    )
                    return renyi_entropy(candidate, n_qubits, q0, alpha)

                best_parameters = np.zeros(9)
                best_value = objective(best_parameters)
                for restart in range(restarts):
                    initial = np.random.default_rng(
                        int(seed) + 1009 * layer_index + 37 * q0 + restart
                    ).normal(0.0, init_scale, 9)
                    trial = minimize(
                        objective,
                        initial,
                        method="L-BFGS-B",
                        options={"maxiter": max_iterations, "ftol": 1e-12},
                    )
                    if trial.fun < best_value:
                        best_parameters = np.asarray(trial.x)
                        best_value = float(trial.fun)
                optimized.append((q0, q1, best_parameters))
            for q0, q1, parameters in optimized:
                disentangled = apply_two_qubit(
                    disentangled,
                    entangler_matrix(parameters),
                    q0,
                    q1,
                    n_qubits,
                )
                gate_rows.append((q0, q1, parameters))

    circuit = Circuit(n_qubits, name="Manual DMRG reverse preparation")
    for qubit, factor in enumerate(product_factors(disentangled, n_qubits)):
        append_axis_rotation(circuit, qubit, product_factor_rotation(factor))
    for q0, q1, parameters in reversed(gate_rows):
        append_inverse_entangler(circuit, q0, q1, parameters)
    prepared = np.asarray(
        circuit.execute(backend="numpy", dtype=np.complex128).state,
        dtype=complex,
    )
    overlap = float(np.clip(abs(np.vdot(target, prepared)), 0.0, 1.0))
    return circuit, overlap, max(0.0, 1.0 - overlap**2)


def qwc_groups(terms: list[PauliTerm]) -> list[dict]:
    non_identity = [term for term in terms if term.operations]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(non_identity)))
    bases = [dict(term.operations) for term in non_identity]
    for left, right in itertools.combinations(range(len(non_identity)), 2):
        shared = bases[left].keys() & bases[right].keys()
        if any(bases[left][qubit] != bases[right][qubit] for qubit in shared):
            graph.add_edge(left, right)
    colors = nx.coloring.greedy_color(graph, strategy="saturation_largest_first")
    grouped: dict[int, dict] = {}
    for index, color in colors.items():
        group = grouped.setdefault(color, {"basis": {}, "terms": []})
        group["basis"].update(bases[index])
        group["terms"].append(non_identity[index])
    return list(grouped.values())


def qwc_measurement_circuit(preparation: Circuit, basis: dict[int, str]) -> Circuit:
    n_qubits = preparation.get_num_qubits()
    if any(not 0 <= qubit < n_qubits for qubit in basis):
        raise ValueError("measurement basis contains an invalid qubit")
    if any(axis not in AXES for axis in basis.values()):
        raise ValueError("measurement basis axes must be X, Y, or Z")
    circuit = preparation.copy()
    for qubit, axis in basis.items():
        if axis == "X":
            circuit.h(qubit)
        elif axis == "Y":
            circuit.rz(-np.pi / 2.0, qubit)
            circuit.h(qubit)
    qubits = list(range(n_qubits))
    circuit.measure(qubits, qubits)
    return circuit


def expectation_from_counts(counts: dict[str, int], term: PauliTerm) -> float:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    return float(
        sum(
            count
            * np.prod(
                [1 - 2 * int(bitstring[qubit]) for qubit, _ in term.operations]
            )
            for bitstring, count in counts.items()
        )
        / total
    )


def exact_energy(execution, terms: list[PauliTerm], n_qubits: int) -> float:
    value = 0.0
    for term in terms:
        if not term.operations:
            value += term.coefficient
            continue
        observable = "".join(axis for _, axis in term.operations)
        qubits = tuple(qubit for qubit, _ in term.operations)
        value += term.coefficient * float(
            execution.expectation(observable, qubits=qubits).real
        )
    return float(value)


def measure_hamiltonian(
    preparation: Circuit,
    terms: list[PauliTerm],
    *,
    shots: int = 0,
    seed: int | None = 2077,
    output_directory: str | Path | None = None,
) -> dict:
    if isinstance(shots, (bool, np.bool_)) or not isinstance(
        shots, (int, np.integer)
    ) or shots < 0:
        raise ValueError("shots must be a nonnegative integer")
    n_qubits = preparation.get_num_qubits()
    for term in terms:
        for qubit, axis in term.operations:
            if not 0 <= qubit < n_qubits or axis not in AXES:
                raise ValueError(f"invalid Pauli operation: {(qubit, axis)!r}")
    groups = qwc_groups(terms)
    execution = preparation.execute(
        backend="numpy", dtype=np.complex128, shots=1, seed=seed
    )
    exact = exact_energy(execution, terms, n_qubits)
    sampled = None if shots == 0 else sum(
        term.coefficient for term in terms if not term.operations
    )
    child_seeds = np.random.SeedSequence(seed).spawn(len(groups)) if shots else []
    qasm_files = []
    destination = Path(output_directory) if output_directory is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    for group_id, group in enumerate(groups, 1):
        measured = qwc_measurement_circuit(preparation, group["basis"])
        if shots:
            basis_execution = preparation.copy()
            for qubit, axis in group["basis"].items():
                if axis == "X":
                    basis_execution.h(qubit)
                elif axis == "Y":
                    basis_execution.rz(-np.pi / 2.0, qubit)
                    basis_execution.h(qubit)
            result = basis_execution.execute(
                backend="numpy", dtype=np.complex128, shots=1, seed=seed
            )
            sample_seed = int(
                child_seeds[group_id - 1].generate_state(1, dtype=np.uint64)[0]
            )
            counts = dict(
                Counter(
                    result.sample(
                        shots=int(shots),
                        qubits=tuple(range(n_qubits)),
                        seed=sample_seed,
                    )
                )
            )
            sampled += sum(
                term.coefficient * expectation_from_counts(counts, term)
                for term in group["terms"]
            )
        if destination is not None:
            label = "".join(
                group["basis"].get(qubit, "I") for qubit in range(n_qubits)
            )
            path = destination / f"measurement_{group_id:02d}_{label}.qasm"
            path.write_text(measured.to_qasm2(), encoding="utf-8")
            qasm_files.append(path)
    return {
        "exact_energy": exact,
        "sampled_energy": None if sampled is None else float(sampled),
        "measurement_groups": len(groups),
        "qasm_files": tuple(qasm_files),
    }


def load_h2_dmrg_state():
    """Run only the non-circuit source stages for the bundled H2 geometry."""
    from unitarylab_chemistry.dmrg import dmrg
    from unitarylab_chemistry.molecule import Molecule, electronic_structure

    resource = files(
        "unitarylab_algorithms.quantum_chemistry.molecular_dmrg"
    ).joinpath("molecules", "H2.xyz")
    with as_file(resource) as xyz_path:
        molecule = Molecule.from_xyz(
            xyz_path,
            basis="sto-3g",
            active_occupied=1,
            active_virtual=1,
        )
        electronic = electronic_structure(molecule)
    dmrg_result = dmrg(
        electronic, chi0=10, maxdim=20, max_sweeps=10, cutoff=1e-8
    )
    return electronic, dmrg_result


def synthetic_smoke_test() -> None:
    target = np.zeros(16, dtype=complex)
    target[3] = target[12] = 1.0 / np.sqrt(2.0)
    circuit, overlap, infidelity = manual_cvd_circuit(
        target, layers=1, max_iterations=20, restarts=1
    )
    assert circuit.get_num_qubits() == 4
    assert overlap > 1.0 - 1e-10
    assert math.isclose(infidelity, max(0.0, 1.0 - overlap**2))
    terms = [
        PauliTerm(-0.25),
        PauliTerm(0.5, ((0, "Z"),)),
        PauliTerm(0.5, ((1, "Z"),)),
        PauliTerm(0.125, ((0, "X"), (1, "X"))),
    ]
    measured = measure_hamiltonian(circuit, terms, shots=0)
    assert measured["sampled_energy"] is None
    assert measured["measurement_groups"] == len(qwc_groups(terms))
    print("overlap:", overlap)
    print("infidelity:", infidelity)
    print("exact energy:", measured["exact_energy"])


def h2_smoke_test() -> dict:
    """Connect the real classical H2 stages to the manual circuit path."""
    electronic, dmrg_result = load_h2_dmrg_state()
    circuit, overlap, infidelity = manual_cvd_circuit(
        dmrg_result.psi0,
        layers=1,
        max_iterations=10,
        restarts=1,
        seed=2077,
    )
    terms = [
        PauliTerm(term.coefficient, tuple(term.operations))
        for term in electronic.pauli_terms
    ]
    measurement = measure_hamiltonian(circuit, terms, shots=0, seed=2077)
    difference = abs(measurement["exact_energy"] - dmrg_result.dmrg_energy)
    assert electronic.n_qubits == circuit.get_num_qubits()
    assert measurement["sampled_energy"] is None
    assert math.isclose(infidelity, max(0.0, 1.0 - overlap**2))
    result = {
        "active_space": (
            electronic.active_electrons,
            len(electronic.active_orbitals),
            electronic.n_qubits,
        ),
        "dmrg_energy": dmrg_result.dmrg_energy,
        "circuit_energy": measurement["exact_energy"],
        "energy_difference": difference,
        "overlap": overlap,
        "infidelity": infidelity,
        "circuit": circuit,
    }
    print({key: value for key, value in result.items() if key != "circuit"})
    return result


if __name__ == "__main__":
    if "--h2" in sys.argv[1:]:
        h2_smoke_test()
    else:
        synthetic_smoke_test()
