"""Gate-level Quantum Backtracking: reversible oracles, W=R_B R_A and QPE."""

# The self-contained research implementation intentionally uses delayed imports,
# large circuit-building routines, and parameter-heavy reversible-oracle APIs.
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-instance-attributes,too-many-statements,too-many-lines
# pylint: disable=import-outside-toplevel,wrong-import-position,multiple-imports
# pylint: disable=redefined-outer-name,invalid-name,duplicate-code,line-too-long

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import argparse, json, math, os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np

BitPath = tuple[int, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_bits(bits: Sequence[int], *, label: str) -> BitPath:
    result = tuple(map(int, bits))
    _require(
        all(bit in (0, 1) for bit in result), f"{label} must contain only 0/1 values"
    )
    return result


def _toggle_match(
    qc: Any, controls: Sequence[int], target: int, bits: Sequence[int]
) -> None:
    controls, bits = tuple(controls), tuple(bits)
    _require(len(controls) == len(bits), "control and bit-pattern lengths differ")
    if not controls:
        qc.x(target)
    elif len(controls) == 1:
        qc.cx(controls[0], target, control_state=bits[0])
    else:
        qc.mcx(list(controls), target, control_state=list(bits))


def _phase_when(
    qc: Any, controls: Sequence[int], bits: Sequence[int], target_one: int
) -> None:
    controls = tuple(controls)
    if not controls:
        qc.z(target_one)
    elif len(controls) == 1:
        qc.cz(controls[0], target_one, control_state=tuple(bits)[0])
    else:
        qc.mcz(list(controls), target_one, control_state=list(bits))


def _toggle_or(qc: Any, flags: Sequence[int], target: int) -> None:
    if len(flags) == 1:
        qc.cx(flags[0], target)
    elif flags:
        qc.x(target)
        qc.mcx(list(flags), target, control_state=[0] * len(flags))


FlagSpec = tuple[Sequence[int], int, Sequence[int]]


def _flagged_target(
    qc: Any,
    specs: Sequence[FlagSpec],
    target: int,
    *,
    all_zero: bool = False,
    extra_controls: Sequence[int] = (),
    extra_bits: Sequence[int] = (),
) -> None:
    for controls, work, bits in specs:
        _toggle_match(qc, controls, work, bits)
    flags = tuple(dict.fromkeys(work for _, work, _ in specs))
    if all_zero:
        _toggle_match(
            qc, [*extra_controls, *flags], target, [*extra_bits, *([0] * len(flags))]
        )
    else:
        _toggle_or(qc, flags, target)
    for controls, work, bits in reversed(tuple(specs)):
        _toggle_match(qc, controls, work, bits)


class ReversibleBinaryBacktrackingProblem(ABC):
    name: str
    max_depth: int
    reject_work_count: int

    @abstractmethod
    def append_accept_at_depth(
        self,
        qc: Any,
        path: Sequence[int],
        depth: int,
        target: int,
        clause_work: Sequence[int],
    ) -> None: ...

    @abstractmethod
    def append_reject_at_depth(
        self,
        qc: Any,
        path: Sequence[int],
        depth: int,
        target: int,
        clause_work: Sequence[int],
    ) -> None: ...

    @abstractmethod
    def append_accept_oracle(
        self,
        qc: Any,
        path: Sequence[int],
        depth_one_hot: Sequence[int],
        target: int,
        clause_work: Sequence[int],
    ) -> None: ...

    @abstractmethod
    def append_reject_oracle(
        self,
        qc: Any,
        path: Sequence[int],
        depth_one_hot: Sequence[int],
        target: int,
        clause_work: Sequence[int],
    ) -> None: ...

    @abstractmethod
    def classical_accept(self, path: BitPath) -> bool: ...

    @abstractmethod
    def classical_reject(self, path: BitPath) -> bool: ...


@dataclass(frozen=True)
class PrefixConstraintProblem(ReversibleBinaryBacktrackingProblem):
    name: str
    max_depth: int
    accepted_paths: tuple[BitPath, ...]
    rejected_prefixes: tuple[BitPath, ...] = ()

    def __post_init__(self) -> None:
        _require(self.max_depth >= 1, "max_depth must be positive")
        accepted = tuple(
            _validate_bits(path, label="accepted path") for path in self.accepted_paths
        )
        rejected = tuple(
            _validate_bits(path, label="rejected prefix")
            for path in self.rejected_prefixes
        )
        _require(
            all(len(path) <= self.max_depth for path in (*accepted, *rejected)),
            "a path is deeper than max_depth",
        )
        _require(
            len(set(accepted)) == len(accepted) and len(set(rejected)) == len(rejected),
            "duplicate accepted/rejected paths are not allowed",
        )
        for solution in accepted:
            _require(
                not any(solution[: len(prefix)] == prefix for prefix in rejected),
                f"accepted path {solution} lies below a rejected prefix",
            )
        object.__setattr__(self, "accepted_paths", accepted)
        object.__setattr__(self, "rejected_prefixes", rejected)

    @property
    def reject_work_count(self) -> int:
        return len(self.rejected_prefixes)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrefixConstraintProblem:
        def parse(value: Any) -> BitPath:
            return _validate_bits(
                (map(int, value) if isinstance(value, str) else value), label="path"
            )

        return cls(
            str(data.get("name", "binary-prefix-problem")),
            int(data["max_depth"]),
            tuple(parse(value) for value in data.get("accepted_paths", [])),
            tuple(parse(value) for value in data.get("rejected_prefixes", [])),
        )

    def classical_accept(self, path: BitPath) -> bool:
        return tuple(path) in self.accepted_paths

    def classical_reject(self, path: BitPath) -> bool:
        return any(
            len(path) >= len(prefix) and tuple(path[: len(prefix)]) == prefix
            for prefix in self.rejected_prefixes
        )

    def append_accept_at_depth(
        self,
        qc: Any,
        path: Sequence[int],
        depth: int,
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        for solution in self.accepted_paths:
            if len(solution) == depth:
                _toggle_match(qc, path[:depth], target, solution)

    def _reject_specs(
        self,
        path: Sequence[int],
        prefixes: Sequence[BitPath],
        clause_work: Sequence[int],
        depth_one_hot: Sequence[int] | None = None,
    ) -> list[FlagSpec]:
        _require(
            len(clause_work) >= len(prefixes), "insufficient reject-clause work qubits"
        )
        specs: list[FlagSpec] = []
        for prefix, work in zip(prefixes, clause_work):
            if depth_one_hot is None:
                specs.append((path[: len(prefix)], work, prefix))
            else:
                specs.extend(
                    ([depth_one_hot[k], *path[: len(prefix)]], work, [1, *prefix])
                    for k in range(len(prefix), self.max_depth + 1)
                )
        return specs

    def append_reject_at_depth(
        self,
        qc: Any,
        path: Sequence[int],
        depth: int,
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        prefixes = tuple(
            prefix for prefix in self.rejected_prefixes if len(prefix) <= depth
        )
        _flagged_target(qc, self._reject_specs(path, prefixes, clause_work), target)

    def append_accept_oracle(
        self,
        qc: Any,
        path: Sequence[int],
        depth_one_hot: Sequence[int],
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        for solution in self.accepted_paths:
            _toggle_match(
                qc,
                [depth_one_hot[len(solution)], *path[: len(solution)]],
                target,
                [1, *solution],
            )

    def append_reject_oracle(
        self,
        qc: Any,
        path: Sequence[int],
        depth_one_hot: Sequence[int],
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        specs = self._reject_specs(
            path, self.rejected_prefixes, clause_work, depth_one_hot
        )
        _flagged_target(qc, specs, target)


@dataclass(frozen=True)
class BinaryCNFProblem(ReversibleBinaryBacktrackingProblem):
    name: str
    variables: tuple[str, ...]
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _require(bool(self.variables), "variables must not be empty")
        _require(
            len(set(self.variables)) == len(self.variables),
            "variable names must be unique",
        )
        clauses = tuple(tuple(map(int, clause)) for clause in self.clauses)
        for clause in clauses:
            _require(bool(clause), "empty CNF clauses are not supported")
            _require(
                all(
                    literal and abs(literal) <= len(self.variables)
                    for literal in clause
                ),
                f"invalid literal in clause {clause}",
            )
            _require(
                len({abs(literal) for literal in clause}) == len(clause),
                f"a variable occurs twice in clause {clause}",
            )
        object.__setattr__(self, "clauses", clauses)

    @property
    def max_depth(self) -> int:
        return len(self.variables)

    @property
    def reject_work_count(self) -> int:
        return len(self.clauses)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinaryCNFProblem:
        return cls(
            str(data.get("name", "binary-cnf-problem")),
            tuple(map(str, data["variables"])),
            tuple(tuple(map(int, clause)) for clause in data["clauses"]),
        )

    @staticmethod
    def _false_assignment(
        clause: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return tuple(abs(literal) - 1 for literal in clause), tuple(
            0 if literal > 0 else 1 for literal in clause
        )

    def _false_specs(
        self, path: Sequence[int], depth: int, clause_work: Sequence[int]
    ) -> list[FlagSpec]:
        _require(
            len(clause_work) >= len(self.clauses), "insufficient CNF clause work qubits"
        )
        specs: list[FlagSpec] = []
        for clause, work in zip(self.clauses, clause_work):
            indices, bits = self._false_assignment(clause)
            if max(indices) < depth:
                specs.append(([path[index] for index in indices], work, bits))
        return specs

    def append_accept_at_depth(
        self,
        qc: Any,
        path: Sequence[int],
        depth: int,
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        if depth == self.max_depth:
            _flagged_target(
                qc, self._false_specs(path, depth, clause_work), target, all_zero=True
            )

    def append_reject_at_depth(
        self,
        qc: Any,
        path: Sequence[int],
        depth: int,
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        _flagged_target(qc, self._false_specs(path, depth, clause_work), target)

    def append_accept_oracle(
        self,
        qc: Any,
        path: Sequence[int],
        depth_one_hot: Sequence[int],
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        specs = self._false_specs(path, self.max_depth, clause_work)
        _flagged_target(
            qc,
            specs,
            target,
            all_zero=True,
            extra_controls=[depth_one_hot[self.max_depth]],
            extra_bits=[1],
        )

    def append_reject_oracle(
        self,
        qc: Any,
        path: Sequence[int],
        depth_one_hot: Sequence[int],
        target: int,
        clause_work: Sequence[int],
    ) -> None:
        _require(
            len(clause_work) >= len(self.clauses), "insufficient CNF clause work qubits"
        )
        specs: list[FlagSpec] = []
        for clause, work in zip(self.clauses, clause_work):
            indices, bits = self._false_assignment(clause)
            specs.extend(
                (
                    [depth_one_hot[depth], *(path[index] for index in indices)],
                    work,
                    [1, *bits],
                )
                for depth in range(max(indices) + 1, self.max_depth + 1)
            )
        _flagged_target(qc, specs, target)

    def classical_accept(self, path: BitPath) -> bool:
        return len(path) == self.max_depth and not self.classical_reject(path)

    def classical_reject(self, path: BitPath) -> bool:
        for clause in self.clauses:
            indices, bits = self._false_assignment(clause)
            if all(index < len(path) for index in indices) and all(
                path[index] == bit for index, bit in zip(indices, bits)
            ):
                return True
        return False


def problem_from_dict(data: dict[str, Any]) -> ReversibleBinaryBacktrackingProblem:
    problem_type = str(data.get("type", "binary_cnf")).lower()
    if problem_type == "binary_cnf":
        return BinaryCNFProblem.from_dict(data)
    if problem_type == "prefix_constraints":
        return PrefixConstraintProblem.from_dict(data)
    raise ValueError(f"unsupported problem type: {problem_type}")


@dataclass(frozen=True)
class NodeLayout:
    path: tuple[int, ...]
    depth: tuple[int, ...]
    accept_work: int
    reject_work: int
    reject_clause_work: tuple[int, ...]
    num_qubits: int

    @classmethod
    def for_problem(cls, problem: ReversibleBinaryBacktrackingProblem) -> NodeLayout:
        path = tuple(range(problem.max_depth))
        depth = tuple(range(problem.max_depth, 2 * problem.max_depth + 1))
        accept = 2 * problem.max_depth + 1
        start = accept + 2
        work = tuple(range(start, start + problem.reject_work_count))
        return cls(path, depth, accept, accept + 1, work, start + len(work))


@dataclass
class DetectionEvidence:
    prefix: list[int]
    method: str
    exists: bool
    zero_phase_probability: float | None = None
    zero_phase_count: int | None = None
    shots: int | None = None
    threshold: float = 3.0 / 8.0
    phase_bits: int | None = None
    phase_counts: dict[str, int] | None = None
    circuit_summary: dict[str, Any] | None = None


@dataclass
class SolveResult:
    problem: str
    exists: bool
    solution_path: list[int] | None
    solution_assignment: dict[str, int] | None
    evidence: list[DetectionEvidence]
    resource_report: dict[str, Any]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "exists": self.exists,
            "solution_path": self.solution_path,
            "solution_assignment": self.solution_assignment,
            "evidence": [asdict(item) for item in self.evidence],
            "resource_report": self.resource_report,
            "artifacts": self.artifacts,
        }


def _givens_parent_child(
    qc: Any, parent_depth: int, child_depth: int, path_bit: int, angle: float
) -> None:
    qc.cx(parent_depth, child_depth)
    qc.mcry(angle, [child_depth, path_bit], parent_depth, control_state=[1, 0])
    qc.cx(parent_depth, child_depth)


def _split_children(
    qc: Any, parent_depth: int, child_depth: int, path_bit: int, angle: float
) -> None:
    qc.mcry(angle, [child_depth, parent_depth], path_bit, control_state=[1, 0])


def append_local_star_reflection(
    qc: Any,
    parent_depth: int,
    child_depth: int,
    path_bit: int,
    accept_work: int,
    reject_work: int,
    *,
    root_weight: int | None = None,
) -> None:
    amplitudes = (
        np.ones(3)
        if root_weight is None
        else np.array([1.0, math.sqrt(root_weight), math.sqrt(root_weight)])
    )
    amplitudes /= np.linalg.norm(amplitudes)
    child_norm = math.hypot(float(amplitudes[1]), float(amplitudes[2]))
    givens = -2.0 * math.asin(child_norm)
    split = 2.0 * math.atan2(float(amplitudes[2]), float(amplitudes[1]))
    _split_children(qc, parent_depth, child_depth, path_bit, -split)
    _givens_parent_child(qc, parent_depth, child_depth, path_bit, -givens)
    _phase_when(
        qc,
        [child_depth, path_bit, accept_work, reject_work],
        [0, 0, 0, 0],
        parent_depth,
    )
    _givens_parent_child(qc, parent_depth, child_depth, path_bit, givens)
    _split_children(qc, parent_depth, child_depth, path_bit, split)


def append_parity_reflection(
    qc: Any,
    problem: ReversibleBinaryBacktrackingProblem,
    layout: NodeLayout,
    *,
    root_depth: int,
    parity: int,
) -> None:
    _require(0 <= root_depth <= problem.max_depth, "invalid subtree root depth")
    _require(parity in (0, 1), "parity must be 0 or 1")
    for depth in range(root_depth + parity, problem.max_depth + 1, 2):
        problem.append_accept_at_depth(
            qc, layout.path, depth, layout.accept_work, layout.reject_clause_work
        )
        problem.append_reject_at_depth(
            qc, layout.path, depth, layout.reject_work, layout.reject_clause_work
        )
        if depth < problem.max_depth:
            weight = problem.max_depth - root_depth if depth == root_depth else None
            append_local_star_reflection(
                qc,
                layout.depth[depth],
                layout.depth[depth + 1],
                layout.path[depth],
                layout.accept_work,
                layout.reject_work,
                root_weight=weight,
            )
            _phase_when(
                qc,
                [layout.accept_work, layout.reject_work],
                [0, 1],
                layout.depth[depth],
            )
        else:
            _phase_when(qc, [layout.accept_work], [0], layout.depth[depth])
        problem.append_reject_at_depth(
            qc, layout.path, depth, layout.reject_work, layout.reject_clause_work
        )
        problem.append_accept_at_depth(
            qc, layout.path, depth, layout.accept_work, layout.reject_clause_work
        )


def build_walk_circuit(
    problem: ReversibleBinaryBacktrackingProblem, *, root_depth: int = 0
) -> tuple[Any, Any, Any]:
    from unitarylab import Circuit

    layout = NodeLayout.for_problem(problem)
    circuits = [
        Circuit(layout.num_qubits, name=f"R_{name}[root_depth={root_depth}]")
        for name in ("A", "B")
    ]
    for parity, circuit in enumerate(circuits):
        append_parity_reflection(
            circuit, problem, layout, root_depth=root_depth, parity=parity
        )
    walk = Circuit(layout.num_qubits, name=f"W=R_B R_A[root_depth={root_depth}]")
    for circuit in circuits:
        walk.append(circuit, target=list(range(layout.num_qubits)))
    return circuits[0], circuits[1], walk


def node_basis_state(layout: NodeLayout, path: BitPath) -> np.ndarray:
    _require(len(path) < len(layout.depth), "path is deeper than the node register")
    state = np.zeros(1 << layout.num_qubits, dtype=complex)
    state[node_basis_index(layout, path)] = 1.0
    return state


def node_basis_index(layout: NodeLayout, path: BitPath) -> int:
    ones = [
        layout.depth[len(path)],
        *(layout.path[index] for index, bit in enumerate(path) if bit),
    ]
    return sum(1 << qubit for qubit in ones)


def build_node_prepare_circuit(layout: NodeLayout, path: BitPath) -> Any:
    from unitarylab import Circuit

    prepare = Circuit(layout.num_qubits, name=f"prepare node {path}")
    prepare.x(layout.depth[len(path)])
    for index, bit in enumerate(path):
        if bit:
            prepare.x(layout.path[index])
    return prepare


def circuit_summary(circuit: Any) -> dict[str, Any]:
    summary = circuit.analyze(show=False).get_summary()
    keys = (
        "num_qubits",
        "size",
        "depth",
        "single_qubit_gates",
        "two_qubit_gates",
        "multi_qubit_gates",
    )
    return {
        **{key: int(summary[key]) for key in keys},
        "gate_counts": {
            str(key): int(value) for key, value in summary["gate_counts"].items()
        },
    }


def build_child_oracle_circuit(problem: ReversibleBinaryBacktrackingProblem) -> Any:
    from unitarylab import Circuit

    layout = NodeLayout.for_problem(problem)
    branch = layout.num_qubits
    next_depth = tuple(range(branch + 1, branch + problem.max_depth + 2))
    valid, expandable = next_depth[-1] + 1, next_depth[-1] + 2
    qc = Circuit(expandable + 1, name="O_child")
    problem.append_accept_oracle(
        qc, layout.path, layout.depth, layout.accept_work, layout.reject_clause_work
    )
    problem.append_reject_oracle(
        qc, layout.path, layout.depth, layout.reject_work, layout.reject_clause_work
    )
    for depth in range(problem.max_depth):
        _toggle_match(
            qc,
            [layout.depth[depth], layout.accept_work, layout.reject_work],
            expandable,
            [1, 0, 0],
        )
    qc.cx(expandable, valid)
    for depth in range(problem.max_depth):
        qc.mcx([expandable, layout.depth[depth], branch], layout.path[depth])
        qc.mcx([expandable, layout.depth[depth]], next_depth[depth + 1])
    for depth in reversed(range(problem.max_depth)):
        _toggle_match(
            qc,
            [layout.depth[depth], layout.accept_work, layout.reject_work],
            expandable,
            [1, 0, 0],
        )
    problem.append_reject_oracle(
        qc, layout.path, layout.depth, layout.reject_work, layout.reject_clause_work
    )
    problem.append_accept_oracle(
        qc, layout.path, layout.depth, layout.accept_work, layout.reject_clause_work
    )
    return qc


class QuantumBacktrackingSolver:
    def __init__(
        self,
        problem: ReversibleBinaryBacktrackingProblem,
        *,
        phase_bits: int = 3,
        shots: int = 128,
        seed: int = 7,
        threshold: float = 3.0 / 8.0,
        backend: str = "numpy",
        device: str = "cpu",
        dtype=np.complex128,
    ) -> None:
        _require(phase_bits >= 1, "phase_bits must be positive")
        _require(shots >= 1, "shots must be positive")
        _require(
            0.0 < threshold < 1.0, "threshold must be strictly between zero and one"
        )
        self.problem, self.layout = problem, NodeLayout.for_problem(problem)
        self.phase_bits, self.shots, self.threshold = phase_bits, shots, threshold
        self.rng = np.random.default_rng(seed)
        self.backend, self.device, self.dtype = backend, device, dtype
        self.evidence: list[DetectionEvidence] = []
        self._root_circuits: tuple[Any, Any, Any, Any, Any] | None = None

    def _query_predicates(self, path: BitPath) -> tuple[bool, bool]:
        from unitarylab import Circuit

        qc = Circuit(self.layout.num_qubits, name=f"predicate query {path}")
        for method, target in (
            (self.problem.append_accept_oracle, self.layout.accept_work),
            (self.problem.append_reject_oracle, self.layout.reject_work),
        ):
            method(
                qc,
                self.layout.path,
                self.layout.depth,
                target,
                self.layout.reject_clause_work,
            )
        output = np.asarray(
            qc.execute(
                node_basis_state(self.layout, path),
                backend=self.backend, device=self.device, dtype=self.dtype,
            ).state
        )
        index = int(np.argmax(np.abs(output) ** 2))
        accepted = bool((index >> self.layout.accept_work) & 1)
        rejected = bool((index >> self.layout.reject_work) & 1)
        if accepted and rejected:
            raise RuntimeError("accept and reject oracles overlap on a node")
        return accepted, rejected

    def detect(self, prefix: Sequence[int]) -> bool:
        from unitarylab.library import QPE

        node = _validate_bits(prefix, label="prefix")
        _require(len(node) <= self.problem.max_depth, "prefix is deeper than max_depth")
        accepted, rejected = self._query_predicates(node)
        if accepted or rejected or len(node) == self.problem.max_depth:
            method = (
                "accept-oracle"
                if accepted
                else "reject-oracle" if rejected else "maximum-depth"
            )
            self.evidence.append(DetectionEvidence(list(node), method, accepted))
            return accepted
        r_a, r_b, walk = build_walk_circuit(self.problem, root_depth=len(node))
        controlled = walk.control(1)
        qpe = QPE(
            walk,
            self.phase_bits,
            prepare_target=build_node_prepare_circuit(self.layout, node),
            return_circuit=True,
        )
        state = np.asarray(
            qpe.execute(backend=self.backend, device=self.device, dtype=self.dtype).state
        )
        phase_size = 1 << self.phase_bits
        magnitudes = np.abs(state.reshape(-1, phase_size))
        magnitudes **= 2
        probabilities = np.sum(magnitudes, axis=0).real
        del magnitudes
        probabilities /= probabilities.sum()
        samples = self.rng.choice(phase_size, size=self.shots, p=probabilities)
        values, counts = np.unique(samples, return_counts=True)
        phase_counts = {
            format(int(value), f"0{self.phase_bits}b"): int(count)
            for value, count in zip(values, counts)
        }
        zero_count = phase_counts.get("0" * self.phase_bits, 0)
        exists = zero_count / self.shots > self.threshold
        self.evidence.append(
            DetectionEvidence(
                list(node),
                "controlled-W QPE",
                exists,
                float(probabilities[0]),
                int(zero_count),
                self.shots,
                self.threshold,
                self.phase_bits,
                dict(
                    sorted(phase_counts.items(), key=lambda item: item[1], reverse=True)
                ),
                circuit_summary(qpe),
            )
        )
        if not node:
            self._root_circuits = r_a, r_b, walk, controlled, qpe
        return exists

    def find_solution(self) -> BitPath | None:
        if not self.detect(()):
            return None
        current: BitPath = ()
        while True:
            accepted, rejected = self._query_predicates(current)
            if accepted:
                return current
            if rejected or len(current) == self.problem.max_depth:
                return None
            for branch in (0, 1):
                candidate = (*current, branch)
                if self.detect(candidate):
                    current = candidate
                    break
            else:
                return None

    def _resource_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "node_encoding": {
                "path_qubits": len(self.layout.path),
                "one_hot_depth_qubits": len(self.layout.depth),
                "predicate_and_work_qubits": self.layout.num_qubits
                - len(self.layout.path)
                - len(self.layout.depth),
                "walk_system_qubits": self.layout.num_qubits,
            },
            "child_oracle": circuit_summary(build_child_oracle_circuit(self.problem)),
        }
        if self._root_circuits:
            r_a, r_b, walk, controlled, qpe = self._root_circuits
            report.update(
                {
                    name: circuit_summary(circuit)
                    for name, circuit in (
                        ("R_A", r_a),
                        ("R_B", r_b),
                        ("W", walk),
                        ("controlled_W", controlled),
                        ("QPE", qpe),
                    )
                }
            )
            report.update(
                {
                    "QPE_controlled_W_applications": (1 << self.phase_bits) - 1,
                    "QPE_phase_precision": 1.0 / (1 << self.phase_bits),
                    "W_expanded": circuit_summary(walk.decompose(n=1)),
                    "W_native_default_basis": circuit_summary(walk.transpile()),
                }
            )
        return report

    def _write_artifacts(
        self, output_dir: Path, *, full_artifacts: bool = False
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(
            "MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve())
        )
        artifacts: dict[str, str] = {}
        warnings: list[str] = []

        def record(key: str, filename: str, circuit: Any, title: str) -> None:
            path = output_dir / filename
            circuit.draw(filename=str(path), title=title)
            artifacts[key] = str(path.resolve())

        def qasm(key: str, filename: str, circuit: Any) -> None:
            path = output_dir / filename
            try:
                circuit.to_qasm_file(str(path), transpile=True)
            except (TypeError, ValueError) as exc:
                circuit.to_qasm_file(str(path), transpile=False)
                warnings.append(
                    f"{key}: native-basis transpilation failed; exported high-level OpenQASM 3 instead ({type(exc).__name__}: {exc})"
                )
            artifacts[key] = str(path.resolve())

        if self._root_circuits:
            r_a, r_b, walk, controlled, qpe = self._root_circuits
            record(
                "circuit_diagram",
                "quantum_backtracking_circuit.png",
                qpe,
                "Quantum Backtracking: controlled-W QPE",
            )
            record(
                "W_expanded_diagram",
                "W_expanded.png",
                walk.decompose(n=1),
                "W expanded to R_A/R_B gates",
            )
            if full_artifacts:
                child = build_child_oracle_circuit(self.problem)
                record(
                    "child_oracle_diagram",
                    "child_oracle.png",
                    child,
                    "Reversible child oracle",
                )
                qasm("child_oracle_qasm", "child_oracle.qasm", child)
                for name, circuit in (
                    ("R_A", r_a),
                    ("R_B", r_b),
                    ("W", walk),
                    ("controlled_W", controlled),
                ):
                    record(f"{name}_diagram", f"{name}.png", circuit, name)
                qasm("W_qasm", "W.qasm", walk)
        else:
            child = build_child_oracle_circuit(self.problem)
            record(
                "circuit_diagram",
                "quantum_backtracking_circuit.png",
                child,
                "Quantum Backtracking: reversible child oracle",
            )
            if full_artifacts:
                qasm("child_oracle_qasm", "child_oracle.qasm", child)
        evidence = next(
            (item for item in self.evidence if item.method == "controlled-W QPE"), None
        )
        if evidence and evidence.phase_counts:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = sorted(evidence.phase_counts)
            values = [evidence.phase_counts[label] for label in labels]
            figure, axis = plt.subplots(figsize=(10, 4.8))
            axis.bar(
                labels,
                values,
                color=[
                    "#ef4444" if label == "0" * self.phase_bits else "#2563eb"
                    for label in labels
                ],
            )
            axis.axhline(
                self.shots * self.threshold,
                color="#111827",
                linestyle="--",
                label=f"threshold {self.threshold:.3f} = {self.shots * self.threshold:.1f} counts",
            )
            axis.set(
                title="Root-subtree QPE measurement evidence",
                xlabel="Phase-register outcome",
                ylabel="Counts",
            )
            axis.tick_params(axis="x", rotation=45)
            axis.legend()
            figure.tight_layout()
            path = output_dir / "qpe_measurement_histogram.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            artifacts["measurement_histogram"] = str(path.resolve())
        if warnings:
            path = output_dir / "qasm_export_warnings.txt"
            path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
            artifacts["qasm_export_warnings"] = str(path.resolve())
        return artifacts

    def solve(
        self, output_dir: str | Path | None = None, *, full_artifacts: bool = False
    ) -> SolveResult:
        solution = self.find_solution()
        names = tuple(
            getattr(
                self.problem,
                "variables",
                tuple(f"x{i + 1}" for i in range(self.problem.max_depth)),
            )
        )
        assignment = (
            None
            if solution is None
            else {name: int(bit) for name, bit in zip(names, solution)}
        )
        artifacts = (
            self._write_artifacts(Path(output_dir), full_artifacts=full_artifacts)
            if output_dir is not None
            else {}
        )
        return SolveResult(
            self.problem.name,
            solution is not None,
            None if solution is None else list(solution),
            assignment,
            self.evidence,
            self._resource_report(),
            artifacts,
        )


def _write_result_files(result: SolveResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evidence = [
        f"- prefix `{item.prefix}` — {item.method}: exists=`{item.exists}`, P(phase=0)=`{item.zero_phase_probability}`, zero count=`{item.zero_phase_count}/{item.shots}`"
        for item in result.evidence
    ]
    artifacts = [f"- {key}: `{value}`" for key, value in result.artifacts.items()]
    report = [
        f"# Quantum Backtracking Report: {result.problem}",
        "",
        f"- Exists: `{result.exists}`",
        f"- Solution path: `{result.solution_path}`",
        f"- Business assignment: `{result.solution_assignment}`",
        "",
        "## Measurement evidence",
        "",
        *evidence,
        "",
        "## Resource report",
        "",
        "```json",
        json.dumps(result.resource_report, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Circuit artifacts",
        "",
        *artifacts,
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _print_backtracking_summary(
    result: SolveResult, *, load_seconds: float, solve_seconds: float, output_dir: Path
) -> None:
    root = next(
        (
            item
            for item in result.evidence
            if not item.prefix and item.method == "controlled-W QPE"
        ),
        None,
    )
    assignment = json.dumps(result.solution_assignment, ensure_ascii=False)
    print(f"""Quantum Backtracking Search
Problem: {result.problem}
Solution exists: {result.exists}
Solution path: {result.solution_path}
Business assignment: {assignment}""")
    if root:
        measured = root.zero_phase_count / root.shots
        print(
            f"Root QPE evidence: exact P(phase=0)={root.zero_phase_probability:.4%}; sampled={measured:.4%} ({root.zero_phase_count}/{root.shots}); threshold={root.threshold:.4%}"
        )
    print(f"Detection records: {len(result.evidence)}")
    resources = result.resource_report
    if "W_expanded" in resources:
        walk = resources["W_expanded"]
        print(
            f"Expanded W: {walk['num_qubits']} qubits; depth={walk['depth']}; size={walk['size']}"
        )
    if "QPE" in resources:
        qpe = resources["QPE"]
        print(
            f"QPE: {qpe['num_qubits']} qubits; depth={qpe['depth']}; size={qpe['size']}; controlled-W calls={resources['QPE_controlled_W_applications']}"
        )
    print(f"Problem loading: {load_seconds:.6f} s")
    print(f"Solving, sampling and output: {solve_seconds:.6f} s")
    print(f"Total (excluding imports): {load_seconds + solve_seconds:.6f} s")
    print(f"Result JSON: {(output_dir / 'result.json').resolve()}")
    print(f"Readable report: {(output_dir / 'report.md').resolve()}")
    for label, path in result.artifacts.items():
        print(f"{label}: {path}")


def research_main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problem",
        type=Path,
        nargs="?",
        default=script_dir / "quantum_backtracking_example_problem.json",
        help="JSON business-problem file (default: example_problem.json)",
    )
    for flag, default, kind in (
        ("--phase-bits", 3, int),
        ("--shots", 128, int),
        ("--seed", 7, int),
        ("--threshold", 3.0 / 8.0, float),
    ):
        parser.add_argument(flag, type=kind, default=default)
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "backtracking_run"
    )
    diagrams = parser.add_mutually_exclusive_group()
    diagrams.add_argument("--diagrams", dest="diagrams", action="store_true")
    diagrams.add_argument("--skip-diagrams", dest="diagrams", action="store_false")
    parser.set_defaults(diagrams=True)
    parser.add_argument("--full-artifacts", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    problem = problem_from_dict(json.loads(args.problem.read_text(encoding="utf-8")))
    loaded = perf_counter()
    solver = QuantumBacktrackingSolver(
        problem,
        phase_bits=args.phase_bits,
        shots=args.shots,
        seed=args.seed,
        threshold=args.threshold,
    )
    result = solver.solve(
        args.output_dir if args.diagrams or args.full_artifacts else None,
        full_artifacts=args.full_artifacts,
    )
    _write_result_files(result, args.output_dir)
    finished = perf_counter()
    _print_backtracking_summary(
        result,
        load_seconds=loaded - started,
        solve_seconds=finished - loaded,
        output_dir=args.output_dir,
    )
    if args.print_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


from typing import Any, Dict, List, Optional
import os


class BaseAlgorithm:
    """Shared base class for algorithm result/log formatting."""

    def __init__(self, name: str, prefix: str = "", text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.name = self.__class__.__name__ if name == "" else name
        self.prefix = self.name[:3].upper() if prefix == "" else prefix
        self.prefix = '[' + self.prefix + ']' if not self.prefix.startswith('[') else self.prefix

        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.status = ""
        self.input: Dict[str, Any] = {}
        self.output: Dict[str, Any] = {}
        self.info: list = []
        self.summary: str = ""

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the algorithm with given parameters. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the run() method.")
    
    def _auxiliary_method(self, *args, **kwargs) -> Any:
        """An example auxiliary method that can be used by subclasses."""
        pass

    def log(self, message) -> None:
        """Print a message and save it into info logs."""
        if isinstance(message, str):
            msg = self.prefix + ": " + message
            print(msg)
            self.info.append(message)
        elif isinstance(message, dict):
            for key, value in message.items():
                msg = f"        - {key} = {value}"
                print(msg)

    def update_input(self, input_dict: Dict[str, Any]) -> None:
        """Update the input parameters dictionary."""
        self.input.update(input_dict)
        print(f"{self.prefix}: Starting {self.name}")
        print(f"{self.prefix}: Input parameters: ")
        self.log(self.input)

    def update_output(self, output_dict: Dict[str, Any]) -> None:
        """Update the output information dictionary."""
        self.output.update(output_dict)
        print(f"{self.prefix}: Output information: ")
        self.log(self.output)

    def save_circuit(self, circuit, name=None) -> str:
        """Save the circuit into a file."""
        if name is None:
            filename = f"{self.name.replace(' ', '_').lower()}_circuit.svg"
        else:
            filename = f"{name}.svg"
        
        filepath = f"{self.algo_dir}/{filename}"
        circuit.draw(filename=filepath, title=f"{self.name} Circuit")
        self.log(f"  Circuit diagram saved: {filepath}")
        return filepath

    def save_txt(self) -> None:
        """Save the algorithm result into a file."""
        filename = f"{self.name.replace(' ', '_').lower()}_result.txt"
        filepath = f"{self.algo_dir}/{filename}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.format_result_ascii())
        self.log(f"  Result saved: {filepath}")
        return filename

    def _build_return_dict(self, success: bool, circuit_path, filepath, circuit=None) -> Dict[str, Any]:
        """Build a dictionary containing all relevant information for returning."""
        success = 'ok' if success else 'failed'
        if isinstance(filepath, str):
            filepath = [filepath]
        files = []
        for filename in filepath:
            files.append({"format": filename[-3:], "filename": filename})
        
        result = {"status": success, "circuit_path": circuit_path, "plot": files, "circuit": circuit}
        return result.update(self.output) or result

    def format_result_ascii(self) -> str:
        """Format the most recent algorithm result as text output."""
        result = ""
        # Algorithm start
        if self.text_mode == "plain":
            result += '\n' + f'{"=" * 25}\n{self.name} Algorithm Result\n{"=" * 25}\n'
        elif self.text_mode == "legacy":
            result += '=' * 70 + '\n' + f"{' ' * 10}⚛️ {self.name} Algorithm Result ⚛️\n" + '=' * 70 + '\n'

        # Solve status
        if self.text_mode == "plain":
            result += f"Status: {self.status}\n\n"
        elif self.text_mode == "legacy":
            result += f"📊 Status: {self.status}\n\n"

        # Input parameters
        if self.text_mode == "plain":
            result += "Input parameters:\n"
        elif self.text_mode == "legacy":
            result += "-" * 70 + '\n' + "📥 Input parameters:\n" + "-" * 70 + "\n"
        result += "\n".join([f"  - {key}: {value}" for key, value in self.input.items()]) + "\n\n"

        # Runtime logs
        if self.text_mode == "plain":
            result += "Runtime logs:\n"
        elif self.text_mode == "legacy":
            result += "-" * 70 + '\n' + "📝 Runtime logs:\n" + "-" * 70 + "\n"
        result += "\n".join([f"  - {line}" for line in self.info]) + "\n\n"

        # Output information
        if self.text_mode == "plain":
            result += "Output information:\n"
        elif self.text_mode == "legacy":
            result += "-" * 70 + '\n' + "📤 Output information:\n" + "-" * 70 + "\n"
        result += "\n".join([f"  - {key}: {value}" for key, value in self.output.items()]) + "\n\n"

        # Summary
        if self.text_mode == "plain":
            result += "Summary:\n"
        elif self.text_mode == "legacy":
            result += "-" * 70 + '\n' + "🔍 Summary:\n" + "-" * 70 + "\n"
        result += f"  {self.summary}\n"

        return result

DEFAULT_PROBLEM = {
    "type": "prefix_constraints",
    "name": "default-prefix-search",
    "max_depth": 2,
    "accepted_paths": ["10"],
    "rejected_prefixes": ["0"],
}


def _parse_problem(value):
    if value in (None, "", "None", "null"):
        return dict(DEFAULT_PROBLEM)
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("problem_json must describe a JSON object")
    return data


class QuantumBacktrackingAlgorithm(BaseAlgorithm):
    """Solve binary constraint trees using walk reflections and QPE."""

    def __init__(self, text_mode: str = "plain", algo_dir: str = None):
        if algo_dir is None:
            algo_dir = os.path.join(
                os.getcwd(),
                "results",
                "search",
                "quantum_backtracking",
            )
        os.makedirs(algo_dir, exist_ok=True)
        super().__init__("Quantum Backtracking", "QBT", text_mode, algo_dir)

    def run(
        self,
        problem_json: str = "",
        phase_bits: int = 3,
        shots: int = 128,
        seed: int = 7,
        threshold: float = 0.375,
        backend: str = "numpy",
        device: str = "cpu",
        dtype=np.complex128,
    ) -> dict[str, Any]:
        """Run quantum backtracking on a CNF or prefix-constraint problem."""
        phase_bits, shots, seed = int(phase_bits), int(shots), int(seed)
        threshold = float(threshold)
        problem_data = _parse_problem(problem_json)
        problem = problem_from_dict(problem_data)
        self.update_input(
            {
                "Problem": problem.name,
                "Problem type": problem_data.get("type", "binary_cnf"),
                "Maximum depth": problem.max_depth,
                "Phase bits": phase_bits,
                "Shots": shots,
                "Seed": seed,
                "Threshold": threshold,
                "Backend": backend,
                "Device": device,
                "Dtype": str(np.dtype(dtype)),
            }
        )
        self.log("Stage 1/4: Constructing reversible accept/reject predicates...")
        started = perf_counter()
        solver = QuantumBacktrackingSolver(
            problem,
            phase_bits=phase_bits,
            shots=shots,
            seed=seed,
            threshold=threshold,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        self.log("Stage 2/4: Detecting solution-bearing subtrees with walk-QPE...")
        result = solver.solve()
        self.log("Stage 3/4: Recovering and verifying the solution path...")
        elapsed = perf_counter() - started
        self.status = "success" if result.exists else "no_solution"
        self.summary = (
            f"Solution path {result.solution_path}."
            if result.exists
            else "No accepted path was detected."
        )
        output = result.to_dict()
        output.update({"Elapsed time (s)": elapsed})
        self.update_output(output)
        self.log("Stage 4/4: Exporting representative circuit and result files...")
        if solver._root_circuits:  # pylint: disable=protected-access
            circuit = solver._root_circuits[-1]  # pylint: disable=protected-access
        else:
            circuit = build_child_oracle_circuit(problem)
        circuit_path = self.save_circuit(circuit)
        filename = self.save_txt()
        return self._build_return_dict(True, circuit_path, filename, circuit)


def test(
    problem_json="",
    phase_bits=3,
    shots=128,
    seed=7,
    threshold=0.375,
    backend="numpy",
    device="cpu",
    dtype="complex128",
):
    """Standard test entry point."""
    algorithm = QuantumBacktrackingAlgorithm(text_mode="legacy")
    return algorithm.run(
        problem_json, phase_bits, shots, seed, threshold,
        backend=backend, device=device, dtype=np.dtype(dtype).type,
    )


def main(argv=None) -> None:
    """Describe the bundled implementation; execute only when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--problem-json", default="", help="Inline CNF/prefix JSON; empty selects the included example.")
    parser.add_argument("--phase-bits", type=int, default=3)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.375)
    parser.add_argument("--backend", default="numpy")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    parameters = {
        "problem_json": args.problem_json, "phase_bits": args.phase_bits,
        "shots": args.shots, "seed": args.seed, "threshold": args.threshold,
        "backend": args.backend, "device": args.device, "dtype": args.dtype,
    }
    if not args.execute:
        print(json.dumps({
            "mode": "diagnostic; algorithm and exports not executed",
            "variant": "gate-level binary backtracking with reversible oracles, W=R_B R_A and QPE",
            "parameters": parameters,
            "default_problem": DEFAULT_PROBLEM,
            "default_scale": {"max_depth": 2, "walk_qubits": 8, "phase_bits": 3, "qpe_qubits": 11, "child_oracle_qubits": 14},
            "dependencies": ["NumPy", "UnitaryLab with Circuit/QPE and drawing dependencies"],
            "optional_dependencies": ["Matplotlib for research PNG exports"],
            "conventions": [
                "phase register occupies low-order bits; path follows variable order",
                "default numpy/cpu/complex128; all predicate/QPE executes receive these settings",
                "one NumPy RNG per solver; branches visited 0 before 1",
                "strict zero_phase_count/shots > threshold; finite sampling can miss solutions",
            ],
            "planned_checks_not_run": [
                "prefix/CNF validation and reversible predicate acceptance",
                "QPE probability normalization and phase count total equals shots",
                "fixed-seed evidence comparison against selected baseline",
                "SVG/TXT export; exists=False is not an unsatisfiability proof",
            ],
            "execute_hint": "Pass --execute for a small real run; diagnostics are not validation results.",
        }, ensure_ascii=False, indent=2))
        return
    parameters["dtype"] = np.dtype(args.dtype).type
    result = QuantumBacktrackingAlgorithm(algo_dir=args.output_dir).run(**parameters)
    problem = problem_from_dict(_parse_problem(args.problem_json))
    path = result["solution_path"]
    print(json.dumps({
        "mode": "executed",
        "status": result["status"], "exists": result["exists"],
        "solution_path": path, "solution_assignment": result["solution_assignment"],
        "returned_path_is_accepted": None if path is None else problem.classical_accept(tuple(path)),
        "evidence": result["evidence"], "resource_report": result["resource_report"],
        "circuit_path": result["circuit_path"], "plot": result["plot"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
