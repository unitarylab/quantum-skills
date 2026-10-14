---
name: quantum-backtracking
description: "Explain or implement quantum backtracking on small binary CNF or prefix-constraint trees using the bundled implementation. Use for constrained-tree solution search; finite sampling cannot prove unsatisfiability."
---

# Quantum Backtracking

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Quantum Backtracking for binary CNF or prefix-constraint trees.

The implementation encodes a constrained binary tree with reversible accept/reject predicates, constructs alternating local reflections, uses quantum phase estimation to detect candidate subtrees, and selects branches classically. It supports SAT-style CNF and accepted-path/rejected-prefix problems; accepted prefixes may produce partial assignments.

When using this skill:

- **Explanation:** Explain the constraint tree, reversible predicates, depth encoding, reflection walk, QPE-based detection, classical branch selection, and finite-sampling limitations. Do not generate code unless requested.
- **Run or reuse:** Generate standalone task code that imports `QuantumBacktrackingAlgorithm` from the installed `unitarylab_algorithms` package when its public API satisfies the task. Use the `Minimal Manual Implementation` when independent reconstruction, modification, or porting is requested. Never import or depend on bundled scripts.
- **Debugging:** Start with the small documented prefix problem or a one-variable CNF, fixed precision, shots, and seed. Check classical predicates and returned-path acceptance before changing code; separate oracle, reflection, phase-order, sampling, and backend errors.
- **Modification or reimplementation:** Preserve predicate uncomputation, reflection order, low-bit phase marginalization, numerical precision, RNG call order, branch order, and the parameter/result contract.
- **Reference scripts:** Read `scripts/algorithm.py`, `scripts/quantum_backtracking_implementation.py`, and the included JSON example only to understand algorithm logic, parameters, conventions, and outputs when needed. Never directly run, import, call, or dynamically execute these code files. For execution or validation, generate standalone task code that calls the installed public API, or independently reconstruct the algorithm when requested; execute only that generated code within the user's requested scope.
- **Validation:** Verify reversible predicate behavior, phase-probability normalization, sample totals, and classical acceptance of returned paths or assignments on small problems. Report sampling and precision limits; `exists=False` is not a proof of unsatisfiability, and `status='ok'` only means the call completed.

## Scope and Limitations

The implemented variant uses reversible accept/reject oracles, one-hot depth encoding, local star reflections, W=R_B R_A, QPE subtree detection and classical branch selection. It supports `prefix_constraints` and `binary_cnf`; it is not a general constraint language or a classical exhaustive solver. Accepted prefixes can produce partial assignments.

Defaults are numpy/cpu/complex128. Execution settings reach both predicate and QPE executes. Backend/device support depends on installed UnitaryLab and hardware; NumPy is CPU-only. The public API accepts a NumPy-compatible dtype; intermediate NumPy calculations retain source precision. No silent precision reduction or backend fallback is added. Other configurations have not been run-validated.

Finite QPE resolution and finite-shot sampling can miss a solution. `exists=False` is not a proof of unsatisfiability; `status='ok'` only means the call completed. Dense simulation grows exponentially. There is no Python API maximum for depth or phase_bits; the original Web phase_bits maximum 8 is UI metadata, not enforced by this leaf's API.

The installed public API writes its default outputs under `<cwd>/results/search/quantum_backtracking`, or under an explicitly supplied `algo_dir`. [The included JSON](scripts/example_problem.json) is a reference-only input-schema example. It is not required for the built-in problem or a runtime dependency; define task inputs independently and pass them through `problem_json`.

## Overview

Input describes a binary constraint tree plus phase_bits, shots, seed and threshold. Output is a detected accepted path/assignment or no detection, ordered evidence, circuit resource summaries and SVG/TXT artifacts.

1. Parse and validate a CNF or prefix problem.
2. Allocate path, one-hot depth and predicate/work registers.
3. Query reversible accept/reject predicates on each candidate node.
4. For a nonterminal node, build parity reflections and QPE of their walk.
5. Normalize the phase marginal and sample with the solver's local RNG.
6. Use strict zero-phase frequency > threshold to decide whether to descend.
7. Check child 0 then child 1 until accepted or no branch is detected.
8. Build full resource reports and export the representative circuit and text result.

## Prerequisites

Understand reversible Boolean predicates, computational-basis encoding, controlled rotations/reflections, phase estimation and discrete sampling. Public API calls require the installed `unitarylab-algorithms` package exporting `QuantumBacktrackingAlgorithm`. Independent reconstruction requires NumPy and a compatible UnitaryLab exposing `Circuit` and `unitarylab.library.QPE`, with its normal simulation and drawing dependencies. Matplotlib is used only in optional research PNG/histogram exports. No dependency version guarantee or installation was validated.

The built-in prefix problem requires no remote source/data or external algorithm checkout. Generate API calls against the installed package without modifying `sys.path`. The bundled sources and research example JSON are reading material for independent reconstruction. The reference source retains `os.environ.setdefault` for missing numerical thread variables; the public API creates its requested output directory.

## Reference Implementation Example

`QuantumBacktrackingAlgorithm` is a public class exported by the installed
`unitarylab_algorithms` package. Generate the following as standalone task code in
the user's task directory. Run it only when execution, debugging, or verification
is requested; it writes circuit and result artifacts under the chosen `algo_dir`.

```python
import numpy as np
from unitarylab_algorithms import QuantumBacktrackingAlgorithm

algorithm = QuantumBacktrackingAlgorithm(
    text_mode="plain", algo_dir="results/quantum_backtracking"
)
result = algorithm.run(
    problem_json="",  # Built-in prefix-constraint problem
    phase_bits=3,
    shots=128,
    seed=7,
    threshold=0.375,
    backend="numpy",
    device="cpu",
    dtype=np.complex128,
)
print(result["exists"], result["solution_path"], result["solution_assignment"])
print(result["circuit_path"], result["plot"])
```

Resolve this import from the active environment's installed package, without adding
the skill directory or another checkout to `sys.path`. If the installed version
lacks the export, report that API/version gap. For independent reconstruction, use
the `Minimal Manual Implementation`; never invoke or import the bundled script class.

## Core Parameters Explained

| Parameter / entry | Type and actual default | Meaning and constraints |
|---|---|---|
| constructor text_mode | str, plain | plain/legacy text formatting |
| constructor algo_dir | str or None, None | None uses `<cwd>/results/search/quantum_backtracking`; constructor creates it |
| run problem_json | str or dict, empty string | empty/None/"None"/"null" selects built-in depth-two prefix problem; otherwise JSON object |
| run phase_bits | int, 3 | >=1; QPE precision 2^-p and 2^p-1 controlled-W applications |
| run shots | int, 128 | >=1; NumPy phase samples per nonterminal detection, not repeated Circuit measurement execution |
| run seed | int, 7 | non-negative seed required by NumPy default_rng; one RNG per solver |
| run threshold | float, 0.375 | strictly in (0,1); measured zero frequency must strictly exceed it |
| run backend | str, numpy | forwarded to every predicate and QPE execution |
| run device | str, cpu | backend-dependent, numpy CPU only; backend reports unsupported modes |
| run dtype | NumPy-compatible dtype, np.complex128 | execution precision; NumPy dtype conversion also used for input logging |

run casts phase_bits/shots/seed to int and threshold to float as in the source. Prefix max_depth>=1; paths contain only 0/1, are within max_depth and unique within each list; no accepted path may descend from a rejected prefix. CNF requires nonempty unique variable names; literals are signed one-based indices; clauses are nonempty with valid indices and no repeated variable in a clause. CNF depth equals variable count. An empty clauses list is allowed. Missing type selects binary_cnf. Caller input is not mutated.

For implementation-level analysis, the lower-level Solver takes a `ReversibleBinaryBacktrackingProblem` with the same sampling and backend/device/dtype defaults, and exposes detect, find_solution and solve. `solve(output_dir=None, full_artifacts=False)` returns SolveResult; optional artifact output is separate from the public algorithm's standard run. These details support independent reconstruction; they do not permit invoking bundled scripts.

## Return Fields

| Public field | Type | Meaning and limits |
|---|---|---|
| status | str | ok after completed run, even when exists=False |
| circuit_path / circuit | str / Circuit | root QPE SVG/circuit; child oracle if no root QPE occurred |
| plot | list[dict] | saved TXT metadata with format/filename; filename is not an absolute path |
| problem / exists | str / bool | problem name and sampled detection outcome |
| solution_path | list[int] or None | path bits in variable order |
| solution_assignment | dict[str,int] or None | CNF variable names or x1... names; can be partial for accepted prefixes |
| evidence | list[dict] | ordered predicate/QPE evidence with prefix/method/exists; probability/count/shots/threshold/phase_bits/count map/circuit summary may be None for terminal predicates |
| resource_report | dict | node registers and child oracle, plus root R_A/R_B/W/controlled_W/QPE/decomposed/transpiled summaries when available |
| artifacts | dict | empty in standard run; optional lower-level research exports populate it |
| Elapsed time (s) | float | solver construction/solve/resource report; excludes final SVG/TXT export |

No full quantum state is returned by standard run. When verification is requested, check the returned path against the task's classical acceptance predicate; a completed API call alone does not prove that a solution exists.

## Implementation Architecture

- PrefixConstraintProblem / BinaryCNFProblem validate data and append reversible predicates. problem_from_dict selects the two variants; _parse_problem handles the default example.
- NodeLayout.for_problem allocates registers; node_basis_index/state and build_node_prepare_circuit encode a specific node.
- _flagged_target computes flags, acts on the target and uncomputes in reverse order. CNF _false_specs matches falsifying assignments for fully bound clauses.
- append_local_star_reflection uses controlled Givens and child-splitting rotations around a phase flip; append_parity_reflection adds predicate queries and their uncomputation.
- build_walk_circuit assembles R_A, R_B and their sequential walk. Solver._query_predicates executes the node circuit and reads flags using argmax.
- Solver.detect performs terminal shortcuts or QPE, phase marginalization, normalization and RNG sampling; find_solution descends in fixed branch order, preserving repeated predicate queries.
- SolveResult stores evidence/resources/artifacts. Solver._resource_report retains child, decomposed and transpiled summaries.
- BaseAlgorithm and QuantumBacktrackingAlgorithm wrap logs, inputs/outputs, SVG/TXT and the result envelope. All implementation and private helpers are in one file.

## Understanding the Key Quantum Components

For depth d, path uses d qubits. Depth uses d+1 one-hot qubits; accept/reject use two flags plus r rejection-work qubits. Thus N=2d+3+r walk-system qubits. r is CNF clause count or rejected-prefix count. For node x of length k, depth[k]=1, path bits encode x and all unused/work bits start at zero.

The original solver queries accept/reject with a circuit, not an exhaustive classical solver. An overlap raises RuntimeError. Reversible predicates are undone after use in each reflection. Local diffusion operates on the parent and two children; its root has extra weight d-root_depth. Alternating local reflections give the walk. QPE uses p additional low-order phase bits and the prepared node register as target. No measurement register is sampled by execute: NumPy samples the phase marginal of the evolved state.

Phase labels use width-p binary integer values; the all-zero label is the detection event. Path bits remain in q0/variable order, not reversed count-string order. Changing register order, conjugation, rotation signs or global phase can alter walk-QPE evidence, even if a returned assignment happens to match.

## Theory-to-Code Mapping

| Object / step | Bundled source | Convention / check |
|---|---|---|
| Node basis | NodeLayout, node_basis_index | one-hot depth[len(path)] and path ones, little-index basis sum(1<<qubit) |
| Accept/reject predicates | append_accept_oracle / append_reject_oracle | controlled flag toggles and reverse uncomputation |
| Local star state | append_local_star_reflection | normalized [1,1,1], or [1,sqrt(d-root_depth),sqrt(d-root_depth)] at subtree root |
| R_A / R_B | append_parity_reflection | alternating depths relative to subtree root |
| W=R_B R_A | build_walk_circuit | append R_A then R_B; control(1) included for reporting |
| Controlled powers / inverse QFT | QPE(walk,p,prepare_target=...,return_circuit=True) | uses public QPE, not an external algorithm implementation |
| Phase marginal | detect reshape(-1,2**p), abs, square, sum(axis=0) | low-bit phase register; normalization before choice |
| Existence detector | zero_count/shots > threshold | strict boundary; exact phase probability also recorded |
| Recovery | find_solution | child0 then child1; terminate only on accept/query/maximum depth rules |
| Resource and artifact evidence | _resource_report / save_circuit / save_txt | preserve complete summaries and representative circuit |

## Mathematical Deep Dive

For a local parent-child star, the diffusion state has normalized amplitudes a=(a0,a1,a2). Non-root weights are (1,1,1); subtree-root weights are (1,sqrt(d-k),sqrt(d-k)). The code computes g=-2*asin(hypot(a1,a2)) and s=2*atan2(a2,a1), applies splitting/Givens rotations with -s/-g, flips the unmarked valid parent phase, then applies g/s to undo preparation. Accept/reject flags change which phases are applied; this is not unconditional Grover diffusion over all bitstrings.

Let W=R_B R_A and W|u_j>=exp(i theta_j)|u_j>. Decompose a prepared node as |x>=sum_j c_j|u_j>. For p phase bits and M=2^p, ideal QPE all-zero amplitude on eigencomponent j is

`a_j(0) = (1/M) * sum(l=0..M-1) exp(i*l*theta_j)`.

Thus finite-resolution zero probability is `P0=sum_j |c_j|² |a_j(0)|²`, not only exact zero-eigenphase weight: nearby phases leak into that outcome. The implementation directly computes the normalized phase marginal, samples shots independent phase outcomes, and accepts when K0/shots>threshold. Conditional on a fixed marginal, K0 is binomial with variance shots*P0*(1-P0); fixed seed preserves an RNG sequence, not correctness across every instance.

The root weighting and spectral properties motivate quantum-backtracking detection, but this fixed-resolution implementation does not establish a uniform error bound or a certified unsatisfiability result. Ideal quantum query improvements do not establish a speedup for this classical simulator.

QPE applies controlled-W 2^p-1 times. Its dense state contains 2^(N+p) complex values, about 16*2^(N+p) bytes at complex128, before simulator workspaces. Probability temporaries scale with that state. Child oracle uses N+d+4=3d+7+r qubits and is constructed for the resource report, although not executed by standard solving. Root circuit objects and evidence remain alive for required reports and export. No time or peak-memory measurement was performed.

## Hands-On Example

Use a terminal accepted root to separate deterministic predicate behavior from stochastic QPE. Generate the following standalone task code in the user's task directory; it calls the installed public API without importing bundled sources. Execute it only when execution, debugging, or verification is requested:

```python
import numpy as np
from unitarylab_algorithms import QuantumBacktrackingAlgorithm

problem = {"type":"prefix_constraints", "name":"accepted-root",
           "max_depth":1, "accepted_paths":[""], "rejected_prefixes":[]}
algorithm = QuantumBacktrackingAlgorithm(
    algo_dir="results/quantum_backtracking_accepted_root"
)
result = algorithm.run(
    problem_json=problem, phase_bits=3, shots=128, seed=7,
    threshold=0.375, backend="numpy", device="cpu", dtype=np.complex128,
)
assert result["status"] == "ok" and result["exists"] is True
assert result["solution_path"] == [] and result["solution_assignment"] == {}
assert result["evidence"][0]["method"] == "accept-oracle"
assert "QPE" not in result["resource_report"]
print(result["circuit_path"], result["plot"])
```

These are exact Boolean/list/dictionary expectations from terminal root handling, not measured results. The predicate query should identify the root flag; child-oracle SVG/TXT still export. For the default nonterminal problem, each QPE record must satisfy count total==shots and exists==(zero_count/shots>threshold); if a path is returned it must be [1,0]. Missing that path can be a finite-resolution/sampling false negative. Compare probabilities/count ordering with the selected baseline under identical settings; do not invent a tolerance or relax one to hide drift. All examples remain unexecuted.

## Minimal Manual Implementation

This independent small reconstruction uses the same variant on the depth-one prefix tree whose only accepted leaf is `1`, with no rejected prefixes. It builds predicate flags, both parity reflections, the walk, QPE, sampling and 0-before-1 recovery. It imports only NumPy and public Circuit/QPE. It omits generic CNF/prefix parsing, resource analysis, logs and artifact drawing, which are non-core for this fixed valid instance; returns a plain path/evidence dictionary. It does not cover arbitrary constraints.

```python
import math
import numpy as np
from unitarylab import Circuit
from unitarylab.library import QPE

def manual_backtracking(phase_bits=3, shots=128, seed=7, threshold=0.375):
    if phase_bits < 1 or shots < 1 or not 0 < threshold < 1:
        raise ValueError("invalid phase_bits/shots/threshold")
    # d=1, r=0: path q0, depth q1/q2, accept q3, reject q4.
    n, path_bit, depth, accept, reject = 5, 0, (1, 2), 3, 4
    rng, evidence = np.random.default_rng(seed), []

    def prepare(node):
        qc = Circuit(n)
        qc.x(depth[len(node)])
        if node and node[0]:
            qc.x(path_bit)
        return qc

    def query(node):
        qc = Circuit(n)
        qc.mcx([depth[1], path_bit], accept, control_state=[1, 1])
        initial = np.zeros(1 << n, dtype=complex)
        initial[(1 << depth[len(node)]) + (int(node[0]) if node else 0)] = 1
        state = np.asarray(qc.execute(initial, backend="numpy",
                                     device="cpu", dtype=np.complex128).state)
        index = int(np.argmax(np.abs(state) ** 2))
        a, r = bool((index >> accept) & 1), bool((index >> reject) & 1)
        if a and r:
            raise RuntimeError("accept and reject overlap")
        return a, r

    def givens(qc, parent, child, angle):
        qc.cx(parent, child)
        qc.mcry(angle, [child, path_bit], parent, control_state=[1, 0])
        qc.cx(parent, child)

    def split(qc, parent, child, angle):
        qc.mcry(angle, [child, parent], path_bit, control_state=[1, 0])

    def walk_for(root_depth):
        reflections = [Circuit(n), Circuit(n)]
        for parity, qc in enumerate(reflections):
            for k in range(root_depth + parity, 2, 2):
                if k == 1:  # accepted leaf predicate, no reject predicate
                    qc.cx(path_bit, accept, control_state=1)
                if k < 1:
                    amplitudes = np.array([1., math.sqrt(1-root_depth),
                                           math.sqrt(1-root_depth)])
                    amplitudes /= np.linalg.norm(amplitudes)
                    g = -2 * math.asin(math.hypot(float(amplitudes[1]),
                                                 float(amplitudes[2])))
                    s = 2 * math.atan2(float(amplitudes[2]), float(amplitudes[1]))
                    split(qc, depth[k], depth[k+1], -s)
                    givens(qc, depth[k], depth[k+1], -g)
                    qc.mcz([depth[k+1], path_bit, accept, reject], depth[k],
                           control_state=[0, 0, 0, 0])
                    givens(qc, depth[k], depth[k+1], g)
                    split(qc, depth[k], depth[k+1], s)
                    qc.mcz([accept, reject], depth[k], control_state=[0, 1])
                else:
                    qc.cz(accept, depth[k], control_state=0)
                if k == 1:
                    qc.cx(path_bit, accept, control_state=1)
        walk = Circuit(n)
        for qc in reflections:
            walk.append(qc, target=list(range(n)))
        return walk

    def detect(node):
        accepted, rejected = query(node)
        if accepted or rejected or len(node) == 1:
            method = "accept-oracle" if accepted else "reject-oracle" if rejected else "maximum-depth"
            evidence.append({"prefix":list(node), "method":method, "exists":accepted})
            return accepted
        qpe = QPE(walk_for(len(node)), phase_bits,
                  prepare_target=prepare(node), return_circuit=True)
        state = np.asarray(qpe.execute(backend="numpy", device="cpu",
                                       dtype=np.complex128).state)
        phase_size = 1 << phase_bits
        magnitudes = np.abs(state.reshape(-1, phase_size))
        magnitudes **= 2
        probabilities = np.sum(magnitudes, axis=0).real
        del magnitudes
        probabilities /= probabilities.sum()
        samples = rng.choice(phase_size, size=shots, p=probabilities)
        values, counts = np.unique(samples, return_counts=True)
        phase_counts = {format(int(v), f"0{phase_bits}b"):int(c)
                        for v, c in zip(values, counts)}
        zero = phase_counts.get("0" * phase_bits, 0)
        exists = zero / shots > threshold
        evidence.append({"prefix":list(node), "method":"controlled-W QPE",
                         "exists":exists, "zero_phase_probability":float(probabilities[0]),
                         "zero_phase_count":zero, "shots":shots, "threshold":threshold,
                         "phase_bits":phase_bits,
                         "phase_counts":dict(sorted(phase_counts.items(),
                                                   key=lambda item:item[1], reverse=True))})
        return exists

    solution = None
    if detect(()):
        current = ()
        while True:
            accepted, rejected = query(current)
            if accepted:
                solution = list(current)
                break
            if rejected or len(current) == 1:
                break
            for branch in (0, 1):
                candidate = (*current, branch)
                if detect(candidate):
                    current = candidate
                    break
            else:
                break
    return {"exists":solution is not None, "solution_path":solution, "evidence":evidence}

# Only explicit execution would call manual_backtracking().
```

The subset matches prefix input `max_depth=1, accepted_paths=["1"], rejected_prefixes=[]`. Compare its business result/QPE evidence against the full implementation on that same input, not against the depth-two default. Terminal evidence omits None metadata fields; circuit summaries and artifact envelopes are intentionally omitted. No manual code was executed.

## Debugging Tips

- Missing/incompatible Circuit or QPE: report the actual dependency/API gap; do not locate another checkout, change sys.path or download core source.
- No result has been produced: generating or reading code does not execute it. If execution is requested, run the generated standalone task code that calls the installed API; never run the bundled scripts. Inspect `exists` separately from process completion.
- CNF invalid/missing keys: check variable names, signed one-based literal indices and nonempty clauses. Prefix paths must be binary and within max_depth; contradictory accepted/rejected prefixes are rejected.
- No detected solution: inspect QPE zero probability, sampled count and strict threshold separately. Finite p/shots may miss a valid subtree; do not silently change threshold or add a classical fallback.
- Wrong path: retain 0-before-1 branch order and q0-first variable ordering; count labels are phase integers, not solution paths.
- Backend/dtype/device errors: follow the installed simulator's restrictions. Execution settings do not rewrite all NumPy intermediates; no automatic retry is present.
- Memory/depth growth: account for path, one-hot depth, clause work and phase qubits. Default QPE is 11 qubits; child report has14, not just depth2.
- Missing exports: inspect algo_dir permissions and simulator drawing dependencies. Standard run exports SVG/TXT; PNG/histogram/QASM belong to explicit research solve modes.
- Predicate overlap RuntimeError and resource/transpile errors remain visible. Do not suppress them or remove reports to claim successful execution.
- Separate planned verification from observed evidence. Nothing here has been run-validated.

## Reference Implementation

Reference-only source files:

- [scripts/algorithm.py](scripts/algorithm.py): reference-only example of parameter forwarding and result reporting.
- [scripts/quantum_backtracking_implementation.py](scripts/quantum_backtracking_implementation.py): reversible predicates, reflection walk, QPE detection, branch selection, embedded base, and result/export logic.
- [scripts/example_problem.json](scripts/example_problem.json): a small prefix-constraint input schema example.

Read the relevant source only when implementation details or conventions are unclear. Do not run the code files, import their modules, call their entrypoints, load them dynamically, or use them as runtime dependencies. Generate all task code and task inputs outside the skill package.

For API use, generate standalone task code against the installed `QuantumBacktrackingAlgorithm`. For independent reconstruction, modification, or porting, use the algorithm description, parameter/result tables, theory-to-code mapping, and manual implementation to generate independent UnitaryLab task code. Run and verify only generated code when the requested scope requires execution. Preserve predicate uncomputation, reflection order, low-bit phase marginalization, precision, RNG call order, and branch order unless an algorithm change is separately requested.

When updating this reference, keep parameter defaults, return fields, mathematical mapping, and top-level routing aligned with the reference source. Source examples and provenance alone are not execution evidence.
