
# Quantum Backtracking

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The implementation encodes a constrained binary tree with reversible accept/reject predicates, constructs alternating local reflections, uses quantum phase estimation to detect candidate subtrees, and selects branches classically. It supports SAT-style CNF and accepted-path/rejected-prefix problems; accepted prefixes may produce partial assignments.

Source pointers: [algorithm-library usage example](../scripts/quantum_backtracking_algorithm.py), [teaching reconstruction](../scripts/quantum_backtracking_implementation.py), and [prefix-problem JSON](../scripts/quantum_backtracking_example_problem.json). Preserve predicate uncomputation, reflection order, low-bit phase marginalization, numerical precision, RNG call order, and branch order. Small checks use a prefix problem or one-variable CNF with fixed precision, shots, and seed; inspect predicate reversibility, phase normalization, count totals, and classical acceptance of returned paths.

Algorithm-library source: QuantumBacktrackingAlgorithm and solver (`unitarylab_algorithms/search/quantum_backtracking/algorithm.py`).

## Scope and Limitations

The implemented variant uses reversible accept/reject oracles, one-hot depth encoding, local star reflections, W=R_B R_A, QPE subtree detection and classical branch selection. It supports `prefix_constraints` and `binary_cnf`; it is not a general constraint language or a classical exhaustive solver. Accepted prefixes can produce partial assignments.

Defaults are numpy/cpu/complex128. Execution settings reach both predicate and QPE executes. Backend/device support depends on installed UnitaryLab and hardware; NumPy is CPU-only. The bundled CLI dtype choice is complex64/complex128; intermediate NumPy calculations retain source precision. There is no automatic precision reduction or backend fallback.

Finite QPE resolution and finite-shot sampling can miss a solution. `exists=False` is not a proof of unsatisfiability; `status='ok'` only means the call completed. Dense simulation grows exponentially. There is no Python API maximum for depth or phase_bits; the original Web phase_bits maximum 8 is UI metadata, not enforced by this leaf's API.

The bundled teaching source embeds `BaseAlgorithm`, defaults its output directory to `results/search/quantum_backtracking`, and retains a separate `research_main` interface. Its diagnostic entrypoint and research exports are distinct from the algorithm-library `run()` interface; upstream calls its research entrypoint `main`.

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

`QuantumBacktrackingAlgorithm` is exported by `unitarylab_algorithms`. This example writes circuit and result artifacts under `algo_dir` when executed.

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
| algorithm-library test dtype | NumPy-compatible dtype or str, complex128 | converted with np.dtype(...).type |
| bundled teaching main | eight run options + output-dir + execute | numeric options parsed, dtype limited to complex64/complex128; default diagnostic |
| bundled example main | execute + output-dir | fixed small execution defaults |

run casts phase_bits/shots/seed to int and threshold to float as in the source. Prefix max_depth>=1; paths contain only 0/1, are within max_depth and unique within each list; no accepted path may descend from a rejected prefix. CNF requires nonempty unique variable names; literals are signed one-based indices; clauses are nonempty with valid indices and no repeated variable in a clause. CNF depth equals variable count. An empty clauses list is allowed. Missing type selects binary_cnf. Caller input is not mutated.

Lower-level Solver takes a `ReversibleBinaryBacktrackingProblem` with the same sampling and backend/device/dtype defaults, and exposes detect, find_solution and solve. `solve(output_dir=None, full_artifacts=False)` returns SolveResult; optional artifact output is separate from standard run. `research_main` retains file-problem and PNG/QASM/report options, default seed7/phase_bits3/shots128/threshold3/8 and solver numpy/cpu/complex128; it is not automatically invoked.

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

No full quantum state is returned by standard run. Main's JSON excerpt additionally reports `returned_path_is_accepted` from a classical predicate, or None when no path was returned; this wrapper diagnostic does not change run's result or search flow.

## Implementation Architecture

- PrefixConstraintProblem / BinaryCNFProblem validate data and append reversible predicates. problem_from_dict selects the two variants; _parse_problem handles the default example.
- NodeLayout.for_problem allocates registers; node_basis_index/state and build_node_prepare_circuit encode a specific node.
- _flagged_target computes flags, acts on the target and uncomputes in reverse order. CNF _false_specs matches falsifying assignments for fully bound clauses.
- append_local_star_reflection uses controlled Givens and child-splitting rotations around a phase flip; append_parity_reflection adds predicate queries and their uncomputation.
- build_walk_circuit assembles R_A, R_B and their sequential walk. Solver._query_predicates executes the node circuit and reads flags using argmax.
- Solver.detect performs terminal shortcuts or QPE, phase marginalization, normalization and RNG sampling; find_solution descends in fixed branch order, preserving repeated predicate queries.
- SolveResult stores evidence/resources/artifacts. Solver._resource_report retains child, decomposed and transpiled summaries.
- BaseAlgorithm and QuantumBacktrackingAlgorithm wrap logs, inputs/outputs, SVG/TXT and the result envelope. All implementation and private helpers are in one file.
- Bundled entrypoints provide diagnostic/execute modes; their `research_main` retains separate research exports and the supplied JSON resource. These wrappers are not public `run()` parameters.

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

QPE applies controlled-W 2^p-1 times. Its dense state contains 2^(N+p) complex values, about 16*2^(N+p) bytes at complex128, before simulator workspaces. Probability temporaries scale with that state. Child oracle uses N+d+4=3d+7+r qubits and is constructed for the resource report, although not executed by standard solving. Root circuit objects and evidence remain alive for required reports and export.

## Hands-On Example

Use a terminal accepted root in independently generated task code to separate deterministic predicate behavior from stochastic QPE. Construct the problem below, obtain a result from that standalone implementation, and apply the documented result checks:

```python
problem = {"type":"prefix_constraints", "name":"accepted-root",
           "max_depth":1, "accepted_paths":[""], "rejected_prefixes":[]}
# result is produced by your standalone implementation for this problem.
assert result["status"] == "ok" and result["exists"] is True
assert result["solution_path"] == [] and result["solution_assignment"] == {}
assert result["evidence"][0]["method"] == "accept-oracle"
assert "QPE" not in result["resource_report"]
print(result["circuit_path"], result["plot"])
```

These Boolean/list/dictionary expectations follow terminal-root handling. For the default nonterminal problem, each QPE record must satisfy count total==shots and exists==(zero_count/shots>threshold); if a path is returned it must be [1,0]. Missing that path can be a finite-resolution/sampling false negative. Compare probabilities/count ordering under identical settings, and report numerical residuals separately from exact structural checks.

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
- Diagnostic output has no solution: expected without --execute; planned checks are not test results.
- CNF invalid/missing keys: check variable names, signed one-based literal indices and nonempty clauses. Prefix paths must be binary and within max_depth; contradictory accepted/rejected prefixes are rejected.
- No detected solution: inspect QPE zero probability, sampled count and strict threshold separately. Finite p/shots may miss a valid subtree; do not silently change threshold or add a classical fallback.
- Wrong path: retain 0-before-1 branch order and q0-first variable ordering; count labels are phase integers, not solution paths.
- Backend/dtype/device errors: follow the installed simulator's restrictions. Execution settings do not rewrite all NumPy intermediates; no automatic retry is present.
- Memory/depth growth: account for path, one-hot depth, clause work and phase qubits. The default QPE has 11 qubits; its child-oracle report has 14.
- Missing exports: inspect algo_dir permissions and simulator drawing dependencies. Standard run exports SVG/TXT; PNG/histogram/QASM belong to explicit research solve modes.
- Predicate overlap RuntimeError and resource/transpile errors remain visible. Do not suppress them or remove reports to claim successful execution.

## Reference Implementation

Reference-only source files:

- [scripts/quantum_backtracking_algorithm.py](../scripts/quantum_backtracking_algorithm.py): example parameter forwarding and diagnostic-report structure.
- [scripts/quantum_backtracking_implementation.py](../scripts/quantum_backtracking_implementation.py): reversible predicates, reflection walk, QPE detection, branch selection, embedded base, and result/export logic.
- [scripts/quantum_backtracking_example_problem.json](../scripts/quantum_backtracking_example_problem.json): a small prefix-constraint input schema example.
