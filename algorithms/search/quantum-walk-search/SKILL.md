---
name: quantum-walk-search
description: "Search for marked vertices on small undirected weighted graphs with continuous-time quantum walks using the bundled implementation. Use for multi-marked graph search rather than Glued Trees transport or Grover iterations."
---

# Continuous-Time Quantum Walk Search

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement continuous-time Quantum Walk Search.

The implementation searches for one or more marked vertices on a small undirected weighted graph. It constructs a graph-search Hamiltonian, prepares a uniform state on valid vertices, simulates time evolution with UnitaryLab, and compares marked-state probabilities with an exact reference. This is a marked-vertex search workflow, distinct from Glued Trees transport and discrete Grover iterations.

When using this skill:

- **Explanation:** Explain the graph model, marked-vertex Hamiltonian, initial state, time selection, measurement probabilities, and implementation limitations. Do not generate code unless requested.
- **Run or reuse:** Generate standalone task code that imports `QuantumWalkSearchAlgorithm` from the installed `unitarylab_algorithms` package when its public API satisfies the task. Use the `Minimal Manual Implementation` when independent reconstruction, modification, or porting is requested. Never import or depend on bundled scripts.
- **Debugging:** Start with a small complete or path graph, an explicit evolution time, and a fixed sampling seed. Compare simulated and exact probabilities before changing code; separate Hamiltonian construction, padding, evolution approximation, sampling, and backend errors.
- **Modification or reimplementation:** Preserve adjacency validation, matrix construction and floating-point order, initial-state ownership, vertex indexing, padding conventions, random-call order, and the return contract.
- **Reference scripts:** Read `scripts/algorithm.py` and `scripts/quantum_walk_search_implementation.py` only to understand algorithm logic, parameters, conventions, and outputs when needed. Never directly run, import, call, or dynamically execute these files. For execution or validation, generate standalone task code that calls the installed public API, or independently reconstruct the algorithm when requested; execute only that generated code within the user's requested scope.
- **Validation:** Check probability normalization, padded-state probability, sample totals, marked counts, and agreement with an exact small-graph reference. Include an all-marked case and a non-power-of-two graph when relevant. Report the observed approximation error rather than treating `target_error` as an achieved-error guarantee.

## Scope and Limitations

The variant is multi-marked continuous-time graph search with dense NumPy Hamiltonians
and a UnitaryLab simulation circuit. Built-in graphs are complete, cycle, path and star;
custom inputs are finite approximately symmetric adjacency matrices without self-loops.
The standard run enforces at most32 vertices. The lower-level solve_graph_search
alone can override this with allow_large=True.

The execution defaults are numpy/cpu/complex128. backend/device/dtype are forwarded
to simulator construction and final execution; their support depends on installed
UnitaryLab and hardware. NumPy is CPU-only. Exact spectral work retains original
NumPy precision, and no automatic backend fallback or precision reduction is added.

Preserve the known original limitation: Taylor and some auxiliary-register methods
may produce circuits larger than the D-dimensional initial state used here.
This implementation does not add auxiliary initialization, postselection or probability
reinterpretation. target_error is not an achieved-error guarantee. Explicit zero time
can encounter original zero-step simulator behavior. The sampling seed does not
seed construction of stochastic simulation circuits.

The installed public API writes its default outputs under
`<cwd>/results/search/quantum_walk_search`, or under an explicitly supplied `algo_dir`.
Bundled sources describe the computation and result/export conventions for reading
and independent reconstruction; their entrypoints are not task execution interfaces.

## Overview

Input is an undirected weighted adjacency A, a nonempty set M of marked vertices,
and optional coupling gamma and evolution time t. Outputs include simulated/exact
marked probabilities, sampling evidence, padding probability, circuit resources,
an overview circuit and a text report.

The ordered flow is:

1. Parse graph/marked input and preserve original validation and tolerances.
2. Choose gamma, construct and pad the graph-search Hamiltonian.
3. Prepare a uniform state on valid vertices; select t by exact coarse/fine scanning
   unless the caller supplies it.
4. Construct a Hamiltonian-simulation circuit and execute once from that state.
5. Compute probabilities and an exact reference; draw vertex samples.
6. Aggregate integer counts and export the prepare-evolve-measure circuit and TXT.

## Prerequisites

Understand weighted adjacency, Hermitian eigenproblems, unitary evolution, basis-state
probabilities and sampling. The required packages are NumPy with np.exceptions
and, for public API calls, the installed `unitarylab-algorithms` package exporting
`QuantumWalkSearchAlgorithm`. Independent reconstruction also requires a compatible
UnitaryLab exposing Circuit, Register, ClassicalRegister and
unitarylab.library.hamiltonian_simulation. UnitaryLab's normal simulator and drawing
dependencies are also needed for execution/exports; this delivery does not install them.
Matplotlib is imported only for optional research PNG exports. No new package version
range has been validated, so no compatibility guarantee is claimed.

The default graph requires no input dataset or external algorithm checkout. Generate
API calls against the installed package; use the two linked scripts only as reading
material for independent reconstruction. The reference source sets missing numerical
thread environment variables to1; the public API creates its requested output directory.

## Reference Implementation Example

`QuantumWalkSearchAlgorithm` is a public class exported by the installed
`unitarylab_algorithms` package. Generate the following as standalone task code in
the user's task directory. Run it only when execution, debugging, or verification
is requested; it writes circuit and result artifacts under the chosen `algo_dir`.

```python
import numpy as np
from unitarylab_algorithms import QuantumWalkSearchAlgorithm

algorithm = QuantumWalkSearchAlgorithm(
    text_mode="plain", algo_dir="results/quantum_walk_search"
)
result = algorithm.run(
    num_vertices=8,
    marked="2,5",
    graph_type="complete",
    method="trotter",
    simulation_steps=8,
    shots=128,
    seed=7,
    backend="numpy",
    device="cpu",
    dtype=np.complex128,
)
print(result["success_probability"], result["exact_probability_at_time"])
print(result["counts"], result["circuit_path"], result["plot"])
```

Resolve this import from the active environment's installed package, without adding
the skill directory or another checkout to `sys.path`. If the installed version
lacks the export, report that API/version gap. For independent reconstruction, use
the `Minimal Manual Implementation`; never invoke or import the bundled script class.

## Core Parameters Explained

| Parameter/entry | Type and actual default | Meaning and constraints |
|---|---|---|
| constructor text_mode | str, plain | plain/legacy formatting |
| constructor algo_dir | str or None, None | None uses <cwd>/results/search/quantum_walk_search |
| run num_vertices | int,8 | Built-in graph size; effective graph must be2–32 vertices |
| run marked | str/int/sequence, "2,5" | Nonempty unique zero-based in-range targets; strings comma-separated |
| run graph_type | str, complete | complete/cycle/path/star; ignored for custom graph |
| run graph_json | str, empty | Inline array or adjacency/edges object, not a path |
| run gamma | optional float,None | Positive finite; empty/None chooses spectral default |
| run evolution_time | optional float,None | Non-negative finite; empty/None performs exact scanning |
| run method | str,trotter | Forwarded to installed simulator; auxiliary limitations above |
| run target_error | float,1e-4 | Positive finite simulator setting, not a guaranteed achieved bound |
| run simulation_steps | int,8 | Positive; passed only for trotter/qdrift; simulator may adjust actual steps |
| run shots | int,128 | Positive vertex sample count |
| run seed | int,7 | Vertex-sampling seed; NumPy RNG requires a non-negative seed |
| run backend | str,numpy | Simulator construction and final execute receive it |
| run device | str,cpu | Depends on backend; numpy only CPU |
| run dtype | np.complex128 | NumPy-compatible execution precision |

graph_json accepts an adjacency array, {"adjacency":[...]}, or
{"num_vertices":N,"edges":[[u,v],[u,v,weight]]}. Repeated edges accumulate.
Its marked/name fields are not consumed by run; provide marked separately.
For file-based input, independently read the task's JSON file, construct the accepted
graph_json value, and pass marked separately; do not call a bundled file loader.
The original UI regex excluded10–19 even though the Python implementation accepts them;
UI metadata does not override the public Python API's validation.

## Return Fields

| Public field | Type/shape | Meaning and limits |
|---|---|---|
| status | str | ok after successful completion; not a search/error-threshold verdict |
| circuit_path, circuit | str,Circuit | Saved measured overview SVG and overview circuit |
| plot | list of format/filename objects | TXT metadata; filename is not an absolute path |
| graph_name, num_vertices, hilbert_dimension | str,int,int | Effective graph label, valid size N and padded dimension D |
| marked_vertices, gamma, evolution_time | list[int],float,float | Sorted targets and resolved settings |
| time_selection, exact_scan_probability | str,float or None | Spectral selection description/scan probability, or explicit time |
| exact_probability_at_time, success_probability | float | Exact and simulated total marked probabilities |
| simulation_probability_error | float | abs(simulated-exact); not a norm error guarantee |
| measured_marked_count, sampled_success_probability, shots | int,float,int | Marked sample total, ratio and requested count |
| counts | dict[str,int] | Binary vertex indices, width log2(D); frequency descending, ties vertex ascending |
| probabilities, padded_probability | list[float] length N,float | Valid-vertex probabilities and sum on D-N padding states |
| method, target_error, simulation_steps | str,float,int or None | Requested settings; steps None for methods other than trotter/qdrift |
| circuit_summary | dict | Evolution num_qubits, size, depth and gate_counts |
| artifacts | dict[str,str] | Optional research exports; standard run normally leaves it empty |
| Elapsed time (s) | float | solve wall time, excludes final overview construction and SVG/TXT exports |

The standard run does not return a statevector. solve_graph_search returns a
QuantumWalkResult holding final_state and simulation, excluded by to_dict.
run_search retains (success_probability, final_state, simulation).

## Implementation Architecture

The same file contains all these implementations:

- validate_adjacency, adjacency_from_edges, built_in_graph and _marked_vertices
  convert/validate graph and marked inputs. np.asarray does not imply a fresh copy;
  validated adjacency is only read, while Hamiltonian padding is separate.
- default_coupling uses1/N for the complete graph or reciprocal spectral radius.
  graph_search_hamiltonian pads, adds marked diagonal energies and applies -gamma.
- _uniform_graph_state constructs valid-vertex amplitudes and zero padding.
- _exact_probability_series uses eigh; estimate_search_time performs129 coarse
  and65 fine evaluations with earliest np.argmax tie selection.
- solve_graph_search resolves settings, calls hamiltonian_simulation, executes
  from initial.copy(), computes exact/simulated probabilities, samples and counts.
- QuantumWalkResult separates public serializable evidence from retained state/simulator.
- BaseAlgorithm formats logs/reports. QuantumWalkSearchAlgorithm.run records input,
  invokes solve, records output, creates _build_overview_circuit and saves SVG/TXT.
- _write_visuals describes optional research artifacts; the public API's standard
  result export remains SVG/TXT, and no local data file is required by default.

## Understanding the Key Quantum Components

The vertex register represents integers v in [0,D), with q=log2(D) qubits.
Counts use conventional width-q binary representations of those integer indices;
do not reverse them to the q0-first strings used by other algorithms.
Only the first N amplitudes form the initial uniform graph state. Padding starts
at zero and is separately observed in padded_probability.

The marked oracle is an energy term -sum(|w><w|), not a Grover phase-oracle iteration.
The graph hopping Hamiltonian is -gamma*A. Together they generate exp(-iHt).
A Householder matrix I-2uu† prepares |s_G> from |0>, with
u=(|0>-|s_G>)/norm(|0>-|s_G>). The overview circuit appends this preparation,
the simulator circuit and computational-basis measurements.

Actual numerical evidence comes from executing the evolution circuit from an
explicit prepared initial state and sampling its probabilities with NumPy.
Overview measurements illustrate the full search; they do not generate the stored counts.
The exact spectral calculation is a classical reference and also chooses automatic time.
Global-phase changes can leave probabilities invariant but still alter the state/circuit;
do not silently change them when preserving source behavior.

## Theory-to-Code Mapping

| Mathematical object/step | Bundled source symbol | Convention/check |
|---|---|---|
| A and M | validate_adjacency, _normalise_marked | Original allclose rtol/atol; marked sorted, unique and in-range |
| D=2^ceil(log2(N)) | _next_power_of_two | Padding aligns valid vertices with first N basis indices |
| gamma | default_coupling | Complete graph1/N; otherwise reciprocal spectral radius, fallback1 |
| H=-gamma*A-P_M | graph_search_hamiltonian | Add1/gamma on padded marked diagonal, then multiply entire matrix by -gamma |
| Uniform graph state | _uniform_graph_state | Complex amplitudes1/sqrt(N), zero padding |
| P_M(t) | _exact_probability_series | eigh, conjugate projection and exp(-i lambda t), marked probability sum |
| selected time | estimate_search_time | 129-point horizon scan followed by65-point local scan |
| U_H(t) | hamiltonian_simulation, simulation.circuit.execute | Three execution settings forwarded; initial copy protected |
| vertex samples | default_rng(seed).choice | Normalize probability vector before choice; retain original call order |
| counts | bincount, flatnonzero, stable sorted | Ascending values before stable frequency-descending ordering |
| measured evidence | measured_marked_count, sampled_success_probability | Integer marked histogram sum divided by shots |
| overview/export | _build_overview_circuit, BaseAlgorithm | Preparation/evolution/measurement SVG and TXT |

## Mathematical Deep Dive

Let P_M=sum(w in M)|w><w|. On the valid graph space,
H=-gamma*A-P_M and |s_G>=sum(v=0..N-1)|v>/sqrt(N).
The implemented diagonal-addition form is algebraically identical:
H=-gamma*(A+P_M/gamma). Padding extends A and H by zero blocks to D dimensions.

For H=V diag(lambda) V†, evolution is
|psi(t)>=V[(V†|s_G>) elementwise exp(-i lambda*t)].
Its marked probability is P_M(t)=sum(w in M)|psi_w(t)|².
The coarse horizon is2*pi*sqrt(N/|M|), sampled at129 points. The earliest
coarse maximum is refined between neighboring coarse points with65 samples;
the earliest fine maximum determines time. This is a finite exact-reference
scan, not a proof of a global optimum or a scalable quantum time-selection procedure.

For an unweighted complete graph A=J-I with gamma=1/N, symmetry reduces ideal
evolution to normalized uniform marked/unmarked states when0<|M|<N.
In that basis H has diagonal elements
-(|M|-1)/N-1 and -(N-|M|-1)/N, and off-diagonal
-sqrt(|M|(N-|M|))/N. The eigenvalue gap is2*sqrt(|M|/N).
Ideal marked probability is
|M|/N+(1-|M|/N)*sin²(sqrt(|M|/N)*t);
the first ideal maximum occurs at(pi/2)*sqrt(N/|M|).
The finite scan and approximate simulator do not guarantee attaining that maximum.
If every vertex is marked, ideal marked probability is1 at every time.

The sampled marked ratio is a finite-shot estimate; its binomial variance is
P_M*(1-P_M)/shots for fixed circuit probabilities. The seed affects sampling,
not the exact reference, and does not ensure deterministic stochastic circuits.

Dense storage is O(D²); eigendecomposition is O(D³), and evaluating T time
points requires O(D²*T) matrix work and O(D*T) state storage. Counts use an
O(D) histogram while samples remain O(shots). q logarithmic vertex qubits
do not imply cheap classical simulation or account for auxiliary simulator registers.
The selected histogram change has static cost reasoning, not measured performance evidence.

## Hands-On Example

This all-marked four-vertex check differs from the default search example: it
checks indexing, sample totals and the distinction between process completion
and approximation quality. It uses nonzero explicit time to avoid zero-step edges.

Generate the following standalone task code in the user's task directory. It calls the installed public API with four vertices, all vertices marked, explicit time, and fixed sampling settings. Execute it only when execution, debugging, or verification is requested:

```python
import numpy as np
from unitarylab_algorithms import QuantumWalkSearchAlgorithm

algorithm = QuantumWalkSearchAlgorithm(
    algo_dir="results/quantum_walk_search_all_marked"
)
result = algorithm.run(
    num_vertices=4, marked="0,1,2,3", graph_type="complete",
    evolution_time=1.0, method="trotter", simulation_steps=8,
    shots=16, seed=7, backend="numpy", device="cpu", dtype=np.complex128,
)
assert result["status"] == "ok"
assert result["num_vertices"] == 4 and result["hilbert_dimension"] == 4
assert sum(result["counts"].values()) == 16
assert result["measured_marked_count"] == 16
assert result["sampled_success_probability"] == 1.0
assert result["simulation_probability_error"] == abs(
    result["success_probability"] - result["exact_probability_at_time"]
)
print("Normalization residual:",
      abs(sum(result["probabilities"]) + result["padded_probability"] - 1.0))
print("Ideal all-marked reference residual:",
      abs(result["exact_probability_at_time"] - 1.0))
print(result["circuit_path"], result["plot"])
```

These exact integer/shape/error-definition checks follow the returned contract.
Ideal all-marked probability is1; floating residuals are reported, with no
new acceptance tolerance substituted for missing source tolerances.
target_error is not an assertion threshold for simulation_probability_error.
No result, residual or assertion above has been observed by the agent.

For padding, next use N=3 with valid marked indices and compare three probabilities
plus padded_probability against the exact reference. For illegal inputs, duplicate/out-of-range
targets, asymmetric adjacency, self-loops and non-finite values should raise ValueError.

## Minimal Manual Implementation

This complete small example builds the same complete-graph continuous-time variant,
including the original coarse/fine time selection, Hamiltonian simulation,
exact reference, random sampling and selected counting order. It imports only
mathematical and public simulator components, no packaged algorithm class or leaf script.
For teaching it omits logs, overview drawing and report export; its return is a plain
evidence dictionary rather than BaseAlgorithm's artifact envelope.

```python
import math
import warnings
import numpy as np
from unitarylab.library import hamiltonian_simulation

def manual_search(n=8, marked=(2, 5), steps=8, shots=128, seed=7):
    if not 2 <= n <= 32 or shots < 1 or steps < 1:
        raise ValueError("invalid graph size, shots or steps")
    marked = tuple(sorted(map(int, marked)))
    if not marked or len(set(marked)) != len(marked):
        raise ValueError("marked must be nonempty and unique")
    if any(v < 0 or v >= n for v in marked):
        raise ValueError("marked is outside graph")
    adjacency = np.ones((n, n)) - np.eye(n)
    d = 1 << (n - 1).bit_length()
    gamma = 1.0 / n
    padded = np.zeros((d, d), dtype=float)
    padded[:n, :n] = adjacency
    padded[marked, marked] += 1.0 / gamma
    hamiltonian = -gamma * padded
    initial = np.pad(np.full(n, 1.0 / math.sqrt(n), dtype=complex), (0, d - n))

    def exact_series(times):
        values, vectors = np.linalg.eigh(hamiltonian)
        states = vectors @ (
            (vectors.conj().T @ initial)[:, None]
            * np.exp(-1j * np.outer(values, times))
        )
        return np.sum(np.abs(states[list(marked)]) ** 2, axis=0).real

    horizon = 2.0 * math.pi * math.sqrt(n / len(marked))
    coarse = np.linspace(0.0, horizon, 129)
    coarse_prob = exact_series(coarse)
    center = float(coarse[int(np.argmax(coarse_prob))])
    spacing = float(coarse[1])
    fine_times = np.linspace(max(0.0, center - spacing),
                             min(horizon, center + spacing), 65)
    fine_prob = exact_series(fine_times)
    best = int(np.argmax(fine_prob))
    time = float(fine_times[best])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Casting complex values to real discards the imaginary part",
            category=np.exceptions.ComplexWarning,
        )
        simulation = hamiltonian_simulation(
            hamiltonian, time, method="trotter", target_error=1e-4,
            steps=steps, backend="numpy", device="cpu", dtype=np.complex128,
        )
    state = np.asarray(simulation.circuit.execute(
        initial_state=initial.copy(), backend="numpy",
        device="cpu", dtype=np.complex128,
    ).state)
    probabilities = np.abs(state) ** 2
    success = float(np.sum(probabilities[list(marked)]))
    exact = float(exact_series(np.asarray([time]))[0])
    samples = np.random.default_rng(seed).choice(
        d, size=shots, p=probabilities / probabilities.sum()
    )
    histogram = np.bincount(samples, minlength=d)
    values = np.flatnonzero(histogram)
    frequencies = histogram[values]
    width = int(math.log2(d))
    counts = {
        format(int(v), f"0{width}b"): int(c)
        for v, c in sorted(zip(values, frequencies),
                           key=lambda item: int(item[1]), reverse=True)
    }
    marked_count = sum(int(histogram[v]) for v in marked)
    return {
        "evolution_time": time, "exact_scan_probability": float(fine_prob[best]),
        "success_probability": success, "exact_probability_at_time": exact,
        "simulation_probability_error": abs(success - exact),
        "counts": counts, "measured_marked_count": marked_count,
        "sampled_success_probability": marked_count / shots,
        "probabilities": [float(v) for v in probabilities[:n]],
        "padded_probability": float(np.sum(probabilities[n:])),
        "final_state": state, "circuit": simulation.circuit,
    }

# Explicit user execution would call: evidence = manual_search()
```

The manual example only covers the complete graph with automatic coupling/time
and trotter NumPy CPU complex128. For custom graphs and other methods, generate
calls to the installed public API with its documented parameters, or read the
fuller source to independently reconstruct the requested variant. Never invoke
the bundled implementation. N and marked can vary within the documented small range.
Repeated spectral decompositions and initial.copy protection intentionally remain.

## Debugging Tips

- Missing NumPy/UnitaryLab or incompatible APIs: report the missing dependency/version;
  do not repair by cloning another checkout, changing sys.path or downloading core code.
- No result has been produced: generating or reading code does not execute it. If execution is requested, run the generated standalone task code that calls the installed API; never run the bundled scripts. Inspect returned probabilities and counts separately from process completion.
- Invalid graph: check square shape, finite values, original allclose symmetry/zero-diagonal
  tolerance and non-negative edge weights. The edge-list constructor forbids self-loops
  and sums duplicate edges.
- Targets mismatch: zero-based vertex integers; inline graph_json does not supply marked.
  For N=2, replace the default2,5 with valid targets.
- Shape mismatch for Taylor/other methods: auxiliary-register limitations are preserved;
  do not append zeros or postselect silently, because that changes the state contract.
- Dtype/device failures: no fallback; GPU support is hardware/backend-dependent,
  including restrictions on complex128. Exact spectral work remains in NumPy.
- Probability differs from exact: distinguish scan discretization, simulation approximation
  and finite-shot sampling. Do not treat target_error or status as proof of accuracy.
- Counts ties/bit order: preserve stable frequency sorting and integer-index binary labels.
  Padded counts represent indices N..D-1, not additional graph vertices.
- Exports missing: run should create SVG/TXT; inspect algo_dir, plotting dependencies
  and write permissions. Full research PNGs require explicit research options.
- Report planned checks and actual observed checks separately. No checks were executed here.

## Reference Implementation

Reference-only source files:

- [scripts/algorithm.py](scripts/algorithm.py): reference-only example of input/output usage and result reporting.
- [scripts/quantum_walk_search_implementation.py](scripts/quantum_walk_search_implementation.py): Hamiltonian construction, evolution, sampling, embedded base, and result/export logic.

Read the relevant source only when implementation details or conventions are unclear. Do not run these files, import their modules, call their entrypoints, load them dynamically, or use them as runtime dependencies. Generate all task code and task inputs outside the skill package.

For API use, generate standalone task code against the installed `QuantumWalkSearchAlgorithm`. For independent reconstruction, modification, or porting, use the algorithm description, parameter/result tables, theory-to-code mapping, and manual implementation to generate independent UnitaryLab task code. Run and verify only generated code when the requested scope requires execution. Preserve matrix construction, floating-point order, vertex indexing, and random-call order unless an algorithm change is separately requested.

When updating this reference, keep parameter defaults, return fields, mathematical mapping, and top-level routing aligned with the reference source. Source examples and provenance alone are not execution evidence.
