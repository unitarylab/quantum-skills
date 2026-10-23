---
name: quantum-skills
description: Use when explaining quantum-computing concepts and algorithms, inspecting algorithm implementations, or constructing, simulating, debugging, or verifying quantum circuits with UnitaryLab, Qiskit, or PennyLane.
---

# Quantum Skills


## Scope

This skill combines two kinds of reference material:

- **Algorithms:** Principles, assumptions, documented algorithm parameters and return values, and concrete implementation details. Some references cover classical baselines or supporting routines; inclusion does not imply a quantum speedup.
- **Simulators and SDKs:** Circuit construction, public APIs, supported execution modes, and result conventions for UnitaryLab, Qiskit, and PennyLane.

Use algorithm references for explanations and algorithm-specific behavior. Add SDK references when the task needs concrete circuit operations or simulator behavior. SDK documentation may contain algorithmic functions; distinguish those interfaces from the separate `unitarylab_algorithms` wrappers.

## Context Loading Rule

Use the minimum reference set needed for the requested output. A single-algorithm task usually needs:

```text
SKILL.md
  + one target algorithm reference
  + simulator references only when framework-specific behavior is needed
```

This is a starting point, not a file-count limit. Read additional algorithm or SDK references only when the request spans them or when assumptions, signatures, return fields, conventions, or implementation behavior remain uncertain. Do not load the entire library by default.

## Resource Roles

- `references/` contains theory, interface documentation, and implementation notes. Distinguish general theory, the documented public API, and educational reconstructions.
- Every file under `scripts/` is reference-only. Read the relevant example or implementation when needed, but never run, import, dynamically execute, or use it as a runtime dependency. A generated API example is not independent evidence that the API exists; an educational implementation does not define the installed package's contract.
- Generate standalone task code for execution. Installed-package public APIs may be called when execution is authorized.

## Sources and Evidence

Use the source and version identified in the relevant reference. For UnitaryLab algorithm wrappers, use the `unitarylab_algorithms` package source; for the UnitaryLab SDK, use the [PyPI release](https://pypi.org/project/unitarylab/). Qiskit and PennyLane references identify their official releases and documentation. Confirm version-specific interfaces against the requested or installed package version.

For version-sensitive code, establish the requested or installed version and check the relevant import, signature, and return contract against its official documentation or source. Static inspection of package metadata/source does not require importing the package. If the available environment or source cannot establish a detail, state the gap and do not invent an interface.

When available sources do not cover a documented module in the target library, ask for the relevant source or version before changing its documented interface or support status.

When a wrapper delegates to an SDK function, check the forwarded arguments and delegated return contract as well as the wrapper signature. If they disagree, report the incompatibility; do not silently change the wrapper call or describe it as verified.

Complexity, convergence, query-count, and success-probability statements describe theory under their stated assumptions. They do not establish classical simulator speed, end-to-end application advantage, or a guarantee of the documented implementation. Distinguish theoretical predictions, static source findings, and observed execution results. Public `status` fields have the meaning defined by each API and do not universally certify mathematical correctness or convergence.


## Required Workflow

Select the task route first, then the required execution depth. Apply only the steps needed for the requested output.

### Step 1: Identify the Requested Task

| Requested output | Reference route |
|---|---|
| Concepts or algorithm principles | Relevant algorithm reference; no SDK reference unless simulator behavior is part of the question |
| Algorithm parameters or API calls | The selected API's algorithm reference and versioned source |
| Understanding or reproducing an implementation | Algorithm reference, then the relevant implementation and SDK details as needed |
| Simulator introduction, circuit construction, or result queries | Selected SDK reference; no algorithm-wrapper selection for an ordinary circuit |
| Algorithm applicability or product decision | Algorithm assumptions, actual problem structure, and implementation/resource limits |
| Problem-specific variant or combination | Related algorithm and SDK references selected through the problem-driven guidance in Step 3 |

For applicability questions, distinguish the ideal computational model from data access, oracle/state preparation, output requirements, classical alternatives, and hardware or maintenance costs when relevant. Give conditional conclusions when the user's workload is not specified.

### Select the Execution Level

#### Lightweight Mode

Use this mode for explanations, static implementation analysis, code generation, setup instructions, or verification instructions that do not require actual execution. Stop when the requested output is complete.


#### Full Verification Mode

Use this mode when the user requests execution, runtime debugging, or actual correctness evidence. Continue through the applicable execution, diagnosis, and reporting steps within the user's authorization. A request to describe verification steps is not by itself a request to run them.

### Step 2: Use UnitaryLab Unless the User Specifies Another Simulator

For tasks that require a concrete implementation framework, use UnitaryLab unless the user explicitly chooses another SDK. Use the selected framework's native APIs. Do not switch because another framework is more convenient or because an algorithm reference includes another framework's example. If a required capability is unavailable, explain the limitation and let the user choose an alternative.

Conceptual explanations, applicability analysis, and SDK comparisons do not require selecting a simulator. Simulator-specific explanations still use the matching SDK reference. Do not install or execute a simulator for an explanation-only request.

#### Simulator Selection and Guides

The typical uses below help compare capabilities. Read the selected guide for Python compatibility and dependencies. Install only the packages needed by the selected implementation path, and only when environment changes are authorized.

| Simulator and guide | Typical uses |
|---|---|
| [UnitaryLab](references/simulator-unitarylab.md) (default) | Quantum fundamentals, small to medium algorithm demos, circuit programming, PDE and Schrodingerization workflows. For focused tasks, follow Direct UnitaryLab Simulator Routing below. |
| [Qiskit](references/simulator-qiskit.md) | Ecosystem features, noise models, and IBM hardware workflows. |
| [PennyLane](references/simulator-pennylane.md) | Differentiable hybrid optimization, device/plugin interfaces, circuit transforms, expectation values, probabilities, samples, and circuit inspection. |

### Step 3: Locate the Minimum Required References

Follow the `Context Loading Rule`. References are flat and use these filename prefixes:

```text
algorithm-*.md   Algorithm knowledge and API guidance
simulator-*.md   Simulator and backend guidance
```

Read the matching reference before giving a concrete import, parameter, return field, or framework-specific behavior. Use both algorithm and SDK references when the task needs both kinds of information.

#### Problem-Driven and Multi-Algorithm Tasks

For practical problems or complex tasks, establish the inputs, desired outputs, constraints, and success criteria; split the task into subproblems when useful. Read relevant algorithm references to compare applicability assumptions, data representations, outputs, and resource needs. Consult the selected SDK references for implementation details.

For combinations or variants, explain how the components connect, the rationale for changes, and which steps need further derivation or validation. Separate examples do not establish the correctness or speedup of a combination. When the references leave a necessary step unresolved, continue the supported analysis and identify the missing information or reasoning. Ask the user about unknown conditions or choices that materially affect the solution.

#### Direct UnitaryLab Simulator Routing

- Circuit construction, gate storage, execution, measurement, transforms, and OpenQASM/Python/CUDA-Q import or export → `references/simulator-unitarylab-references-circuitsbuild.md`
- Core signatures, result fields, gates, backends, and `unitarylab.library` → `references/simulator-unitarylab-references-api-reference.md`
- Circuit analysis and structure queries → `references/simulator-unitarylab-references-CircuitInfo.md`
- Density matrices, Kraus noise, and density CUDA → `references/simulator-unitarylab-references-density-matrix-and-noise.md`
- Native PyTorch QML models, layers, kernels, metrics, objectives, and execution → `references/simulator-unitarylab-references-qml.md`
- Classical Ising/QUBO heuristics through `unitarylab.library.qubocore` → `references/simulator-unitarylab-references-qubocore.md`
- Molecular electronic structure and DMRG/CVD chemistry → `references/simulator-unitarylab-references-chemistry.md`
- Local compilation, mapping, and hardware submission → `references/simulator-unitarylab-references-compilation-and-hardware.md`

For a focused task with a clear matching leaf reference, read that leaf directly without loading the overview first. Read `references/simulator-unitarylab.md` when the request spans multiple UnitaryLab features, requires backend selection, has no clear matching leaf reference, or the leaf reference does not provide enough information to complete the task reliably. Load additional references only as needed.

For a known algorithm, route directly to its leaf reference below.  Do not load every algorithm reference for a single-algorithm request.

#### Direct Algorithm Routing

Use the algorithm name to read the final leaf reference directly.

**Quantum Primitives**

- Amplitude Amplification → `references/algorithm-primitives-amplitude-amplification.md`
- Grover Search → `references/algorithm-primitives-grover.md`
- Amplitude Estimation → `references/algorithm-primitives-amplitude-estimation.md`
- Quantum Phase Estimation → `references/algorithm-primitives-quantum-phase-estimation.md`
- Hadamard Test → `references/algorithm-primitives-hadamard-test.md`
- Hadamard Transform → `references/algorithm-primitives-hadamard-transform.md`

**Quantum Search**

- Glued Trees → `references/algorithm-search-glued-trees.md`
- Hidden Shift → `references/algorithm-search-hidden-shift.md`
- Quantum Walk Search / Continuous-Time Quantum Walk Search → `references/algorithm-search-quantum-walk-search.md`
- Quantum Backtracking / Binary CNF and Prefix-Constraint Search → `references/algorithm-search-quantum-backtracking.md`

**Quantum Linear Systems**

- HHL → `references/algorithm-linear-systems-hhl.md`
- AQC Linear Solver → `references/algorithm-linear-systems-aqc.md`
- LCU Linear Solver → `references/algorithm-linear-systems-lcu.md`
- Quantum Fourier Transform → `references/algorithm-linear-systems-quantum-fourier-transform.md`
- QSP / Basic QSP → `references/algorithm-linear-systems-quantum-signal-processing.md`
- QSVT-QLSA → `references/algorithm-linear-systems-qsvt-qlsa.md`
- VQLS → `references/algorithm-linear-systems-vqls.md`

**Quantum Cryptography**

- Shor Factoring → `references/algorithm-cryptography-shor.md`
- Discrete Logarithm → `references/algorithm-cryptography-discretelog.md`
- Simon's Algorithm → `references/algorithm-cryptography-simon.md`

**Hamiltonian Simulation**

- Trotter-Suzuki → `references/algorithm-hamiltonian-simulation-trotter.md`
- QDrift → `references/algorithm-hamiltonian-simulation-qdrift.md`
- Cartan-Lax → `references/algorithm-hamiltonian-simulation-cartan.md`
- QSP-HS / QSP Hamiltonian Simulation → `references/algorithm-hamiltonian-simulation-qsp.md`
- Taylor-Series Hamiltonian Simulation → `references/algorithm-hamiltonian-simulation-taylor.md`


Route QSP or Basic QSP to the basic single-qubit Linear Systems reference. Route QSP-HS or QSP
Hamiltonian Simulation to the Hamiltonian Simulation QSP reference.

**Schrodingerization**

- Advection Schrodingerization → `references/algorithm-schrodingerization-advection-schrodingerization.md`
- 1D Heat Schrodingerization → `references/algorithm-schrodingerization-heat-1d-schrodingerization.md`
- 2D Heat Schrodingerization → `references/algorithm-schrodingerization-heat-2d-schrodingerization.md`

**Quantum Machine Learning**

- QAOA → `references/algorithm-quantum-machine-learning-qaoa.md`
- QCBM → `references/algorithm-quantum-machine-learning-qcbm.md`
- VQC → `references/algorithm-quantum-machine-learning-vqc.md`
- VQE → `references/algorithm-quantum-machine-learning-vqe.md`
- Fermi-Hubbard VQE → `references/algorithm-quantum-machine-learning-fermi-hubbard-vqe.md`
- CVQNN → `references/algorithm-quantum-machine-learning-cvqnn.md`
- 2D Transverse-Field Ising Dynamics / Ising → `references/algorithm-quantum-machine-learning-ising.md`

**Quantum Chemistry**

- Molecular DMRG → `references/algorithm-quantum-chemistry-molecular-dmrg.md`

**Quantum State Preparation**

- Möttönen State Preparation → `references/algorithm-state-preparation-mottonen.md`
- Sparse Superposition → `references/algorithm-state-preparation-superposition.md`
- MPS State Preparation → `references/algorithm-state-preparation-mps.md`
- Multiplexer State Preparation → `references/algorithm-state-preparation-multiplexer.md`
- Pauli-Word State Preparation → `references/algorithm-state-preparation-pauli.md`

**Eigensolvers**

- NumPyEigensolver → `references/algorithm-eigensolvers-numyeigensolver.md`
- NumPyMinimumEigensolver → `references/algorithm-eigensolvers-minimum-eigensolvers-numpy-minimum-eigensolver.md`
- Variational Quantum Deflation → `references/algorithm-eigensolvers-vqd.md`

**Gradients**

- Finite Difference Gradient → `references/algorithm-gradients-finite-difference.md`
- Parameter-Shift Gradient → `references/algorithm-gradients-parameter-shift.md`
- Linear-Combination Gradient → `references/algorithm-gradients-linear-combination.md`
- SPSA Gradient → `references/algorithm-gradients-spsa.md`
- Reverse Gradient → `references/algorithm-gradients-reverse.md`
- Quantum Fisher Information → `references/algorithm-gradients-qfi.md`

**Quantum Error Correction**

- QEC / Quantum Error Correction / qLDPC / CSS / HGP → `references/algorithm-quantum-error-correction.md`


### Step 4: Choose the Algorithm Implementation Path When Needed

For algorithm API calls or independent algorithm implementations, choose the implementation path explicitly before writing code. Skip this step for explanation-only requests, ordinary circuit construction,
and simulator-use tasks that do not implement an algorithm.

Choose the path based on the user's request and the existing public APIs in the selected
framework; check `unitarylab-algorithms` by default.

For solutions with multiple algorithmic components, choose a path for each component.

- If an existing public API satisfies the requested behavior, parameters, outputs, and
  implementation constraints, use Path A. Do not also rebuild the algorithm through Path B.
- Use Path B when the user needs an independent implementation or no suitable API satisfies the request, **and** the relevant references and any required task-specific derivation establish the algorithmic steps, conventions, and expected outputs while the selected SDK provides the necessary operations.
- For variants or combinations, use the problem analysis in Step 3 to establish the steps and assumptions before selecting an implementation path.

Confirm the relevant API, signature, and return contract before deciding; do not infer support
from the algorithm name alone. Task labels such as execution, debugging, verification, or
modification do not by themselves determine the path.

Distinguish API capability gaps from environment problems. A missing package, failed import, or
version mismatch does not by itself justify Path B. Check the active environment, package path,
and compatible API version first; install or adjust dependencies only when the task requires it.
If the environment remains blocked, report the limitation rather than treating it as evidence
that the API lacks the required capability.

Generate standalone task code for the selected path: an installed-package API call for Path A,
or an independent implementation for Path B. Execute or verify it only when the task requires it.
The restriction on bundled `scripts/` does not prohibit calling installed-package public APIs.

#### Path A: Call an Existing Public API

Use the existing API when it fully satisfies the user's request under the selection rule above.

Identify algorithm
  -> Read algorithm reference
  -> Confirm API, parameters, and return fields
  -> Generate the smallest API call

#### Path B: Implement from Algorithm References

Reconstruct, adapt, or combine algorithms using the established steps and the selected SDK's public APIs. Read relevant source or educational implementations only as needed. When reproducing an existing implementation, preserve its documented execution order, state conventions, and output semantics. For an adaptation or combination, identify the changes and their rationale, and keep its behavior and claims distinct from those of existing wrappers.

Read related algorithm references
  -> Inspect the matching implementation when needed
  -> Establish algorithmic steps and component interfaces
  -> Implement with the selected SDK
  -> Do not import the implementation script into user code


### Step 5: Generate the Smallest Appropriate Solution

Generate the requested explanation, API call, standalone implementation, or example using the selected framework and implementation path. Include only the task-relevant assumptions, interface constraints, and limitations. In Lightweight Mode, stop here when the requested output is complete.

### Step 6: Execute Only When Needed

Install or execute only within the user's requested scope and permissions. Setup commands written in an answer are instructions, not actions already performed. Follow the selected SDK guide for installation; an algorithm-wrapper call may also require `unitarylab-algorithms`, while a direct SDK circuit does not require that package merely because this skill contains algorithm references.

When execution is requested, use the smallest standalone task code that exercises the relevant behavior. Report a task-appropriate signal such as a statevector, measurement probabilities, a residual, or a classical/theoretical comparison. Keep exact simulator distributions distinct from finite-shot counts. Real-hardware submission additionally requires explicit authorization and confirmed provider, target, credentials, shots, quota, and possible cost.

### Step 7: Diagnose Failures If Present

If execution or verification fails, enter `Error Handling` before making broad code changes. Classify
the failure, reproduce the smallest case, read the relevant reference or implementation script when
needed, test one causal hypothesis at a time, and verify the correction.

### Step 8: Complete and Report

Report the requested result and only the relevant framework/path, source version, evidence, limitations, and unresolved issues. Mark unexecuted code and theoretical expected outputs as such; do not present them as observed results. For runtime failures, report the error, causal hypothesis, change, and observed verification outcome.

## Error Handling

Classify errors before changing code:

| Symptom | First check |
|---|---|
| Import failure | Active environment and package path |
| Missing or unexpected argument | Current API reference and documented signature |
| State dimension error | Statevector length `2**n` or density-matrix shape `(2**n, 2**n)`, plus the corresponding normalization/trace contract |
| Backend or device failure | Supported backend, device, dtype, and options |
| Large numerical error | Convention, qubit order, normalization, and reference formula |
| Circuit mismatch | Gate order, controls, targets, inverse operation, and decomposition |
| Missing result field | Simulator result contract and execution mode |

For authorized runtime debugging, reproduce the failure with the smallest relevant example before broad changes. For static review, inspect the same contracts without executing code and state what remains unverified.
