---
name: molecular-dmrg
description: "Guide molecular electronic-structure calculations with MolecularDMRGAlgorithm. Use for active-space DMRG and conversion of the resulting state into CVD circuits for energy measurement."
---

# Molecular DMRG

## How to Use This Skill

Use this skill when the user asks to explain, call, run, debug, validate, or modify `MolecularDMRGAlgorithm`.

For user-facing code, use the public package import and `run()` method:

```python
from unitarylab_algorithms import MolecularDMRGAlgorithm
```

The public algorithm always calls the complete workflow. It does not expose the underlying workflow's `target` argument, so it cannot stop after electronic structure, DMRG, or CVD. Treat `scripts/algorithm.py` as the public API example and `scripts/molecular_dmrg_implementation.py` as the complete source-guided circuit reconstruction. Both are reference-only; do not import from or depend on the skill directory at runtime.

When using this skill:

- **Explanation:** Distinguish the electronic-structure, DMRG, CVD, and measurement stages and state which reported energies belong to which stage.
- **Run or reuse:** Start with an explicit H2/STO-3G active space. Omitting `xyz_file` currently selects CH4 in the Python source.
- **Debugging:** Identify the failing stage before changing parameters. File resolution and dtype normalization happen before the workflow starts.
- **Modification:** Preserve the public signature, stage forwarding, energy definitions, artifact contract, and the source's distinction between internal `self.status="success"` and returned `status="ok"`.
- **Validation:** Check output structure and energy identities separately from DMRG, CVD, FCI, or sampling convergence.

## Overview

`MolecularDMRGAlgorithm` converts an XYZ molecular geometry into a state-preparation circuit and grouped measurement artifacts through four ordered stages:

1. PySCF/OpenFermion electronic structure, active-space selection, Jordan-Wigner transformation, and an optional active-space FCI reference.
2. A fixed-charge-sector molecular ground-state solve using physics-tenpy infrastructure.
3. SciPy L-BFGS-B classical variational disentanglement (CVD) and reconstruction of a UnitaryLab preparation circuit.
4. UnitaryLab exact Hamiltonian expectation, qubit-wise-commuting (QWC) grouping, optional finite-shot sampling, and OpenQASM 2.0 export.

The algorithm saves a circuit SVG and BaseAlgorithm text report in `algo_dir`. Stage data and QASM files are written under `algo_dir/output/`.

The source does not provide the legacy MP2, CISD, CCSD, full post-SCF FCI controls, Julia DMRG files, or Julia convergence artifacts described by older implementations.

## Prerequisites

- Python package `unitarylab-algorithms` with a UnitaryLab build that exports `unitarylab.library.chemistry.Molecule` and `MolecularDMRGWorkflow`.
- PySCF, OpenFermion, and `openfermionpyscf` for electronic structure.
- physics-tenpy (`tenpy`) for the molecular model and DMRG infrastructure.
- SciPy for active-space FCI eigensolving and CVD optimization.
- NetworkX for QWC graph coloring.
- NumPy and a UnitaryLab execution backend supported by the installed version.

Cost grows with the basis and actual active-space size. The source converts the DMRG result to a dense statevector for CVD and circuit comparison, so this workflow remains exponentially constrained even when DMRG itself uses an MPS representation.

## Reference Implementation Example

```python
from unitarylab_algorithms import MolecularDMRGAlgorithm

algorithm = MolecularDMRGAlgorithm(text_mode="plain")
result = algorithm.run(
    xyz_file="H2",
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    active_occupied_orbitals=1,
    active_virtual_orbitals=1,
    max_iterations=10,
    cvd_restarts=1,
    cvd_init_scale=0.5,
    chi0=10,
    nsweeps=10,
    maxdim=20,
    cutoff=1e-8,
    shots=0,
    seed=2077,
    backend="torch",
    device="cpu",
    dtype="complex128",
)

print(result["status"])                    # "ok"
print(result["Active space"])              # CAS(...), ... qubits
print(result["DMRG energy"])               # Hartree
print(result["UnitaryLab exact energy"])    # Hartree
print(result["Exact energy difference"])
print(result["circuit_path"])
print(result.get("plot", []))
```

The reduced iteration and bond-dimension values make this a small API check, not a convergence claim. For H2 with two selected spatial orbitals, the source's DMRG stage uses exact diagonalization within the target charge sector because TenPy two-site DMRG requires more than two sites.

## Core Parameters Explained

**Constructor `MolecularDMRGAlgorithm(text_mode, algo_dir)`:**

| Parameter | Type | Default | Source behavior |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Passed to `BaseAlgorithm` for text-report formatting. |
| `algo_dir` | `str` or `None` | `None` | If omitted, creates `results/quantum_chemistry/molecular_dmrg` beneath the current working directory. |

**`run(...)` parameters:**

| Parameter | Type | Default | Source behavior |
|---|---|---|---|
| `xyz_file` | path-like, `str`, or `None` | `None` | `None` resolves to bundled `CH4.xyz`; a bare name becomes `molecules/<name>.xyz`; an explicit path is resolved directly. |
| `basis` | `str` | `"sto-3g"` | Forwarded to `Molecule.from_xyz()` and PySCF. |
| `charge` | `int` | `0` | Used to compute the total electron count and forwarded to PySCF. |
| `multiplicity` | `int` | `1` | Must be a positive integer and compatible with electron count parity. |
| `active_occupied_orbitals` | `int` | `3` | Maximum occupied orbitals selected downward from the HOMO; forwarded as `active_occupied`. |
| `active_virtual_orbitals` | `int` | `3` | Maximum virtual orbitals selected upward from the LUMO; may be truncated when the basis provides fewer orbitals. |
| `max_iterations` | positive `int` | `100` | Forwarded to CVD as the L-BFGS-B `maxiter` for each local gate optimization. |
| `cvd_restarts` | positive `int` | `3` | Number of randomized L-BFGS-B attempts per local gate, in addition to the zero-parameter baseline. |
| `cvd_init_scale` | non-negative `float` | `0.5` | Standard deviation of randomized CVD gate initialization. |
| `chi0` | positive `int` | `20` | Initial MPS bond dimension for non-small DMRG cases. |
| `nsweeps` | positive `int` | `100` | Forwarded as the maximum DMRG sweep count. |
| `maxdim` | positive `int` | `100` | Maximum DMRG bond dimension. |
| `cutoff` | positive `float` | `1e-8` | DMRG singular-value cutoff; also used when estimating the small-system result's bond rank. |
| `shots` | non-negative `int` | `10000` | Shots per QWC group. Zero disables sampling but not exact expectation or QASM export. |
| `seed` | non-negative `int` or `None` | `2077` | Used by CVD and measurement. CVD replaces `None` with `2077`; measurement preserves `None`. |
| `backend` | `str` | `"torch"` | Forwarded to `measurement_QASM()` and then `Circuit.execute()`. |
| `device` | `str` | `"cpu"` | Forwarded to circuit execution. |
| `dtype` | NumPy-compatible dtype | `np.complex128` | Normalized with `np.dtype(dtype).type` before workflow construction. |

`active_occupied_orbitals` and `active_virtual_orbitals` are maxima, not a guaranteed CAS shape. The actual active orbitals determine `n_qubits = 2 * len(active_orbitals)` and are reported after PySCF orbital selection.

**Common misunderstandings:**

- The Python default geometry is CH4, even though `README_zh.md`, `README_en.md`, and `parameters.json` describe H2 as the UI/example default. Pass `xyz_file="H2"` explicitly for H2.
- A filename with a suffix is treated as a path relative to the current working directory, not as a bundled molecule lookup. The resolved file must exist and end in `.xyz`.
- `shots` applies to each non-identity QWC group, not to one global sampling call.
- `shots=0` returns `Sampled energy=None`; it still computes `UnitaryLab exact energy`, builds QWC groups, and writes every measurement QASM file.
- `max_iterations` controls each local L-BFGS-B optimization. It is not the DMRG sweep limit; `nsweeps` controls that limit.
- The source sets `self.status="success"` for the saved text report, while `_build_return_dict(True, ...)` exposes `result["status"] == "ok"`.

## Return Fields

| Key | Type | Source definition |
|---|---|---|
| `status` | `str` | `"ok"` because the wrapper calls `_build_return_dict(True, ...)`; exceptions abort instead of returning `"failed"`. |
| `circuit_path` | `str` | Saved preparation-circuit SVG path. |
| `plot` | `list[dict[str, str]]` | Base result text descriptor followed by QASM descriptors; QASM entries are normalized to `format="qasm"`. |
| `circuit` | `Circuit` | CVD preparation circuit returned by the workflow. |
| `HF energy` | numeric | PySCF Hartree-Fock energy. |
| `Active-space FCI energy` | numeric or `"skipped"` | FCI energy when `fci_dimension` is nonzero; otherwise the literal string `"skipped"`. |
| `Active space` | `str` | `CAS(<active_electrons>e,<actual_active_orbitals>o), <n_qubits> qubits`. |
| `DMRG energy` | numeric | `dmrg_result.dmrg_energy`. |
| `CVD layers` | `int` | `cvd_result.circuit_layers`; the wrapper does not expose a layer override. |
| `CVD infidelity` | `float` | `max(0.0, 1.0 - cvd_result.overlap_mod**2)`. |
| `UnitaryLab exact energy` | `float` | Sum of Pauli-term expectations evaluated on the preparation circuit. |
| `Exact energy difference` | numeric | `abs(UnitaryLab exact energy - DMRG energy)`. |
| `Sampled energy` | `float` or `None` | QWC sampled estimator, or `None` when `shots=0`. |
| `Measurement groups` | `int` | Number of QWC groups over non-identity Pauli terms. |
| `Circuit depth` | `int` | `circuit.analyze(show=False).depth()`. |
| `Circuit gates` | `int` | `circuit.analyze(show=False).size()`. |
| `QASM directory` | `str` | Absolute path to `algo_dir/output`. |

The saved text filename supplied by `save_txt()` is a basename. QASM filenames come from the measurement result as paths. Consumers should use each descriptor's `filename` instead of assuming every `plot` entry has the same path form.

## Implementation Architecture

`MolecularDMRGAlgorithm.run()` is a thin public wrapper around `MolecularDMRGWorkflow`. Its source-level flow is:

| Stage | Code action | Observable result |
|---|---|---|
| 1 — Setup | `_resolve_xyz_file()`, `np.dtype(dtype).type`, create `algo_dir/output`, and call `update_input()` | Resolved geometry, normalized dtype, and output directories. |
| 2 — Molecule | `Molecule.from_xyz(..., active_occupied=..., active_virtual=...)` | Validated geometry plus molecular/basis/charge/multiplicity and active-space request. |
| 3 — Workflow | Construct `MolecularDMRGWorkflow` with renamed DMRG/CVD parameters and call `.run()` with no `target` | All four stage-result objects: `electronic_structure`, `dmrg`, `cvd`, and `measurement`. |
| 4 — Contract checks | Reject any missing stage and reject a missing CVD circuit | The wrapper returns only after the full pipeline produces a circuit and measurement result. |
| 5 — Metrics | Analyze circuit, calculate infidelity and exact energy difference, and select numeric FCI or `"skipped"` | Public output fields documented above. |
| 6 — Export | `save_circuit()`, `save_txt()`, merge QASM paths, call `_build_return_dict(True, ...)`, and normalize QASM formats | SVG, text report, QASM descriptors, circuit, and `status="ok"`. |

**Parameter forwarding:** `chi0` → `dmrg_chi0`; `maxdim` → `dmrg_maxdim`; `nsweeps` → `dmrg_max_sweeps`; `cutoff` → `dmrg_cutoff`; `max_iterations` → `cvd_maxiter`; `cvd_restarts` and `cvd_init_scale` retain their names.

The wrapper does not forward `layers` or `renyi_indices`. The workflow therefore uses `layers=None`; CVD derives `ceil(n_qubits / 2) + 1` layers and an all-ones Rényi-index sequence.

**Data flow:** XYZ path → `Molecule` → `ElectronicStructureResult` → `DmrgResult` → `CvdResult.circuit` → `MeasurementResult` → BaseAlgorithm return dictionary.

## Understanding the Key Quantum Components

### 1. Electronic structure and active space

The source runs PySCF SCF, sorts spatial orbitals by energy, selects up to the requested occupied orbitals down from the HOMO and virtual orbitals up from the LUMO, freezes earlier occupied orbitals, and constructs an active-space fermionic Hamiltonian. OpenFermion normal-orders it and the implementation maps it to real Pauli terms with Jordan-Wigner, dropping real coefficients whose magnitude is at most `1e-10`.

The active electron count and alpha/beta populations are derived from charge, multiplicity, and frozen occupancy. If the estimated number-preserving FCI dimension exceeds `1e6`, the FCI calculation is skipped; this is why the public field can be the string `"skipped"`.

### 2. Molecular DMRG state

The DMRG stage builds a fixed `(N, 2*Sz)` charge sector from active alpha and beta electron counts. One- and two-spatial-orbital cases are solved by exact diagonalization inside that sector. Larger cases initialize a fixed-charge MPS and run TenPy `TwoSiteDMRGEngine` with `max_sweeps=nsweeps`, `chi_max=maxdim`, `svd_min=cutoff`, `min_sweeps=2`, and `max_E_err=1e-12`.

The final MPS is converted into a dense, spin-blocked, qubit-0-least-significant-bit statevector before CVD.

### 3. CVD preparation circuit

CVD normalizes the dense target state, chooses the source-derived default layer count, and optimizes brick-wall nearest-neighbor two-qubit disentanglers. Every local gate has a zero-parameter baseline plus `cvd_restarts` seeded L-BFGS-B candidates. The objective is the Rényi entropy across the corresponding bond; the public wrapper uses the default Rényi index 1 for every layer.

The optimized disentangling sequence is inverted into a UnitaryLab state-preparation circuit. `overlap_mod` is the clipped magnitude of the overlap between the DMRG target and circuit state; the wrapper reports squared-overlap infidelity.

### 4. QWC measurement and QASM

Non-identity Pauli terms are grouped by graph coloring under qubit-wise commutativity. The circuit is executed once to evaluate the exact Pauli expectations. For each QWC group, basis-change gates are appended and a measured OpenQASM 2.0 circuit is written. If `shots>0`, each group is sampled independently using child seeds derived from `seed`; identity terms contribute directly to the sampled energy.

## Theory-to-Code Mapping

| Concept | Source object or expression |
|---|---|
| Molecular geometry and basis | `Molecule.from_xyz(xyz_path, basis, charge, multiplicity, ...)` |
| Actual active space | `active_orbitals`, `active_electrons`, `n_qubits` in `ElectronicStructureResult` |
| Active-space Hamiltonian | Frozen-core `get_molecular_hamiltonian()` → normal-ordered fermion terms → Jordan-Wigner Pauli terms |
| Optional exact reference | Number- and spin-preserving FCI when estimated dimension is at most `1e6` |
| Fixed particle/spin sector | DMRG target charge `(n_active_alpha + n_active_beta, n_active_alpha - n_active_beta)` |
| Tensor-network solve | TenPy `TwoSiteDMRGEngine` for more than two spatial orbitals |
| Small-system solve | Exact sector diagonalization for one or two spatial orbitals |
| CVD depth | `default_layers(n_qubits) = ceil(n_qubits / 2) + 1` |
| CVD objective | Bond Rényi entropy minimized by L-BFGS-B for local two-qubit gates |
| Preparation error | `max(0, 1 - |<psi_DMRG|psi_circuit>|^2)` |
| Exact circuit energy | Sum of `coefficient * execution.expectation(Pauli)` over Hamiltonian terms |
| Measurement grouping | NetworkX greedy coloring with `strategy="saturation_largest_first"` |
| Sampled energy | Identity contribution plus QWC parity expectations when `shots>0` |
| Exported measurement circuits | `measurement_<group_id>_<basis>.qasm` under `algo_dir/output` |

## Mathematical Deep Dive

The active-space Hamiltonian is transformed into a real Pauli expansion

$$
H = \sum_j c_j P_j,
$$

and the exact circuit energy returned by the measurement stage is

$$
E_{\mathrm{circuit}} = \sum_j c_j\langle\psi_{\mathrm{circuit}}|P_j|\psi_{\mathrm{circuit}}\rangle.
$$

The wrapper reports

$$
\Delta E = |E_{\mathrm{circuit}} - E_{\mathrm{DMRG}}|
$$

and, for $s=|\langle\psi_{\mathrm{DMRG}}|\psi_{\mathrm{circuit}}\rangle|$, the CVD infidelity

$$
\max(0, 1-s^2).
$$

A small energy difference tests energy preservation by the reconstructed circuit, while small infidelity tests state overlap. Neither metric by itself proves DMRG convergence to the molecular ground state. For finite-shot execution, each Pauli expectation is estimated from parity in its compatible basis; QWC grouping lets terms share a measurement circuit, but `shots` is applied separately to every group.

## Hands-On Example

```python
import math
from pathlib import Path

from unitarylab_algorithms import MolecularDMRGAlgorithm

result = MolecularDMRGAlgorithm(text_mode="plain").run(
    xyz_file="H2",
    basis="sto-3g",
    active_occupied_orbitals=1,
    active_virtual_orbitals=1,
    max_iterations=10,
    cvd_restarts=1,
    chi0=10,
    nsweeps=10,
    maxdim=20,
    cutoff=1e-8,
    shots=0,
    seed=2077,
    backend="torch",
    device="cpu",
    dtype="complex128",
)

assert result["status"] == "ok"
assert result["circuit"] is not None
assert Path(result["circuit_path"]).is_file()
assert result["Sampled energy"] is None

dmrg_energy = float(result["DMRG energy"])
circuit_energy = float(result["UnitaryLab exact energy"])
assert math.isclose(
    float(result["Exact energy difference"]),
    abs(circuit_energy - dmrg_energy),
    rel_tol=1e-12,
    abs_tol=1e-12,
)

qasm_files = [
    item["filename"] for item in result["plot"] if item["format"] == "qasm"
]
assert len(qasm_files) == result["Measurement groups"]
print(result["Active space"], result["CVD infidelity"], qasm_files)
```

Do not impose a universal energy or infidelity tolerance from this smoke test. Report the molecule, basis, actual active space, DMRG controls, CVD controls, backend, dtype, seed, and shots whenever comparing runs.

## Non-Circuit Stages

Do not invent quantum circuits for stages that the source executes classically:

| Stage | Source output | Circuit involvement |
|---|---|---|
| PySCF/OpenFermion electronic structure | `ElectronicStructureResult`, including `pauli_terms`, active orbitals, electrons, qubits, HF energy, and optional FCI energy | None. SCF, active-space selection, normal ordering, Jordan-Wigner conversion, and FCI are classical preprocessing. |
| TenPy molecular solve | `DmrgResult`, including `dmrg_energy` and dense LSB-ordered `psi0` | None. Small active spaces use sector exact diagonalization; larger ones use MPS DMRG. |
| CVD parameter search | Optimized nine-parameter gate rows | The objective evaluation and L-BFGS-B search are classical; the optimized parameters define the preparation circuit. |
| QWC grouping and energy reduction | Measurement bases, counts, and energy estimator | Graph coloring and counts-to-energy reduction are classical; basis rotations and measurements are circuits. |

For a real molecular reconstruction, obtain the circuit inputs through the source stage functions:

```python
from unitarylab_chemistry.dmrg import dmrg
from unitarylab_chemistry.molecule import Molecule, electronic_structure

molecule = Molecule.from_xyz(
    "H2.xyz",
    basis="sto-3g",
    active_occupied=1,
    active_virtual=1,
)
electronic = electronic_structure(molecule)
dmrg_result = dmrg(electronic, chi0=10, maxdim=20, max_sweeps=10, cutoff=1e-8)

dmrg_state = dmrg_result.psi0
pauli_terms = electronic.pauli_terms
n_qubits = electronic.n_qubits
```

These are implementation-level module paths, not the stable top-level algorithm API. Use them only for reconstruction, debugging, or validation. Read `scripts/molecular_dmrg_implementation.py` when a task requires the complete source-guided circuit reconstruction, QWC grouping, exact/sample energy calculation, or a real H2 integration path.

## Minimal Manual Implementation

Electronic structure and DMRG are classical in this source; they produce the Hamiltonian and the
dense target state but do not construct a quantum circuit. The circuit begins at CVD. The following
minimal reconstruction accepts `DmrgResult.psi0` (or any normalized LSB-ordered target state),
optimizes the source's brick-wall two-qubit disentanglers, and builds the inverse preparation with
basic `H`, `RX`, `RY`, `RZ`, and `CX` gates. It does not call `MolecularDMRGAlgorithm`,
`MolecularDMRGWorkflow`, or `cvd_dmrg()`.

```python
import math
import numpy as np
from scipy.optimize import minimize
from unitarylab import Circuit

P = {
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]], complex),
    "Z": np.array([[1, 0], [0, -1]], complex),
}

def axis_matrix(v):
    v = np.asarray(v, float); angle = np.linalg.norm(v)
    if angle < 1e-12: return np.eye(2, dtype=complex)
    x, y, z = v / angle; s = np.sin(angle / 2)
    return np.array([[np.cos(angle/2)-1j*z*s, (-1j*x-y)*s],
                     [(y-1j*x)*s, np.cos(angle/2)+1j*z*s]])

def entangler_matrix(p):
    p = np.asarray(p, float)
    if p.shape != (9,):
        raise ValueError("an entangler requires exactly nine parameters")
    gate = np.kron(axis_matrix(p[3:6]), axis_matrix(p[:3]))
    identity = np.eye(4, dtype=complex)
    for k, axis in enumerate("XYZ"):
        gate = (np.cos(p[6+k])*identity
                - 1j*np.sin(p[6+k])*np.kron(P[axis], P[axis])) @ gate
    return gate

def apply_two_qubit(state, matrix, q0, q1, n):
    if not 0 <= q0 < q1 < n:
        raise ValueError("sites must satisfy 0 <= q0 < q1 < n")
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (4, 4):
        raise ValueError("two-qubit matrix must have shape (4, 4)")
    tensor = np.asarray(state).reshape([2] * n)
    local = [n - 1 - q1, n - 1 - q0]  # q0 is the LSB
    rest = [axis for axis in range(n) if axis not in local]
    order = local + rest
    moved = tensor.transpose(order).reshape(4, -1)
    moved = (matrix @ moved).reshape([2, 2] + [2] * (n - 2))
    return moved.transpose(np.argsort(order)).reshape(-1)

def entropy_across_bond(state, n, bond):
    if not 0 <= bond < n - 1:
        raise ValueError("bond must lie between adjacent qubits")
    matrix = state.reshape(2**(n-bond-1), 2**(bond+1))
    probabilities = np.linalg.svd(matrix, compute_uv=False) ** 2
    probabilities /= probabilities.sum()
    probabilities = probabilities[probabilities > np.finfo(float).eps]
    return float(-np.sum(probabilities * np.log(probabilities)))

def append_axis(circuit, qubit, v):
    v = np.asarray(v, float); angle = np.linalg.norm(v)
    if angle < 1e-12: return
    x, y, z = v / angle; s = np.sin(angle / 2)
    a = np.cos(angle / 2) - 1j*z*s; c = (y - 1j*x)*s
    circuit.rz(float(-np.angle(c) - np.angle(a)), qubit)
    circuit.ry(float(2*np.arctan2(abs(c), abs(a))), qubit)
    circuit.rz(float(np.angle(c) - np.angle(a)), qubit)

def append_pauli_pair(circuit, q0, q1, axis, angle):
    if axis == "X": circuit.h(q0); circuit.h(q1)
    elif axis == "Y": circuit.rx(np.pi/2, q0); circuit.rx(np.pi/2, q1)
    circuit.cx(q0, q1); circuit.rz(float(2*angle), q1); circuit.cx(q0, q1)
    if axis == "X": circuit.h(q0); circuit.h(q1)
    elif axis == "Y": circuit.rx(-np.pi/2, q0); circuit.rx(-np.pi/2, q1)

def append_inverse_entangler(circuit, q0, q1, p):
    for k in reversed(range(3)):
        append_pauli_pair(circuit, q0, q1, "XYZ"[k], -p[6+k])
    append_axis(circuit, q0, -p[:3]); append_axis(circuit, q1, -p[3:6])

def product_factors(state, n):
    remainder = np.asarray(state); high_to_low = []
    for _ in range(n - 1):
        left, values, right = np.linalg.svd(remainder.reshape(2, -1), False)
        high_to_low.append(left[:, 0]); remainder = values[0] * right[0]
    high_to_low.append(remainder / np.linalg.norm(remainder))
    return list(reversed(high_to_low))

def factor_rotation(factor):
    factor = np.asarray(factor, dtype=complex)
    norm = float(np.linalg.norm(factor))
    if factor.shape != (2,) or norm <= 0:
        raise ValueError("factor must be a nonzero two-component vector")
    a, b = factor / norm
    theta = 2*np.arctan2(abs(b), abs(a))
    if theta < 1e-12: return np.zeros(3)
    delta = np.angle(b) - (np.angle(a) if abs(a) > 1e-15 else 0.0)
    return np.array([-theta*np.sin(delta), theta*np.cos(delta), 0.0])

def manual_cvd_circuit(dmrg_state, layers=None, maxiter=100, restarts=3,
                       init_scale=0.5, seed=2077):
    target = np.asarray(dmrg_state, complex).reshape(-1)
    if target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("state must be nonempty and finite")
    n = int(round(np.log2(target.size)))
    if n < 1 or target.size != 2**n: raise ValueError("invalid state length")
    norm = float(np.linalg.norm(target))
    if norm <= 0: raise ValueError("state must have nonzero norm")
    target /= norm
    layers = math.ceil(n/2) + 1 if layers is None else int(layers)
    if layers < 1 or maxiter < 1 or restarts < 1 or init_scale < 0:
        raise ValueError("layers, maxiter, and restarts must be positive; init_scale must be nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    disentangled = target.copy(); rows = []
    for layer in range(layers):
        for start in (0, 1):
            group_input = disentangled.copy(); optimized = []
            for q0 in range(start, n - 1, 2):
                q1 = q0 + 1
                def objective(p):
                    candidate = apply_two_qubit(
                        group_input, entangler_matrix(p), q0, q1, n)
                    return entropy_across_bond(candidate, n, q0)
                best = np.zeros(9); best_value = objective(best)
                for restart in range(restarts):
                    initial = np.random.default_rng(
                        seed + 1009*layer + 37*q0 + restart).normal(0, init_scale, 9)
                    trial = minimize(objective, initial, method="L-BFGS-B",
                                     options={"maxiter": maxiter, "ftol": 1e-12})
                    if trial.fun < best_value: best, best_value = trial.x, trial.fun
                optimized.append((q0, q1, np.asarray(best)))
            for q0, q1, p in optimized:
                disentangled = apply_two_qubit(
                    disentangled, entangler_matrix(p), q0, q1, n)
                rows.append((q0, q1, p))

    circuit = Circuit(n, name="Manual DMRG reverse preparation")
    for qubit, factor in enumerate(product_factors(disentangled, n)):
        append_axis(circuit, qubit, factor_rotation(factor))
    for q0, q1, p in reversed(rows):
        append_inverse_entangler(circuit, q0, q1, p)
    prepared = np.asarray(circuit.execute(backend="numpy", dtype=np.complex128).state)
    overlap = float(np.clip(abs(np.vdot(target, prepared)), 0.0, 1.0))
    return circuit, overlap, max(0.0, 1.0 - overlap**2)

def qwc_measurement_circuit(preparation, basis):
    if len(basis) != preparation.get_num_qubits():
        raise ValueError("basis length must equal the circuit qubit count")
    circuit = preparation.copy()
    for q, axis in enumerate(basis):
        if axis == "X": circuit.h(q)
        elif axis == "Y": circuit.rz(-np.pi/2, q); circuit.h(q)
        elif axis not in {"I", "Z"}: raise ValueError(axis)
    qubits = list(range(len(basis))); circuit.measure(qubits, qubits)
    return circuit

# Replace this stand-in with dmrg_result.psi0 from the classical DMRG stage.
target = np.zeros(16, complex); target[3] = target[12] = 1/np.sqrt(2)
circuit, overlap, infidelity = manual_cvd_circuit(
    target, layers=1, maxiter=20, restarts=1)
measurement = qwc_measurement_circuit(circuit, ("X", "Z", "Y", "I"))
print(overlap, infidelity, circuit.analyze(show=False).depth())
print(measurement.to_qasm2())
```

The input state must follow the source convention: qubit 0 is the least-significant bit. QWC graph
coloring and sampled-energy aggregation are classical; only each group's basis rotations and
measurements are circuits. Use the public API example when the task needs PySCF/DMRG execution and
the complete artifact contract rather than circuit reconstruction.

## Delegated Workflow Integration

The following implementation reconstructs the orchestration performed by
`MolecularDMRGAlgorithm.run()` without importing that algorithm class. It calls the same
UnitaryLab chemistry primitives, preserves the source parameter forwarding and energy formulas,
checks all four stage results, saves the preparation circuit, and exposes the scientific outputs.

```python
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
from unitarylab.library.chemistry import Molecule, MolecularDMRGWorkflow


def manual_molecular_dmrg(
    output_directory,
    *,
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    active_occupied_orbitals=1,
    active_virtual_orbitals=1,
    max_iterations=10,
    cvd_restarts=1,
    cvd_init_scale=0.5,
    chi0=10,
    nsweeps=10,
    maxdim=20,
    cutoff=1e-8,
    shots=0,
    seed=2077,
    backend="torch",
    device="cpu",
    dtype="complex128",
):
    """Run the source-equivalent four-stage workflow for bundled H2."""
    output_directory = Path(output_directory).expanduser().resolve()
    workflow_directory = output_directory / "output"
    workflow_directory.mkdir(parents=True, exist_ok=True)
    resolved_dtype = np.dtype(dtype).type

    h2_resource = files(
        "unitarylab_algorithms.quantum_chemistry.molecular_dmrg"
    ).joinpath("molecules", "H2.xyz")
    with as_file(h2_resource) as xyz_path:
        molecule = Molecule.from_xyz(
            xyz_path,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            active_occupied=active_occupied_orbitals,
            active_virtual=active_virtual_orbitals,
        )
        workflow_result = MolecularDMRGWorkflow(
            molecule,
            shots=shots,
            dmrg_chi0=chi0,
            dmrg_maxdim=maxdim,
            dmrg_max_sweeps=nsweeps,
            dmrg_cutoff=cutoff,
            cvd_maxiter=max_iterations,
            cvd_restarts=cvd_restarts,
            cvd_init_scale=cvd_init_scale,
            workdir=workflow_directory,
            seed=seed,
            backend=backend,
            device=device,
            dtype=resolved_dtype,
        ).run()

    electronic = workflow_result.electronic_structure
    dmrg_result = workflow_result.dmrg
    cvd_result = workflow_result.cvd
    measurement = workflow_result.measurement
    if any(
        stage is None
        for stage in (electronic, dmrg_result, cvd_result, measurement)
    ):
        raise RuntimeError("the Molecular DMRG workflow did not complete all stages")
    if cvd_result.circuit is None:
        raise RuntimeError("the CVD stage did not return a preparation circuit")

    circuit = cvd_result.circuit
    circuit_info = circuit.analyze(show=False)
    circuit_path = output_directory / "molecular_dmrg_circuit.svg"
    circuit.draw(filename=str(circuit_path), title="Molecular DMRG Circuit")

    exact_difference = abs(
        measurement.exact_energy - dmrg_result.dmrg_energy
    )
    infidelity = max(0.0, 1.0 - cvd_result.overlap_mod**2)
    return {
        "status": "ok",
        "circuit_path": str(circuit_path),
        "plot": [
            {"format": "qasm", "filename": str(path)}
            for path in measurement.qasm_files
        ],
        "circuit": circuit,
        "HF energy": electronic.hf_energy,
        "Active-space FCI energy": (
            electronic.fci_energy if electronic.fci_dimension else "skipped"
        ),
        "Active space": (
            f"CAS({electronic.active_electrons}e,"
            f"{len(electronic.active_orbitals)}o), "
            f"{electronic.n_qubits} qubits"
        ),
        "DMRG energy": dmrg_result.dmrg_energy,
        "CVD layers": cvd_result.circuit_layers,
        "CVD infidelity": infidelity,
        "UnitaryLab exact energy": measurement.exact_energy,
        "Exact energy difference": exact_difference,
        "Sampled energy": measurement.sampled_energy,
        "Measurement groups": measurement.n_qwc_groups,
        "Circuit depth": circuit_info.depth(),
        "Circuit gates": circuit_info.size(),
        "QASM directory": str(workflow_directory),
    }


result = manual_molecular_dmrg("results/manual_molecular_dmrg")
print(result["Active space"])
print(result["DMRG energy"], result["UnitaryLab exact energy"])
print(result["CVD infidelity"], result["Exact energy difference"])
```

This manual version intentionally omits `BaseAlgorithm` logging and its formatted text report; those
are presentation-layer features rather than chemistry or circuit logic. It otherwise uses the same
delegated workflow, default terminal stage, parameter mapping, circuit metrics, energy definitions,
and QASM results as the source wrapper. The H2 resource is resolved with `importlib.resources` so the
example does not import a private algorithm module or depend on the current working directory.

## Debugging Tips

1. **Import fails at `unitarylab.library.chemistry`:** The installed UnitaryLab build is older than the API expected by this algorithm. Installing only `unitarylab-algorithms` does not guarantee that the delegated chemistry symbols exist.
2. **Geometry lookup fails:** A bare `H2` resolves in the bundled `molecules/` directory; `H2.xyz` resolves relative to the current working directory. Use an absolute path when the distinction matters.
3. **XYZ parsing fails:** The file must contain an integer atom count on line 1, a comment line, and at least the declared number of atom rows with symbol plus three numeric coordinates.
4. **Electron/spin validation fails:** Check atomic numbers, charge, and `multiplicity - 1`; electron count and spin must have compatible parity.
5. **Actual active space is smaller than requested:** The virtual-orbital selection is truncated to what the chosen basis supplies. Read the returned `Active space` instead of inferring qubits from the requested maxima.
6. **FCI is `"skipped"`:** The estimated number- and spin-preserving FCI dimension exceeded `1e6`. This does not mean the complete workflow stopped.
7. **DMRG does not converge:** First confirm that the case has more than two spatial orbitals; smaller cases use exact sector diagonalization. For larger cases, vary `nsweeps`, `maxdim`, and `cutoff` one at a time.
8. **Circuit energy differs from DMRG:** Inspect both `CVD infidelity` and `Exact energy difference`, then increase `max_iterations` or `cvd_restarts` while holding the molecular and DMRG inputs fixed.
9. **Sampled energy is `None`:** This is expected when `shots=0`. Set a positive integer to enable sampling and compare the sampled estimator with `UnitaryLab exact energy`.
10. **Backend or device execution fails:** Reproduce with `backend="torch"`, `device="cpu"`, and `dtype="complex128"`; then verify support before changing one execution option at a time.
