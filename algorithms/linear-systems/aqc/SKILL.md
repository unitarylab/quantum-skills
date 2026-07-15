---
name: aqc
description: Explains and demonstrates UnitaryLab's small-scale adiabatic quantum linear-system solver. Use it for AQC or QLSP questions, simulator examples, and work involving Householder state preparation, SVD block encoding, adiabatic schedules, post-selection, solution rescaling, or residual checks.
---

# Adiabatic Quantum Linear-System Solver (AQC)

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab Adiabatic Quantum Computing (AQC) linear-system solver.

The implementation generates a deterministic small real symmetric system $Ax=b$, normalizes it, prepares $|b\rangle$ with a Householder transform, block-encodes $A$ with an SVD construction, applies a discretized adiabatic schedule on `n` system qubits and five ancillas, post-selects the ancilla pattern, and reconstructs a classical solution vector. It is an educational dense-matrix/statevector solver, not a scalable fault-tolerant QLSP.

Use this skill when you need to:

- Demonstrate or explain the repository's adiabatic approach to a generated quantum linear-system problem.
- Inspect Householder state preparation, SVD block encoding, schedule parameters, five-ancilla layout, or little-endian post-selection.
- Diagnose normalization, condition-number, automatic-step, post-selection, reconstruction, residual, direction-error, fidelity, or scale-conversion issues.
- Modify or independently implement the solver while preserving either the documented source contract or an explicitly selected recommended extension contract.

When using this skill:

- **Explanation:** Explain the generated system, schedule, block encoding, register layout, post-selection, normalization/rescaling, and the distinction between a quantum state direction and a classical solution vector. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code that imports `AQCAlgorithm` from `unitarylab_algorithms`. Start with `n=1` or `n=2`; do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare condition number, selected `T`, register width, post-selection amplitude, residual, direction error, returned `"ok"` status, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the `run()` signature, deterministic seed, five-ancilla layout, normalized circuit problem, little-endian post-selection, trivial `kappa≈1` branch, reconstruction scale, and return fields unless the user explicitly requests the recommended extension.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/aqc_implementation.py` as reference-only material for troubleshooting, API comparison, and validation. The formal source at `unitarylab_algorithms/linear_algebra/aqc/algorithm.py` is authoritative.
- **Validation:** Validate with a small deterministic case and report residual, normalized-direction error, fidelity when computed, post-selection amplitude or probability, condition number, backend/device/dtype, dependency assumptions, and dense-simulation scale limits. Keep source-equivalent and recommended-extension results explicitly separated.

Keep two contracts separate throughout the response:

- **Source-equivalent behavior:** Preserve the current `run()` signature, generated inputs, normalized-problem `Classical Solution` in the normal branch, first-nonzero-component internal scale, direction comparison without global-phase alignment, and exact source return keys.
- **Recommended extension behavior:** Put quantum and classical vectors on the original generated-problem scale, align global phase for direction diagnostics, report fidelity and post-selection probability, use least-squares internal scaling, and update source code, callers, tests, and return fields deliberately rather than describing these additions as current behavior.

## Overview

1. Generate a real symmetric matrix with a diagonal shift of `10I` and a random right-hand side with NumPy seed `42`, dimension $N=2^n$.
2. Normalize $A$ by $\lVert A\rVert_2$ and $b$ by $\lVert b\rVert_2$; retain `a_scale` and `b_scale` so $A_\mathrm{orig}=a_\mathrm{scale}A$ and $b_\mathrm{orig}=b_\mathrm{scale}b$.
3. Before constructing the adiabatic circuit, handle `kappa≈1` with a classical-solve shortcut. This branch does not execute an AQC circuit and avoids the singular factor `kappa/(kappa-1)` in the schedule.
4. Otherwise build Householder $U_b$ and an SVD block encoding of normalized $A$.
5. Use `n` system qubits and five ancillas, initialize the system to $|b\rangle$, and apply `T` discrete steps. For `T=0`, the source chooses `ceil(10*kappa)` and rounds up to even.
6. In little-endian order, select ancilla `|10000⟩` (`anl[4]=1`, `anl[0:4]=0`). The current source returns only the post-selection amplitude.
7. The recommended implementation additionally returns post-selection probability, fidelity, and a phase-aligned direction error. The current source does not align global phase.

For a positive-definite Hermitian matrix, `kappa≈1` implies equal eigenvalues and therefore a scalar multiple of the identity (after normalization, `A≈I`). This implication must not be extended to a general matrix merely from a condition number near one.

With seed `42`, the generated matrix is positive definite for `n≤5`. At `n=6`, its minimum eigenvalue is approximately `-1.173`, its condition number is approximately `106.33`, and automatic selection gives about `1064` adiabatic steps. Although the source documents `n≤6`, practical examples should start with `n=1` or `n=2`.

## Prerequisites

- Linear systems, condition numbers, Hermitian/symmetric matrices, SVD, block encodings, and statevector post-selection.
- Python: `numpy`, `unitarylab.Circuit`, and `unitarylab.Register`.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import AQCAlgorithm

result = AQCAlgorithm(text_mode="plain").run(
    n=2, T=0, p=1.4, backend="torch", device="cpu", dtype=np.complex128
)
print(result["status"])
print(result["Quantum Solution (x)"])
print(result["Classical Solution"])
print(result["Residual Norm ||Ax-b||"])
print(result.get("Fidelity"))
```

## Core Parameters Explained

**Constructor `AQCAlgorithm(text_mode, algo_dir)`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Formatting mode passed to `BaseAlgorithm`. |
| `algo_dir` | `str` or `None` | `None` | Output directory. |

**`run(n, T, p, backend, device, dtype)` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` or `np.integer` | `2` | System-qubit count; `N=2**n`; reject `bool`; source targets `1 ≤ n ≤ 6`. |
| `T` | `int` or `np.integer` | `0` | Positive step count, or `0` for automatic even selection; reject `bool`. |
| `p` | finite real | `1.4` | Schedule exponent; reject `bool`, NaN, and infinity; require `p > 1`. |
| `backend` | `str` | `'torch'` | UnitaryLab backend. |
| `device` | `str` | `'cpu'` | Execution device. |
| `dtype` | | `np.complex128` | Complex dtype passed to `Circuit.execute`; manual code must pass this parameter too. |

The source generates `A` and `b` internally; `run()` does not accept user-supplied inputs.

## Source Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` in the packaged return dictionary. |
| `Quantum Solution (x)` | `np.ndarray` | Reconstructed solution; normal branch rescales it to the original generated system. |
| `Classical Solution` | `np.ndarray` | Normal branch: `np.linalg.solve(A,b)` on normalized `A,b`; trivial branch uses original-scale inputs. |
| `Residual Norm ||Ax-b||` | `float` | Residual evaluated against original-scale `A_orig,b_orig`. |
| `Error vs Classical (L2)` | `float` | Normalized-direction L2 error, without global-phase alignment in the current source. |
| `Internal Scale Factor` | `float` | Source's first nonzero-component scale estimate. |
| `Post-selection Amplitude` | `float` | Normal branch only: norm of the selected system vector. |
| `Simulation Time (s)` / `Elapsed Time (s)` | `float` | Timing fields; simulation time is absent in the trivial branch. |
| `circuit_path`, `plot`, `circuit` | source-defined | Circuit and exported-output descriptors. |

## Recommended Extension Fields

These fields describe the corrected implementation contract; they are not guaranteed to be returned by the current `algorithm.py`.

| Key | Type | Description |
|---|---|---|
| `Classical Solution` | `np.ndarray` | `np.linalg.solve(A_orig,b_orig)`, on the same original scale as the quantum solution. |
| `Phase-aligned Direction Error (L2)` | `float` | L2 error after aligning the global phase of normalized directions. Keep the source's `Error vs Classical (L2)` unchanged; repurposing that key would be a breaking semantic change. |
| `Fidelity` | `float` | `abs(q.conj() @ c)**2` for normalized quantum and classical directions. |
| `Post-selection Probability` | `float` | Square of the post-selection amplitude. |
| `Internal Scale Factor` | `complex` | Least-squares scale `c = <Av,b>/<Av,Av>`, rather than a single-component estimate. A near-zero denominator is an error, not justification for `c=1`. |
| `AQC Circuit Executed` | `bool` | `False` for the `kappa≈1` classical shortcut and `True` for the normal AQC branch. |

## Implementation Architecture

`run()` has four logged stages: (1) validation/preprocessing, deterministic generation and normalization; (2) dense circuit construction; (3) statevector simulation; and (4) post-selection, reconstruction, diagnostics, and exports. In the recommended implementation, test `kappa≈1` between stages 1 and 2 and return the classical solution directly; do not build or execute an AQC circuit for that shortcut.

Each `_add_adiabatic_step()` is an **11-stage logical sequence**, not “11 operations”: it includes grouped reflection, controlled rotation, block-encoding, mirrored uncomputation, and final phase-reflection subroutines. The schedule is

$$f(s)=\frac{\kappa}{\kappa-1}\left[1-\left(1+s(\kappa^{p-1}-1)\right)^{1/(1-p)}\right],\quad s=k/T,$$

with `theta = 2*arctan2(f, 1-f)`.

## Understanding the Key Quantum Components

### 1. Householder state preparation

`_uprep_b(b)` is source-equivalent mainly for normalized real vectors. The construction uses `sqrt((b[0]+1)/2)` and divides by this quantity, so it can be numerically unstable when `b[0]≈-1`; a robust implementation should use a phase-safe Householder construction or a fallback branch. The optional debug/test path may check `\lVert b\rVert≈1` and `U_b|0\rangle≈b`.

### 2. SVD block encoding

`_bloc_enc(A)` embeds singular values with `sqrt(I-sigma**2)`. The optional debug/test path may verify that `A` is square and `\lVert A\rVert_2≤1`, then check `B.conj().T @ B≈I` and `B[:N,:N]≈A`; these checks are potentially expensive and should not run on every normal execution.

### 3. Register layout and little-endian post-selection

```text
global qubits: 0 ... n-1 | n ... n+4
               system    | ancilla
```

Select the branch explicitly: set bits `n+0..n+3` to zero and bit `n+4` to one, or equivalently use the source slice `state_arr[2**(n+4):]` and then take exactly the first `N=2**n` amplitudes. Return the selected vector norm (amplitude) **and** its squared probability; do not conflate them.

### 4. Discrete adiabatic step

Repeat the 11-stage logical sequence for `k=1,...,T`. Preserve the mirrored schedule and both `U_b` reflection sequences when reproducing source behavior.

### 5. Classical reconstruction

Let `v` be the normalized post-selected direction. For the normalized problem, use the complex least-squares scale

$$c=\frac{(Av)^\dagger b}{(Av)^\dagger(Av)},\qquad x_\mathrm{problem}=cv.$$

Then restore original scale with `x_orig=(b_scale/a_scale)x_problem`. Preserve the source's unaligned normalized-direction diagnostic under `Error vs Classical (L2)`. For the recommended diagnostic, let `q` and `r` be normalized quantum and classical original-scale solutions, choose `phase = (r.conj()@q)/abs(r.conj()@q)` when nonzero, and report `norm(phase.conj()*q-r)` under `Phase-aligned Direction Error (L2)`; fidelity is `abs(r.conj()@q)**2`. Reusing the old key for this new calculation is a breaking semantic change.

## Theory-to-Code Mapping

| Theory concept | Source implementation |
|---|---|
| $|b\rangle$ preparation | `_uprep_b`; `qc.unitary(Ub_mat, target=sys_qubits[:])` |
| Block encoding | `_bloc_enc(A)` via SVD |
| Registers | `Register('sys', n_sys)`, `Register('anl', 5)` |
| Normalized problem | `A/a_scale`, `b/b_scale` |
| Auto steps | `ceil(10*kappa)`, rounded to even |
| Post-selection | `state_arr[2**(n+4):][:2**n]` |
| Source classical reference | `np.linalg.solve(A,b)` in normal branch |
| Recommended reference | `np.linalg.solve(A_orig,b_orig)` |
| Circuit exports | full circuit plus one-step slice |

## Mathematical Deep Dive

The SVD construction assumes $\lVert A\rVert_2≤1$ and has the normalized matrix as its upper-left block. The ideal evolution targets a system direction proportional to $A^{-1}|b\rangle$; finite `T`, block-encoding conventions, and post-selection affect its amplitude and direction. Keep the normalized circuit equations separate from the original system used for residuals. A low direction error does not imply a low residual if scaling is wrong, and a small `kappa` alone does not certify `A=I`.

## Hands-On Example

```python
import numpy as np
from unitarylab_algorithms import AQCAlgorithm

result = AQCAlgorithm().run(n=2, T=20, p=1.4, backend="torch", dtype=np.complex128)
print(f"Status: {result['status']}")
print(f"Amplitude: {result.get('Post-selection Amplitude', float('nan')):.3e}")
print(f"Probability: {result.get('Post-selection Probability', float('nan')):.3e}")
print(f"Residual: {result['Residual Norm ||Ax-b||']:.3e}")
print(f"Fidelity: {result.get('Fidelity', float('nan')):.6f}")
assert result['status'] == 'ok'
assert np.isfinite(result['Residual Norm ||Ax-b||'])
```

## Minimal Manual Implementation

Use `from unitarylab import Circuit, Register` consistently. The manual implementation should mirror the source's deterministic generation, normalization, five-ancilla circuit and 11-stage logical sequence, but apply the recommended post-processing: validate `n`, positive `T`, `p>1`, `N=2**n`, vector lengths, Householder preparation, block-encoding unitarity/upper-left block, least-squares scaling, original-scale classical solution, phase-aligned error, fidelity, amplitude, and probability. Pass `dtype` through:

```python
import numpy as np
from unitarylab import Circuit, Register

def prepare_b_unitary(b):
    b = np.asarray(b, dtype=complex).reshape(-1, 1)
    if len(b) == 0 or not np.all(np.isfinite(b)):
        raise ValueError("b must be a non-empty finite vector")
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-15:
        raise ValueError("b must be nonzero")
    if not np.isclose(norm_b, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("b must be normalized before Householder preparation")
    if abs(b[0, 0].imag) > 1e-12:
        raise ValueError("This source-equivalent Householder method expects a real vector")
    if abs(b[0, 0].real + 1.0) < 1e-12:
        raise RuntimeError("Householder construction is unstable when b[0] is near -1")
    v = np.zeros_like(b)
    v[0, 0] = np.sqrt((b[0, 0] + 1) / 2)
    for k in range(1, len(b)):
        v[k, 0] = b[k, 0] / (2 * v[0, 0])
    return 2 * (v @ v.conj().T) - np.eye(len(b))

def block_encode(A):
    U, s, Vh = np.linalg.svd(A)
    S = np.diag(s); R = np.diag(np.sqrt(np.clip(1 - s**2, 0, None)))
    M = np.block([[S, R], [R, -S]])
    I = np.eye(len(s)); Z = np.zeros_like(I)
    return np.block([[U, Z], [Z, I]]) @ M @ np.block([[Vh, Z], [Z, I]])

def add_adiabatic_step(qc, k, T, n, sys, anc, Ub, BlA, kappa, p, mcz):
    s = k / T
    f = kappa / (kappa - 1) * (1 - (1 + s * (kappa**(p - 1) - 1))**(1 / (1 - p)))
    theta = 2 * np.arctan2(f, 1 - f)

    # Stage 1: open the interference branch.
    qc.h(anc[2])

    # Stage 2: first U_b reflection sequence.
    qc.unitary(Ub.conj().T, target=sys[:])
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.append(
        mcz,
        target=list(range(n)) + [n + 4],
        control=[n + 1, n + 2],
        control_state=[1, 1],
    )
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.unitary(Ub, target=sys[:])

    # Stage 3: scheduled controlled rotation.
    qc.cz(control=anc[1], target=anc[3], control_state=0)
    qc.cry(theta, control=anc[1], target=anc[3], control_state=0)

    # Stage 4: controlled Hadamard.
    qc.ch(control=anc[1], target=anc[3], control_state=1)

    # Stage 5: controlled block encoding and its adjoint.
    qc.cz(target=anc[4], control=anc[3], control_state=0)
    qc.unitary(
        BlA,
        target=list(range(n + 1)),
        control=[n + 3, n + 4],
        control_state=[1, 1],
    )
    qc.unitary(
        BlA.conj().T,
        target=list(range(n + 1)),
        control=[n + 3, n + 4],
        control_state=[1, 0],
    )
    qc.cx(anc[3], anc[4])

    # Stage 6: switch the symmetric branch.
    qc.x(anc[1])

    # Stage 7: mirrored controlled Hadamard.
    qc.ch(control=anc[1], target=anc[3], control_state=1)

    # Stage 8: mirrored scheduled rotation.
    qc.cz(control=anc[1], target=anc[3], control_state=0)
    qc.cry(theta, control=anc[1], target=anc[3], control_state=0)

    # Stage 9: second U_b reflection sequence.
    qc.unitary(Ub.conj().T, target=sys[:])
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.append(
        mcz,
        target=list(range(n)) + [n + 4],
        control=[n + 1, n + 2],
        control_state=[1, 1],
    )
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.unitary(Ub, target=sys[:])

    # Stage 10: close the interference branch.
    qc.h(anc[2])

    # Stage 11: final reflection and global phase.
    qc.x(anc[3])
    qc.mcz(controls=[n, n + 2], target=anc[3], control_state=[0, 0])
    qc.x(anc[3])
    qc.gp(np.pi)

def solve_manual_aqc(n=2, T=0, p=1.4, backend="torch", device="cpu", dtype=np.complex128):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer, not bool")
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)):
        raise TypeError("T must be an integer, not bool")
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise TypeError("p must be a finite real number")
    n, T, p = int(n), int(T), float(p)
    if not 1 <= n <= 6: raise ValueError("n must be in [1, 6]")
    if T < 0: raise ValueError("T must be a non-negative integer")
    if not np.isfinite(p) or p <= 1: raise ValueError("p must be finite and greater than 1")
    np.random.seed(42); N = 2**n
    R = np.random.randn(N, N)
    A_orig = ((R + R.T) / 2 + 10 * np.eye(N)).astype(complex)
    b_orig = np.random.randn(N).astype(complex)
    a_scale = float(np.linalg.norm(A_orig, 2)); b_scale = float(np.linalg.norm(b_orig))
    A, b = A_orig / a_scale, b_orig / b_scale
    kappa = float(np.linalg.cond(A))
    classical = np.linalg.solve(A_orig, b_orig)
    if abs(kappa - 1.0) < 1e-12:
        return {"status": "ok", "Quantum Solution (x)": classical.copy(),
                "Classical Solution": classical,
                "Residual Norm ||Ax-b||": float(np.linalg.norm(A_orig @ classical - b_orig)),
                "Error vs Classical (L2)": 0.0,
                "Phase-aligned Direction Error (L2)": 0.0, "Fidelity": 1.0,
                "Internal Scale Factor": 1.0, "Adiabatic Steps (T)": 0,
                "AQC Circuit Executed": False, "circuit": None}
    T_val = int(np.ceil(10 * kappa)) if T == 0 else T
    if T == 0 and T_val % 2: T_val += 1
    sys, anc = Register("sys", n), Register("anl", 5)
    qc = Circuit(sys, anc); Ub, BlA = prepare_b_unitary(b), block_encode(A)
    qc.unitary(Ub, target=sys[:]); mcz = Circuit(n + 1)
    mcz.mcz(controls=list(range(n)), target=n, control_state=[1] * n)
    for k in range(1, T_val + 1): add_adiabatic_step(qc, k, T_val, n, sys, anc, Ub, BlA, kappa, p, mcz)
    state = np.asarray(qc.execute(backend=backend, device=device, dtype=dtype).state, dtype=complex).reshape(-1)
    if len(state) != 2**(n + 5): raise RuntimeError("Unexpected state-vector length")
    post = state[2**(n + 4):][:N]; amp = float(np.linalg.norm(post))
    if amp < 1e-15: raise RuntimeError("Near-zero post-selection amplitude; increase T")
    v = post / amp; Av = A @ v; denom = np.vdot(Av, Av)
    if abs(denom) < 1e-15:
        raise RuntimeError("Least-squares scaling failed: <Av,Av> is near zero")
    c = np.vdot(Av, b) / denom
    x = (c * v) * (b_scale / a_scale)
    q = x / np.linalg.norm(x); r = classical / np.linalg.norm(classical)
    overlap = np.vdot(r, q); phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1.0
    aligned_error = float(np.linalg.norm(np.conj(phase) * q - r))
    source_direction_error = float(np.linalg.norm(v - r))
    return {"status": "ok", "Quantum Solution (x)": x, "Classical Solution": classical,
            "Residual Norm ||Ax-b||": float(np.linalg.norm(A_orig @ x - b_orig)),
            "Error vs Classical (L2)": source_direction_error,
            "Phase-aligned Direction Error (L2)": aligned_error,
            "Fidelity": float(abs(overlap)**2),
            "Internal Scale Factor": c, "Post-selection Amplitude": amp,
            "Post-selection Probability": amp**2, "Adiabatic Steps (T)": T_val,
            "AQC Circuit Executed": True, "circuit": qc}
```

## Debugging Tips

1. Validate `n` as an integer in range, `N=2**n`, state-vector length `2**(n+5)`, positive `T` after auto-selection, and `p>1` before evaluating the schedule.
2. In an optional debug/test run, check `\lVert A\rVert_2≤1`, block-encoding unitarity, its upper-left block, `\lVert b\rVert≈1`, and `U_b|0\rangle≈b`; omit these expensive matrix checks from routine runs.
3. For near-zero post-selection amplitude (`<1e-15` in source), increase `T` or reduce `n`; probability is amplitude squared.
4. Preserve `Error vs Classical (L2)` for the source's unaligned diagnostic. Report the improved result separately as `Phase-aligned Direction Error (L2)` together with fidelity.
5. For a large residual with good direction, inspect `a_scale`, `b_scale`, and the least-squares `internal_scale`.
6. Preserve little-endian slicing and exactly `N` system amplitudes.
7. Use `T=0` for automatic selection or positive `T`; use `p>1`. The source documents these constraints but does not fully validate them, so recommended code must.
8. The full circuit may be truncated; inspect the representative one-step export. Describe it as an 11-stage logical sequence.
