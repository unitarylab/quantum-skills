# Adiabatic Quantum Linear-System Solver (AQC)

## Scope and Reference Use

The implementation generates a deterministic small real symmetric system $Ax=b$, normalizes it, prepares $|b\rangle$ with a Householder transform, block-encodes $A$ with an SVD construction, applies a discretized adiabatic schedule on `n` system qubits and five ancillas, post-selects the ancilla pattern, and reconstructs a classical solution vector. It is an educational dense-matrix/statevector solver, not a scalable fault-tolerant QLSP.

This reference covers:

- Demonstrate or explain the repository's adiabatic approach to a generated quantum linear-system problem.
- Inspect Householder state preparation, SVD block encoding, schedule parameters, five-ancilla layout, or little-endian post-selection.
- Diagnose normalization, condition-number, automatic-step, post-selection, reconstruction, residual, direction-error, fidelity, or scale-conversion issues.
- Compare the public source contract with the separately selected teaching implementation below.

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

Wrapper source: `unitarylab_algorithms/linear_algebra/aqc/algorithm.py`.

Reference scripts: [aqc_algorithm.py](../scripts/aqc_algorithm.py) and [aqc_implementation.py](../scripts/aqc_implementation.py). Start diagnostic comparisons with `n=1` or `n=2`, recording condition number, selected `T`, register width, residual, normalized-direction error, post-selection amplitude, backend/device/dtype, and any teaching-only fidelity/probability fields.

Keep two contracts separate throughout the response:

- **Source-equivalent behavior:** Preserve the current `run()` signature, seed `42`, generated inputs, five-ancilla layout, normalized circuit problem, little-endian post-selection, `kappa≈1` branch, normalized-problem `Classical Solution` in the normal branch, first-nonzero-component internal scale, direction comparison without global-phase alignment, reconstruction scale, and exact source return keys.
- **Teaching implementation behavior:** The independent example below puts both vectors on the original generated-problem scale, aligns phase for an additional direction diagnostic, reports fidelity/probability, and uses least-squares scaling. These differences apply only when the user selects that teaching example; they do not change the public source API or direct an automatic source-library upgrade.

## Overview

1. Generate a real symmetric matrix with a diagonal shift of `10I` and a random right-hand side with NumPy seed `42`, dimension $N=2^n$.
2. Normalize $A$ by $\lVert A\rVert_2$ and $b$ by $\lVert b\rVert_2$; retain `a_scale` and `b_scale` so $A_\mathrm{orig}=a_\mathrm{scale}A$ and $b_\mathrm{orig}=b_\mathrm{scale}b$.
3. Choose `T` (`T=0` selects `ceil(10*kappa)`, rounded up to even), create `n` system qubits and five ancillas, build Householder $U_b$ and the SVD block encoding, and initialize the system to $|b\rangle$.
4. After this setup, the source handles `kappa≈1` with a shortcut that returns the normalized `b` direction with original-scale rescaling. It computes a classical solve for comparison, skips the adiabatic loop and statevector execution, and avoids the singular schedule factor `kappa/(kappa-1)`.
5. Otherwise apply the `T` discrete adiabatic steps and execute the statevector circuit.
6. In little-endian order, select ancilla `|10000⟩` (`anl[4]=1`, `anl[0:4]=0`). The current source returns only the post-selection amplitude.
7. The independent teaching implementation below additionally returns post-selection probability, fidelity, and a phase-aligned direction error. The public source does not align global phase.

For a positive-definite Hermitian matrix, `kappa≈1` implies equal eigenvalues and therefore a scalar multiple of the identity (after normalization, `A≈I`). This implication must not be extended to a general matrix merely from a condition number near one.

The construction `symmetric_random_matrix + 10I` does not guarantee positive definiteness for every size. Inspect the generated spectrum and condition number for the chosen input; the fixed diagonal shift is not a size-independent positivity certificate. Small `n=1` or `n=2` examples keep the documented dense construction manageable.

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
| `n` | `int` | `2` | System-qubit count; `N=2**n`; documented target `1 ≤ n ≤ 6`. `run()` does not enforce this range or reject `bool`. |
| `T` | `int` | `0` | Positive step count, or `0` for automatic even selection. `run()` does not validate type or range. |
| `p` | `float` | `1.4` | Schedule exponent; the mathematical schedule requires `p > 1`. `run()` does not validate this constraint or finiteness. |
| `backend` | `str` | `'torch'` | UnitaryLab backend. |
| `device` | `str` | `'cpu'` | Execution device. |
| `dtype` | | `np.complex128` | Complex dtype passed to `Circuit.execute`; manual code must pass this parameter too. |

The source generates `A` and `b` internally; `run()` does not accept user-supplied inputs.

## Source Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on normal return; does not certify residual, direction accuracy, or post-selection quality. |
| `Quantum Solution (x)` | `np.ndarray` | Reconstructed solution; normal branch rescales it to the original generated system. |
| `Classical Solution` | `np.ndarray` | Normal branch: `np.linalg.solve(A,b)` on normalized `A,b`; trivial branch uses original-scale inputs. |
| `Residual Norm ||Ax-b||` | `float` | Residual evaluated against original-scale `A_orig,b_orig`. |
| `Error vs Classical (L2)` | `float` | Normal branch: unaligned normalized-direction L2 error. Trivial branch: L2 error between original-scale returned and classical vectors. |
| `Internal Scale Factor` | numeric | Source's first nonzero-component scale estimate, potentially complex in the normal branch. |
| `Post-selection Amplitude` | `float` | Normal branch only: norm of the selected system vector. |
| `Simulation Time (s)` / `Elapsed Time (s)` | `float` | Timing fields; simulation time is absent in the trivial branch. |
| `circuit_path` | `list[str]` or `str` | Normal branch: paths for `aqc_full` and `aqc_one_step`; trivial branch: one path. |
| `plot` | `list[dict]` | Text-output file metadata. |
| `circuit` | `Circuit` | Full normal-branch circuit or initialized shortcut circuit; this is not the teaching extension's `None` shortcut result. |

## Teaching Implementation Fields

These fields describe the independent teaching example below. They apply when that example is explicitly selected, and are not fields or changed semantics promised by the current public `algorithm.py`.

| Key | Type | Description |
|---|---|---|
| `Classical Solution` | `np.ndarray` | `np.linalg.solve(A_orig,b_orig)`, on the same original scale as the quantum solution. |
| `Phase-aligned Direction Error (L2)` | `float` | L2 error after aligning the global phase of normalized directions. Keep the source's `Error vs Classical (L2)` unchanged; repurposing that key would be a breaking semantic change. |
| `Fidelity` | `float` | `abs(q.conj() @ c)**2` for normalized quantum and classical directions. |
| `Post-selection Probability` | `float` | Square of the post-selection amplitude. |
| `Internal Scale Factor` | `complex` | Least-squares scale `c = <Av,b>/<Av,Av>`, rather than a single-component estimate. A near-zero denominator is an error, not justification for `c=1`. |
| `AQC Circuit Executed` | `bool` | `False` for the `kappa≈1` classical shortcut and `True` for the normal AQC branch. |

## Implementation Architecture

`run()` has four logged stages: (1) deterministic generation and normalization; (2) dense circuit construction; (3) statevector simulation; and (4) post-selection, reconstruction, diagnostics, and exports. It performs little input validation. The teaching extension below moves the `kappa≈1` check before circuit construction and returns a classical solve; that ordering differs from the source.

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

Select the branch explicitly: set bits `n+0..n+3` to zero and bit `n+4` to one, or equivalently use the source slice `state_arr[2**(n+4):]` and then take exactly the first `N=2**n` amplitudes. The source returns the selected vector norm (amplitude); the teaching extension additionally reports its squared probability.

### 4. Discrete adiabatic step

Repeat the 11-stage logical sequence for `k=1,...,T`. Preserve the mirrored schedule and both `U_b` reflection sequences when reproducing source behavior.

### 5. Classical reconstruction

Let `v` be the normalized post-selected direction. The source uses `b[i]/(A[i] @ v)` for the first component with nonzero `b[i]`, falling back to `1.0` when that scale has near-zero magnitude. The teaching extension instead uses the complex least-squares scale

$$c=\frac{(Av)^\dagger b}{(Av)^\dagger(Av)},\qquad x_\mathrm{problem}=cv.$$

The teaching example restores original scale with `x_orig=(b_scale/a_scale)x_problem` and preserves the source's unaligned diagnostic under `Error vs Classical (L2)`. For its additional phase-aligned diagnostic, let `q` and `r` be normalized original-scale solutions, choose `phase=(r.conj()@q)/abs(r.conj()@q)` when nonzero, and report `norm(phase.conj()*q-r)` under `Phase-aligned Direction Error (L2)`; fidelity is `abs(r.conj()@q)**2`. Reusing the public key for this different calculation would change its semantics.

## Theory-to-Code Mapping

| Theory concept | Source implementation |
|---|---|
| $\lvert b\rangle$ preparation | `_uprep_b`; `qc.unitary(Ub_mat, target=sys_qubits[:])` |
| Block encoding | `_bloc_enc(A)` via SVD |
| Registers | `Register('sys', n_sys)`, `Register('anl', 5)` |
| Normalized problem | `A/a_scale`, `b/b_scale` |
| Auto steps | `ceil(10*kappa)`, rounded to even |
| Post-selection | `state_arr[2**(n+4):][:2**n]` |
| Source classical reference | `np.linalg.solve(A,b)` in normal branch |
| Teaching example reference | `np.linalg.solve(A_orig,b_orig)` |
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

The independent example uses `from unitarylab import Circuit, Register`, deterministic generation, normalization, five ancillas, and the 11-stage sequence. It adds checks for `n`, `T`, and `p`, least-squares scaling, an original-scale classical reference, and separate phase/fidelity/probability diagnostics. It moves the shortcut before circuit construction and passes `dtype` through. These are teaching-example differences, not the default public wrapper contract; optional Householder and block-encoding checks are described above.

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
4. The teaching example reports its additional diagnostic as `Phase-aligned Direction Error (L2)` together with fidelity; the public source's existing key keeps its original semantics.
5. For a large residual with good direction, inspect `a_scale`, `b_scale`, and the active implementation's scale calculation: first-nonzero-component in the public source, least-squares in the teaching example.
6. Preserve little-endian slicing and exactly `N` system amplitudes.
7. `T=0` selects automatically; the mathematical schedule otherwise needs positive `T` and `p>1`. The public source does not fully validate them, while the teaching example adds checks.
8. The full circuit may be truncated; inspect the representative one-step export. Describe it as an 11-stage logical sequence.
