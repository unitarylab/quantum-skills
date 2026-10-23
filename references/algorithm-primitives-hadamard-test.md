# Hadamard Test

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The Hadamard Test uses a single ancilla qubit to estimate the complex expectation value $\langle\psi|U|\psi\rangle$ for a unitary $U$ and state $|\psi\rangle$. It is a fundamental subroutine in quantum algorithms, supporting expectation value estimation, state overlap testing (swap test), and single-bit phase estimation.

The documented modes estimate real/imaginary expectations, squared state overlap, or the argument of a complex expectation. The last quantity is an eigenphase only when the prepared state is an eigenstate of `U`.

Source pointers: [algorithm-library usage example](../scripts/hadamard_test_algorithm.py) and [independent teaching implementation](../scripts/hadamard_test_implementation.py). The teaching solver has a separate result envelope.

Algorithm-library source: `HadamardTestAlgorithm` (`unitarylab_algorithms/fundamental_algorithm/hadamard_test/algorithm.py`).

## Overview

Three supported modes:
1. **`'expectation'`**: Estimates $\text{Re}$ or $\text{Im}$ of $\langle\psi|U|\psi\rangle$ by measuring $\langle Z\rangle_{\text{anc}} = p(0) - p(1)$.
2. **`'swap_test'`**: Estimates $|\langle\phi|\psi\rangle|^2$ using a controlled-SWAP circuit.
3. **`'phase_estimation'`**: Runs real and imaginary tests, then returns `atan2(Im, Re)/(2π) mod 1`. This is an eigenphase only for an eigenstate; it is undefined as phase information when the complex expectation is zero.

The core circuit (for mode `'expectation'`):
1. Ancilla qubit in $|0\rangle$.
2. Apply $H$ to ancilla (optionally $S^\dagger$ for imaginary part).
3. Optionally prepare $|\psi\rangle$ on target.
4. Apply controlled-$U$ (ancilla controls, target receives $U$).
5. Apply $H$ to ancilla.
6. Measure ancilla: $\langle Z\rangle = p(0) - p(1) = \text{Re}\langle\psi|U|\psi\rangle$.

## Prerequisites

- Quantum gates: H, S, $S^\dagger$, controlled-U.
- Python: `numpy`, `Circuit`.

## Reference Implementation Example

```python
from unitarylab_algorithms import HadamardTestAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Example: estimate Re(<+|RZ(0.8)|+>) = cos(0.4)
U = Circuit(1, name="RZ_0.8")
U.rz(0.8, 0)

prepare_psi = Circuit(1, name="|+>")
prepare_psi.h(0)

algo = HadamardTestAlgorithm()

# Estimate real part
result_re = algo.run(mode='expectation', U=U, prepare_psi=prepare_psi,
                     imag=False, shots=20000, backend='torch')
print(result_re['Estimated Value'])   # ≈ cos(0.4) ≈ 0.9211

# Estimate imaginary part
result_im = algo.run(mode='expectation', U=U, prepare_psi=prepare_psi,
                     imag=True, shots=20000, backend='torch')
print(result_im['Estimated Value'])   # ≈ 0: <+|RZ(0.8)|+> is real
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `'expectation'` | One of `'expectation'`, `'swap_test'`, `'phase_estimation'`. |
| `U` | `Circuit` or `None` | `None` | Unitary operator. Required for `'expectation'` and `'phase_estimation'`. |
| `prepare_psi` | `Circuit` or `None` | `None` | Prepares $\|\psi\rangle$. Required for `'swap_test'` and optional for others. |
| `prepare_phi` | `Circuit` or `None` | `None` | Prepares $\|\phi\rangle$. Required only for `'swap_test'`. |
| `imag` | `bool` | `False` | Selects the imaginary expectation by inserting $S^\dagger$ after the first ancilla H; ignored by the other modes. |
| `shots` | `int` or `None` | `20000` | Positive values enable binomial shot noise; `None` or nonpositive values use exact ancilla probabilities. |
| `backend` | `str` | `'torch'` | Forwarded to `Circuit.execute()`; no class-level backend restriction. |
| `device` | `str` | `'cpu'` | Forwarded execution device. |
| `dtype` | dtype | `np.complex128` | Forwarded simulation dtype. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `imag=True` inserts $S^\dagger$ after the first ancilla H, giving $\text{Im}\langle\psi|U|\psi\rangle$.
- For `'swap_test'`, `U` is not needed; the overlap is measured via controlled-SWAP.
- Shot noise is simulated via `numpy.random.binomial`; the exact statevector probability is known internally.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on normal completion; invalid inputs and execution/export errors raise exceptions. It is not an accuracy test. |
| `Estimated Value` | `float` | `Re`/`Im` expectation, clipped squared overlap, or normalized complex-expectation argument in $[0,1)$, according to mode. |
| `Computation Time (s)` | `float` | Wall time from input setup through post-processing; circuit/text exports occur afterward. |
| `circuit_path` | `list[str]` | List of file paths to saved SVG circuit diagram(s). One path for `'expectation'` and `'swap_test'`; two paths for `'phase_estimation'`. |
| `plot` | `list[dict]` | List of saved result files, each as `{"format": "txt", "filename": "<path>"}`. |
| `circuit` | `Circuit` | The primary `Circuit` object (first circuit built in the selected mode). |

## Implementation Architecture

`HadamardTestAlgorithm` in `algorithm.py` organizes execution into five stages within `run()`, with a single circuit-building helper and shot-noise simulation.

**`run(mode, U, prepare_psi, prepare_phi, imag, shots, backend)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Validation | Checks `mode` string and required parameters per mode | Prevents invalid mode / missing input combinations |
| 2 — Circuit Construction | Dispatches to `_build_hadamard_test_circuit()` based on mode; for `swap_test` builds joint prep and U_swap; for `phase_estimation` builds both real and imag circuits | Creates one or two `Circuit` objects stored in `circuits` dict |
| 3 — Simulation + Sampling | For each circuit: `circ.execute()` → `result.marginal_probabilities(qubits=[0], threshold=0.0)` → extracts `p0_exact`; simulates shot noise via `numpy.random.binomial(shots, p0_exact)` | Runs statevector simulation; converts ancilla probabilities to noisy `<Z>` estimate |
| 4 — Post-Processing | Accumulates `measurements` from `<Z> = p0 - p1`; computes final `est_val` per mode | Expectation: returns `<Z>` directly; Swap test: clips to `[0,1]`; Phase estimation: calls `_estimate_phi_from_real_imag()` |
| 5 — Export | `self.save_circuit(circ, filename)` for each circuit; `self.save_txt()` for text output | Saves SVG circuit diagram(s) and result text file; calls `_build_return_dict()` |

**Helper Methods:**

- **`_build_hadamard_test_circuit(U, prepare_psi, imag)`** — Constructs the single-ancilla circuit. Ancilla is qubit 0; target is qubits `1..n`. Applies `H` to ancilla, optionally `sdag` for imaginary part, appends `prepare_psi` to target, then appends `U` as a controlled unitary (`qc.append(U, target=tgt, control=[anc], control_state='1')`), applies `H` again to ancilla.
- **`_estimate_phi_from_real_imag(cos_est, sin_est)`** — Computes `atan2(sin_est, cos_est) / (2π) % 1.0` to extract the eigenphase $\phi \in [0,1)$.
- **`_as_statevector(result)`** — Converts its argument to a flat NumPy array; it is not called by this `run()` path and does not normalize amplitudes.
- **`update_output(output)` / `_build_return_dict(success, circuit_path, filepath, circuit)`** — Stores `Estimated Value` and `Computation Time (s)` in `self.output`; packages the final result dict.

**Mode dispatch logic:**
- `'expectation'`: builds one circuit with `imag` flag
- `'swap_test'`: builds joint state `|φ⊗ψ⟩` and controlled-SWAP unitary; runs real circuit only
- `'phase_estimation'`: builds two circuits (real + imag); combines via `_estimate_phi_from_real_imag()`

**Data flow:** mode selection → `_build_hadamard_test_circuit()` per branch → `execute()` → ancilla `p0` → shot noise → `<Z>` → mode-specific reduction → `est_val` → `update_output()` → `_build_return_dict()`.

## Understanding the Key Quantum Components

The ancilla qubit (qubit 0) acts as a quantum probe. After the circuit, its measurement probabilities encode the inner product:
- Real test: $p(0) - p(1) = \text{Re}\langle\psi|U|\psi\rangle$
- Imaginary test: $p(0) - p(1) = \text{Im}\langle\psi|U|\psi\rangle$

### 2. Phase Kickback via Controlled-$U$
The controlled-$U$ gate (controlled by the ancilla, applied to the target register) creates quantum interference. When the ancilla is in superposition $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$, the target evolves conditionally, and the ancilla accumulates relative phase information about $U$.

### 3. $S^\dagger$ for Imaginary Part
Inserting $S^\dagger$ after the first $H$ rotates the ancilla by $-\pi/2$, effectively shifting the measurement basis to extract the imaginary part.

### 4. Swap Test Circuit
For mode `'swap_test'`, a SWAP gate on the joint system $|\phi\rangle|\psi\rangle$ is used. The controlled-SWAP with ancilla in superposition produces:
$$p(0) = \frac{1}{2}(1 + |\langle\phi|\psi\rangle|^2)$$
So $|\langle\phi|\psi\rangle|^2 = 2p(0) - 1$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Ancilla in superposition $\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$ | `qc.h(anc)` in `_build_hadamard_test_circuit()` |
| $S^\dagger$ rotation for imaginary part | `qc.sdag(anc)` inserted if `imag=True` |
| State preparation $|\psi\rangle$ on target | `qc.append(prepare_psi, target=tgt)` |
| Controlled-$U$ gate | `qc.append(U, target=tgt, control=[anc], control_state='1')` |
| Final $H$ measurement on ancilla | Second `qc.h(anc)` closes the Hadamard test |
| $\langle Z\rangle = p(0) - p(1)$ | `exp_val = p0 - p1` in Stage 3 |
| Shot noise model | `c0 = rng.binomial(shots, p0_exact)`; `p0 = c0 / shots` |
| SWAP test: $p(0) = (1 + \|\langle\phi|\psi\rangle\|^2)/2$ | `U_swap` built as SWAP gate on joint system; same circuit path as expectation |
| Phase $\phi = \text{atan2}(\text{Im}, \text{Re}) / 2\pi$ | `_estimate_phi_from_real_imag(re_est, im_est)` |
| Return value `Estimated Value` | `est_val` stored via `update_output()`, returned by `_build_return_dict()` — single float for all modes |

**Notes on encapsulation:** The swap test is implemented by constructing a new `U_swap` from SWAP gates and a joint preparation `Circuit`, then running the exact same `_build_hadamard_test_circuit` code path. No special circuit structure is needed for the swap test case beyond this reuse. Shot noise is applied as a post-simulation binomial sampling; the underlying statevector probability is exact.

## Mathematical Deep Dive

Starting from $|0\rangle_{\text{anc}}|\psi\rangle_{\text{tgt}}$, after the full circuit:
$$p(0) = \frac{1 + \text{Re}\langle\psi|U|\psi\rangle}{2}, \quad p(1) = \frac{1 - \text{Re}\langle\psi|U|\psi\rangle}{2}$$
$$\langle Z\rangle_{\text{anc}} = p(0) - p(1) = \text{Re}\langle\psi|U|\psi\rangle$$

**Imaginary Hadamard test** (with $S^\dagger$ inserted):
$$p(0) - p(1) = \text{Im}\langle\psi|U|\psi\rangle$$

**Combined complex estimation:**
$$\langle\psi|U|\psi\rangle = \text{Re} + i \cdot \text{Im}$$

**Phase estimation mode:**
For $z=\langle\psi|U|\psi\rangle$, $|z|=\sqrt{\text{Re}^2+\text{Im}^2}$ and the returned normalized phase is $(\operatorname{atan2}(\text{Im},\text{Re})/(2\pi))\bmod1$. An eigenstate has $z=e^{2\pi i\phi}$; a general state need not have $|z|=1$.

## Hands-On Example

```python
from unitarylab_algorithms import HadamardTestAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Swap test: estimate |<phi|psi>|^2
n = 2
prep_phi = Circuit(n, name="phi")
prep_phi.h(0); prep_phi.h(1)  # |++>

prep_psi = Circuit(n, name="psi")
prep_psi.h(0); prep_psi.cx(0, 1)  # Bell state

algo = HadamardTestAlgorithm()
result = algo.run(
    mode='swap_test',
    prepare_psi=prep_psi,
    prepare_phi=prep_phi,
    shots=40000,
    backend='torch'
)
print(f"Overlap^2: {result['Estimated Value']:.4f}")
# |<Bell|++>|^2 = |(<00|+<11|)/sqrt(2) * (|++>)|^2 = 0.5

# Phase estimation mode
U2 = Circuit(1, name="T_gate")
U2.t(0)  # T|0> = |0> so <0|T|0> = 1
result2 = algo.run(mode='phase_estimation', U=U2, shots=20000)
print(result2['Estimated Value'])
```

## Minimal Manual Implementation

```python
from unitarylab.core import Circuit
from typing import Optional

def hadamard_test_circuit(U: Circuit, prepare_psi=None, imag=False) -> Circuit:
    n = U.get_num_qubits()
    qc = Circuit(1 + n, name="HadamardTest")
    anc = 0
    tgt = list(range(1, 1 + n))

    qc.h(anc)
    if imag:
        qc.sdag(anc)

    if prepare_psi is not None:
        qc.append(prepare_psi, target=tgt)

    qc.append(U, target=tgt, control=[anc], control_state="1")
    qc.h(anc)
    return qc
```

## Debugging Tips

1. **`'swap_test'` requires both `prepare_psi` and `prepare_phi`**: Both must have the same number of qubits.
2. **Shot noise makes estimate vary**: Increase `shots` for a more stable estimate. The exact value is computed from the statevector; shot noise is simulated for realism.
3. **A component is close to zero**: Real eigenvalues $\pm1$ have imaginary expectation zero and real expectation $\pm1$. For the RZ example on $|+\rangle$, the imaginary part is zero even though $|+\rangle$ is not an RZ eigenstate.
4. **`mode='expectation'` with `prepare_psi=None`**: The target starts in $|0\rangle$, so you get $\langle 0|U|0\rangle$.
5. **Wrong mode string**: The mode must be exactly `'expectation'`, `'swap_test'`, or `'phase_estimation'`.
