---
name: hidden-shift
description: "Use when users ask about Boolean Hidden Shift, bent-function phase oracles, the X_s U_f X_s construction, Hadamard interference, sample-versus-execute measurement paths, bit-order debugging, or UnitaryLab's HiddenShiftAlgorithm. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and reference-guided reimplementation."
---

# Boolean Hidden Shift Algorithm

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement `HiddenShiftAlgorithm`.

The implementation recovers a hidden binary translation `s` for the fixed quadratic bent function

```text
f(x) = XOR_i x[2i] x[2i+1]
```

by applying the shifted phase oracle `U_g = X_s U_f X_s`, a second self-dual `U_f`, and three Hadamard layers. It supports an explicit shift or seeded random generation and provides two measurement paths.

When using this skill:

- **Explanation:** Explain the promise, fixed bent-function family, phase-oracle construction, self-duality, bit ordering, and limitations. Do not generate code unless requested.
- **Run or reuse:** Generate standalone user code importing `HiddenShiftAlgorithm` from `unitarylab_algorithms`. Do not import this skill's scripts at runtime.
- **Debugging:** Start with explicit two-bit shifts and verify source bit order before changing the circuit.
- **Modification or reimplementation:** Preserve the oracle pairing, `X_s U_f X_s` order, three Hadamard layers, measurement-mode distinction, backend-output reversal, parameter precedence, and return contract.
- **Reference scripts:** `scripts/algorithm.py` calls only the installed `unitarylab-algorithms` public API and never injects a workspace source path. `scripts/hidden_shift_implementation.py` is a separate independent reconstruction and verification aid.
- **Validation:** Exercise all two-bit shifts in both modes and assert counts, bit-string lengths, and recovered values.

This is not a generic hidden-shift oracle interface. It implements one self-dual bent-function family on a positive even number of bits.

## Overview

1. Validate `shots`, `seed`, backend, and measurement mode.
2. If `shift` is nonblank, parse it and replace `qubits` with its length; otherwise generate `qubits` seeded random bits.
3. Require a positive even register size.
4. Build `U_f` using a `CZ` gate on every adjacent pair `(0,1), (2,3), ...`.
5. Build `H -> X_s -> U_f -> X_s -> H -> U_f -> H`.
6. In sample mode, evolve once without measurement gates and sample the final state. In execute mode, append measurements and execute for `shots`.
7. Reverse backend output strings into source `q0, q1, ...` order, select the most common bit string, and compare it with the expected shift.
8. Save the circuit and text report and return the standardized result.

## Prerequisites

- Boolean functions over `GF(2)`, XOR, and binary inner products.
- Phase oracles, Hadamard/Walsh transforms, and phase interference.
- Bent functions and the meaning of self-duality for the selected quadratic function.
- Python `random.Random`, `Counter`, and UnitaryLab `Circuit`, `Register`, and `ClassicalRegister`.

## Reference Implementation Example

This example resolves `unitarylab_algorithms` from the active Python environment's installed package. Do not prepend the repository workspace to `sys.path`.

```python
from unitarylab_algorithms import HiddenShiftAlgorithm

algorithm = HiddenShiftAlgorithm(text_mode="plain")
result = algorithm.run(
    qubits=6,
    shift="101100",
    shots=100,
    seed=41,
    backend="numpy",
    measurement_mode="sample",
)

print(result["status"])
print(result["Secret shift"])
print(result["Most common bitstring"])
print(result["Counts"])
print(result["Matched"])
print(result["circuit_path"])
print(result.get("plot", []))
```

## Core Parameters Explained

**Constructor `HiddenShiftAlgorithm(text_mode, algo_dir)`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Base result-report formatting mode. |
| `algo_dir` | `str` or `None` | `None` | Output directory. When omitted, uses `results/search/hidden_shift` under the current working directory. |

**`run(qubits, shift, shots, seed, backend, measurement_mode)`:**

| Parameter | Type | Default | Source contract |
|---|---|---:|---|
| `qubits` | integer-like | `6` | Used only when `shift` is blank; must then be positive and even. |
| `shift` | `str` or `None` | `""` | Blank/`None` requests seeded generation. Nonblank input must be binary and its length overrides `qubits`. |
| `shots` | integer-like | `100` | Positive number of samples or measured executions. |
| `seed` | integer-like | `41` | Non-negative; used for random shift generation, circuit execution, and sampling. |
| `backend` | `str` | `"numpy"` | Exactly `"numpy"` or `"torch"`; device is fixed to CPU. |
| `measurement_mode` | `str` | `"sample"` | Exactly `"sample"` or `"execute"`. |

`parameters.json` advertises UI maxima of 20 qubits and 100000 shots. `algorithm.py` does not enforce those maxima; do not describe them as callable validation. The UI regex also requires a nonblank shift to have even length, while runtime enforces even length after deriving `qubits`.

All-zero shifts are valid. Unlike Simon's algorithm, Hidden Shift does not reject the trivial zero string.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` when the most common bit string equals the expected shift; otherwise `"failed"`. |
| `circuit_path` | `str` | Saved SVG path. |
| `plot` | `list[dict]` | Text-report descriptor produced by the base return builder. |
| `circuit` | `Circuit` | Constructed circuit; contains measurement gates only in execute mode. |
| `Secret shift` | `str` | Explicit or generated shift in `q0, q1, ...` order. |
| `Most common bitstring` | `str` | Dominant normalized backend outcome. |
| `Counts` | `dict[str, int]` | Normalized bit-string frequencies. |
| `Matched` | `bool` | Equality between dominant output and expected shift. |
| `Circuit construction (s)` | `float` | Circuit construction wall time. |
| `Execution and sampling (s)` | `float` | Execution/sampling wall time. |
| `Total time (s)` | `float` | Total wall time. |

The base result does not expose the constructor's internal `summary`, `input`, or `info` dictionaries.

## Implementation Architecture

`HiddenShiftAlgorithm.run()` has three logged stages and four module-level helpers.

| Stage | Code action | Algorithmic role |
|---|---|---|
| 1 — Build | Resolve shift and call `make_hs_circuit()` | Construct the exact hidden-shift interference circuit. |
| 2 — Execute | Use `sample_circuit()` or measured `execute()` | Produce final-state frequencies through the selected measurement contract. |
| 3 — Verify/export | Select the dominant output, compare with `bitstring(shift_bits)`, save files | Convert execution into the public success/result contract. |

**Key helpers:**

- `bitstring(bits)` joins integer-converted bits without reversing them; it defines the source-visible `q0, q1, ...` expected string.
- `make_oracle_f(qubit_count)` rejects non-positive or odd sizes and applies one `CZ` per adjacent pair.
- `make_hs_circuit(shift, measure=False)` validates a non-empty, even-length binary sequence and builds the exact seven-block circuit.
- `sample_circuit()` rejects circuits that already have measurements, executes once, samples the final state, and reverses every returned bit string.

**Parameter precedence:** a nonblank explicit shift determines the qubit count. The numeric `qubits` argument is ignored after the shift is parsed.

**Data flow:** parameters → explicit/seeded shift bits → circuit → sample or counts path → reverse output strings → `Counter` → dominant answer → match/status → artifacts/result.

## Understanding the Key Quantum Components

### 1. Bent-function oracle

For even `n`, the implemented function is

```text
f(x) = x0*x1 XOR x2*x3 XOR ... XOR x[n-2]*x[n-1].
```

Each `CZ(2i, 2i+1)` contributes phase `(-1)^(x[2i]x[2i+1])`, so their product implements `U_f |x> = (-1)^f(x) |x>`.

### 2. Shifted oracle

For hidden shift `s`, the translated function is `g(x) = f(x XOR s)`. Because Pauli X toggles the selected input bits,

```text
U_g = X_s U_f X_s.
```

The source applies X before and after the first `U_f` on exactly the indices where the supplied shift bit is one.

### 3. Hadamard interference and self-duality

The circuit is

```text
H^n -> U_g -> H^n -> U_f -> H^n.
```

The Walsh transform converts translation by `s` into a linear phase. For this selected self-dual bent function, the second `U_f` cancels the function-dependent phase, leaving the final Hadamard layer to produce `|s>` in the ideal noiseless model.

### 4. Measurement paths

Sample mode has no circuit measurements: it executes one state and calls `result.sample(shots, seed)`. Execute mode appends measurements and calls `execute(..., shots=shots, seed=seed)`, then uses `result.counts`.

### 5. Bit ordering

The source shift text is ordered `q0, q1, ...`. Backend sample/count strings arrive in the opposite display order, so both paths normalize outcomes with `outcome[::-1]`. Omitting this reversal can produce a plausible but reversed secret.

## Theory-to-Code Mapping

| Theory or contract | Source object or expression |
|---|---|
| Binary source string | `bitstring(shift_bits)` |
| Pairwise quadratic `f` | `CZ` loop in `make_oracle_f()` |
| Initial uniform superposition | First `circuit.h(qubits)` |
| `X_s` before oracle | First conditional `circuit.x(shifted_qubits)` |
| `U_f` inside `U_g` | First `circuit.append(oracle_f, target=qubits)` |
| Undo `X_s` | Second conditional X block |
| Walsh transform | Second Hadamard block |
| Self-dual phase cancellation | Second `U_f` append |
| Recover computational state | Final Hadamard block |
| Sample-mode normalization | `Counter(outcome[::-1] for outcome in result.sample(...))` |
| Execute-mode normalization | Reversed keys from `result.counts.items()` |
| Success condition | `answer == bitstring(shift_bits)` |

## Mathematical Deep Dive

Let the normalized Walsh transform of `f` be

```text
W_f(y) = 2^(-n/2) sum_x (-1)^(f(x) + x dot y).
```

For a bent function, all Walsh coefficients have the same magnitude and may be written `(-1)^f*(y)` for a dual function `f*`. Translation gives

```text
W_g(y) = (-1)^(s dot y) W_f(y).
```

For the pairwise quadratic used here, the relevant dual phase is canceled by the second application of `U_f`. The remaining state before the last Hadamard has amplitudes proportional to `(-1)^(s dot y)`. The identity

```text
H^n sum_y (-1)^(s dot y) |y> / sqrt(2^n) = |s>
```

then yields the secret shift deterministically in the ideal circuit.

The circuit uses `n` qubits, three Hadamard layers (`3n` H gates), two bent-function oracles (`n` CZ gates total), and up to `2n` X gates for the shifted oracle. Classical state-vector simulation still scales exponentially with `n`.

## Hands-On Example

```python
from unitarylab_algorithms import HiddenShiftAlgorithm

for mode in ("sample", "execute"):
    result = HiddenShiftAlgorithm().run(
        shift="0110",
        shots=32,
        seed=41,
        backend="numpy",
        measurement_mode=mode,
    )
    assert result["status"] == "ok"
    assert result["Matched"]
    assert result["Most common bitstring"] == "0110"
    assert sum(result["Counts"].values()) == 32
```

For a seeded-random run, leave `shift=""`; record both `qubits` and `seed` so the secret is reproducible.

## Minimal Manual Implementation

The following is a self-contained minimal implementation of the same Hidden Shift algorithm. It does not import `HiddenShiftAlgorithm`, the source module, or this skill's reference scripts. It constructs the bent-function phase oracle and the full interference circuit directly from `H`, `X`, `CZ`, and measurement operations, then preserves both source execution modes and bit ordering.

```python
from collections import Counter
import random

from unitarylab import Circuit, ClassicalRegister, Register


def build_hidden_shift_circuit(bits, measure=False):
    """Build H -> X_s U_f X_s -> H -> U_f -> H from basic gates."""
    bits = tuple(bits)
    if not bits or len(bits) % 2 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("shift must be a non-empty even-length binary sequence")
    n = len(bits)
    quantum = Register("q", n)
    classical = ClassicalRegister("result", n)
    circuit = Circuit(quantum, classical, name="Hidden Shift")
    qubits = list(range(n))
    shifted = [index for index, bit in enumerate(bits) if bit]

    # First H layer: uniform superposition.
    circuit.h(qubits)

    # U_g = X_s U_f X_s, with U_f = product_i CZ(2i, 2i+1).
    if shifted:
        circuit.x(shifted)
    for qubit in range(0, n, 2):
        circuit.cz(qubit, qubit + 1)
    if shifted:
        circuit.x(shifted)

    # Walsh transform, the second U_f, and final recovery transform.
    circuit.h(qubits)
    for qubit in range(0, n, 2):
        circuit.cz(qubit, qubit + 1)
    circuit.h(qubits)

    if measure:
        circuit.measure(quantum[:], classical[:])
    return circuit


def run_hidden_shift(
    qubits=6,
    shift="",
    shots=100,
    seed=41,
    backend="numpy",
    measurement_mode="sample",
):
    """Resolve the shift, execute the same two source paths, and verify it."""
    if shots <= 0 or seed < 0:
        raise ValueError("shots must be positive and seed non-negative")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    if measurement_mode not in ("sample", "execute"):
        raise ValueError("measurement_mode must be 'sample' or 'execute'")

    shift = "" if shift is None else str(shift).strip()
    if shift:
        if any(bit not in "01" for bit in shift):
            raise ValueError("shift must be a binary string")
        bits = [int(bit) for bit in shift]
        qubits = len(bits)  # explicit shift overrides qubits
    else:
        rng = random.Random(seed)
        bits = [rng.randint(0, 1) for _ in range(qubits)]

    if qubits <= 0 or qubits % 2:
        raise ValueError("the number of qubits must be positive and even")

    measured = measurement_mode == "execute"
    circuit = build_hidden_shift_circuit(bits, measure=measured)

    if measurement_mode == "sample":
        result = circuit.execute(backend=backend, device="cpu", seed=seed)
        counts = Counter(
            outcome[::-1]
            for outcome in result.sample(shots=shots, seed=seed)
        )
    else:
        result = circuit.execute(
            backend=backend,
            device="cpu",
            shots=shots,
            seed=seed,
        )
        counts = Counter(
            {outcome[::-1]: count for outcome, count in result.counts.items()}
        )

    expected = "".join(str(bit) for bit in bits)  # q0, q1, ... order
    answer = counts.most_common(1)[0][0]
    return {
        "status": "ok" if answer == expected else "failed",
        "secret_shift": expected,
        "most_common_bitstring": answer,
        "counts": dict(counts),
        "matched": answer == expected,
        "circuit": circuit,
    }


for mode in ("sample", "execute"):
    result = run_hidden_shift(
        shift="101100",
        shots=100,
        seed=41,
        backend="numpy",
        measurement_mode=mode,
    )
    assert result["matched"]
    assert result["most_common_bitstring"] == "101100"
    assert sum(result["counts"].values()) == 100
```

The output reversal is part of the algorithm contract: UnitaryLab backend strings are reversed before comparison with the source-visible `q0, q1, ...` shift. The full independent reference adds compatibility checks, exhaustive two-bit cases, CLI options, and separate static/execution test entry points:

```bash
python scripts/hidden_shift_implementation.py
python scripts/hidden_shift_implementation.py --execute --backend numpy
```

The default command prints a structural report rather than a single pass/fail marker: the implemented Boolean oracle, circuit sequence, source bit order, backend normalization rule, shifted-qubit indices, `H/CZ/X` counts, measurement placement for every two-bit shift, the deterministic seeded shift, explicit-shift precedence, and rejected-input errors. With `--execute`, it additionally prints every shift and measurement mode, the recovered string, match/status fields, complete normalized counts, count totals, and whether the executed circuit contains measurement operations.

## Debugging Tips

1. **Odd or empty shift:** The circuit helper requires a non-empty even-length sequence. A blank public `run()` shift means “generate one,” not an empty circuit.
2. **Non-binary shift:** Public `run()` rejects any character outside `0` and `1`.
3. **Explicit shift length differs from `qubits`:** This is intentional; the explicit shift wins.
4. **Recovered value is reversed:** Normalize backend outcomes with `[::-1]` in both measurement paths.
5. **`sample_circuit()` rejects the circuit:** It accepts only the unmeasured sample-mode circuit.
6. **Count total differs from shots:** Inspect backend result semantics and verify the selected mode before changing the algorithm.
7. **Unsupported backend or mode:** The source accepts only `numpy`/`torch` and `sample`/`execute`.
8. **Current environment rejects `seed` in `Circuit.execute()`:** The installed UnitaryLab is older than the source contract. Use a compatible version; do not silently remove deterministic seed handling.

The authoritative callable is `unitarylab_algorithms/search/hidden_shift/algorithm.py`. `parameters.json` describes UI constraints and the bilingual READMEs explain the intended algorithm, but neither overrides implemented Python behavior.
