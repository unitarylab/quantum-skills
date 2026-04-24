---
name: gradients
description: Master routing guide for all quantum gradient and geometric tensor methods in this folder. Read this file first, then follow the leaf skill for the chosen method.
---

# Quantum Gradients — Master Guide

## Purpose
This file routes you to the right gradient method.

Sub-skills in this folder:
- `finite-difference/SKILL.md`
- `parameter-shift/SKILL.md`
- `linear-combination/SKILL.md`
- `spsa/SKILL.md`
- `reverse/SKILL.md`
- `qfi/SKILL.md`

## Method Selection

| Situation | Method |
|---|---|
| Circuit uses arbitrary gates, no analytic rule needed | `finite-difference` |
| Circuit uses supported rotation/controlled gates, need exact gradient | `parameter-shift` |
| Same as above, but also need QGT or complex-valued gradient | `linear-combination` |
| Many parameters, limited circuit evaluation budget | `spsa` |
| Small circuit, statevector simulation, no external backend needed | `reverse` |
| Need the Quantum Fisher Information matrix | `qfi` |

---

## Method Comparison

| Method | Analytic? | Circuit evals per call | Gate restriction | QGT support | Backend |
|---|---|---|---|---|---|
| Finite difference | No | `2n` (central) or `n+1` | None | No | Estimator / Sampler |
| Parameter shift | Yes | `2n` | `rx ry rz cx cy cz rxx ryy rzz rzx p ...` | No | Estimator / Sampler |
| Linear combination (LCU) | Yes | `n` – `2n` | Same as above + `h t s iswap` | Yes (`LinCombQGT`) | Estimator / Sampler |
| SPSA | No (stochastic) | `2 × batch_size` (fixed) | None | No | Estimator / Sampler |
| Reverse | Yes | O(P) per gradient | `rx ry rz cp crx cry crz` | Yes (`ReverseQGT`) | None (statevector) |
| QFI | Yes (via QGT) | Same as `LinCombQGT` | Same as LCU | Yes (is a QGT wrapper) | Estimator |

`n` = number of target parameters; `P` = number of parameterized gates.

---

## Method Summaries

### finite-difference
Numerically approximates `∂f/∂θ_i` by perturbing one parameter at a time.

- Classes: `FiniteDiffEstimatorGradient`, `FiniteDiffSamplerGradient`
- Key params: `epsilon`, `method` (`"central"` / `"forward"` / `"backward"`)
- No gate restrictions; works on any circuit
- Read: `finite-difference/SKILL.md`

### parameter-shift
Computes exact gradient via `(f(θ + π/2) - f(θ - π/2)) / 2` per parameter.

- Classes: `ParamShiftEstimatorGradient`, `ParamShiftSamplerGradient`
- No free tuning parameter (shift is fixed at `π/2`)
- Requires circuits from the supported gate set
- Read: `parameter-shift/SKILL.md`

### linear-combination (LCU)
Augments the circuit with an ancilla qubit to compute exact analytic gradients, and can also compute the full Quantum Geometric Tensor.

- Classes: `LinCombEstimatorGradient`, `LinCombSamplerGradient`, `LinCombQGT`
- Key param: `derivative_type` (`REAL` / `IMAG` / `COMPLEX`)
- Requires circuits from the supported gate set
- Read: `linear-combination/SKILL.md`

### SPSA
Estimates gradients stochastically by perturbing all parameters simultaneously. Only `2 × batch_size` circuit evaluations regardless of parameter count.

- Classes: `SPSAEstimatorGradient`, `SPSASamplerGradient`
- Key params: `epsilon`, `batch_size`, `seed`
- Best for high-dimensional parameter spaces with limited evaluation budget
- Read: `spsa/SKILL.md`

### reverse
Reverse-mode gradient using statevector simulation. No external primitive needed. Efficient in number of circuit evaluations but scales exponentially with qubit count.

- Classes: `ReverseEstimatorGradient`, `ReverseQGT`
- Key param: `derivative_type`, `phase_fix`
- Only for small, noiseless statevector simulations
- Read: `reverse/SKILL.md`

### qfi
Wraps a `BaseQGT` (typically `LinCombQGT`) to return the Quantum Fisher Information matrix: `QFI = 4 × Re(QGT)`.

- Class: `QFI`
- Required: a `BaseQGT` instance as `qgt` argument
- Output field: `qfis` (list of `n × n` real matrices)
- Read: `qfi/SKILL.md`

---

## Shared API Pattern
All gradient classes share the same run interface:

```python
result = gradient_or_qfi.run(
    circuits=[qc],
    observables=[obs],      # only for estimator-based methods
    parameter_values=[[...]], 
    parameters=[[...]],     # subset to differentiate; None = all
).result()
```

Return fields:
- Gradient classes: `gradients`, `metadata`, `precision` or `shots`
- QGT / QFI classes: `qgts` or `qfis`, `metadata`

---

## Quick Decision Guide

```
Does the circuit use only supported rotation/controlled gates?
├── No  → finite-difference
└── Yes
    ├── Need QGT or complex-valued gradient?
    │   ├── Yes → linear-combination (LinCombQGT / LinCombEstimatorGradient)
    │   └── No  → parameter-shift (simpler, same cost)
    │
    ├── Statevector simulation, small circuit, no backend needed?
    │   └── Yes → reverse
    │
    ├── High-dimensional parameters, limited eval budget?
    │   └── Yes → spsa
    │
    └── Need QFI matrix specifically?
        └── Yes → qfi (wraps LinCombQGT internally)
```
