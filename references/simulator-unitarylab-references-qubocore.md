# Classical Ising and QUBO Heuristics

Follow [SKILL.md](../SKILL.md) for task routing and execution permission; see the [SDK version and APIs](simulator-unitarylab.md#sources-and-version).

Use `unitarylab.library.qubocore` for native CPU heuristic optimization. These are classical
Ising solvers, not quantum-circuit execution or a proof of quantum advantage.

## Public Functions

```python
from unitarylab.library import qubocore
```

| Function | Method |
|---|---|
| `bsb_minimize` | Ballistic simulated bifurcation |
| `dsb_minimize` | Discrete simulated bifurcation |
| `asb_minimize` | Adiabatic simulated bifurcation |
| `simcim_minimize` | SimCIM |
| `cim_minimize` | Coherent Ising machine heuristic |
| `oim_minimize` | Oscillator Ising machine heuristic |
| `phia_minimize` | Parallel HMC Ising annealer |
| `gahmc_minimize` | Exact zero-crossing HMC |

The public API uses functions taking arrays/scalars and returning dictionaries, not C++
solver objects. Import requires the bundled native `_qubocore` extension; do not substitute
another optimizer silently if it is missing or incompatible with the machine.

## Problem Contract

All functions minimize `E(s) = -0.5 * s.T @ J @ s - h.T @ s` with spins in `{-1, +1}`.
`J` must be finite, real, square, symmetric within the implemented tolerance, and have
an exactly zero diagonal, with at least one spin. Optional `h` is a finite real vector
of matching length. Inputs are converted to contiguous float64 arrays; complex inputs
are rejected, rather than implicitly taking their real part.

A binary QUBO matrix is not directly interchangeable with `J`: convert the objective to
this spin/sign convention, retaining linear and constant terms when comparing energies.
`asb_minimize` does not support local fields; do not pass `h` to that solver.

For example, the BSB entry is:

```python
result = qubocore.bsb_minimize(
    J, h=None, n_steps=1000, dt=1.0, c0=None, a0=1.0,
    n_agents=1, seed=7,
)
print(result["spins"], result["energy"])
```

`n_steps` and `n_agents` must be positive where exposed. Other methods have their own
step sizes, schedules, trajectory lengths, or sample counts; do not reuse BSB's keywords
for all functions. Auto-calibrated `c0`, `xi0`, and `K_c` accept `None` or a positive finite
value. Seeds are integers or `None`; the wrapper maps integer seeds modulo `2**64` and
uses OS entropy for `None`. This differs from the circuit execution seed contract.

## Results

| Field | Meaning |
|---|---|
| `spins` | Selected spin vector, shape `(N,)`, int8 |
| `energy` | Selected Ising energy |
| `all_spins` | Agent spin vectors, shape `(n_agents, N)` |
| `all_energies` | Agent energies, shape `(n_agents,)` |
| `n_steps`, `n_agents` | Reported execution metadata |

Check spin values and recompute energy using the same objective. A returned solution is a heuristic candidate, not a guaranteed global optimum. Fixed seeds help reproduce a configuration but do not establish performance or optimality.
