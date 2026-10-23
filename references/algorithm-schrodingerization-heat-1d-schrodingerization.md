
# 1D Heat Equation via Schrödingerization

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The `HeatEquationAlgorithm` source contract (`unitarylab_algorithms/schrodingerization/equation_heat/algorithm.py`) dispatches classical and Trotter paths through `unitarylab.library.equation` SDK helpers; `block` currently falls back to classical. `../scripts/heat_1d_schrodingerization_algorithm.py` shows the wrapper call. The manual example below is only a homogeneous classical propagator.

Establish domain, grid, time, diffusion coefficient, initial condition, boundary values, and solver path. Preserve the one-dimensional grid/solution correspondence and inspect finite values and boundary behavior. The classical path includes `get_rhs_1d(...) + source(x)`; the current Trotter path does not forward either forcing contribution. Although `run()` accepts backend/device/dtype and passes them to `_solve_trotter`, that method does not forward them to `schro_trotter`.

## Overview

This reference describes the wrapper for $\partial u/\partial t = a \partial^2 u/\partial x^2 + f(x)$, subject to the path-specific forcing limits above. Auxiliary-variable lifting represents non-unitary diffusion through unitary evolution and recovery. The pipeline covers:

1. **Parse** the problem parameters (domain, diffusion coefficient, BC, initial condition)
2. **Discretize** the spatial domain using finite differences (central scheme)
3. **Schrödingerize** the non-unitary system into a Hamiltonian evolution problem
4. **Evolve** via classical exponentiation or Trotter splitting
5. **Visualize** the solution and export quantum circuit diagrams

## Reference Implementation Example

```python
from unitarylab_algorithms.schrodingerization.equation_heat.algorithm import HeatEquationAlgorithm

result = HeatEquationAlgorithm().run()
```


## Inputs and Outputs

The `run()` method dispatches to the classical, Trotter, or block solver and returns the solver's result dictionary directly. It does not call `_build_return_dict()`.

```python
{
    "status": "ok",
    "message": "Heat equation solved",
    "grid": {"n_points": ..., "dx": ..., "dt": ..., "nt": ...},
    "x": [...],
    "u": [...],
    "circuit": [...],
    "plot": {"format": "svg", "filename": "<solution_plot_path>"},
}
```

Additional solver-specific keys appended to the result:

| Key | Type | Description |
|-----|------|-------------|
| `x` | `list[float]` | Spatial grid points |
| `u` | `list[float]` | Solution values $u(x, T)$ |
| `grid.n_points` | `int` | $N_x = 2^{n_x}$ |
| `grid.dx` | `float` | Spatial step size |
| `grid.dt` | `float` | Time step (Trotter only) |
| `grid.nt` | `int` | Number of time steps (Trotter only) |

------

## Mathematical Deep Dive

### 1.1 1D Heat Equation

$$
\frac{\partial u}{\partial t} = a \frac{\partial^2 u}{\partial x^2} + f(x)
$$

- $a > 0$: diffusion coefficient
- $f(x)$: source term

### 1.2 Schrödingerized Hamiltonian

$$
\frac{d\psi}{dt} = -i H \psi
$$

with simplified Hamiltonian for periodic, source-free cases:
$$
H \approx -a \hat{\eta} \otimes \hat{p}^2 \approx a D_{\eta} \otimes D^\Delta
$$

- $\hat{\eta}$: auxiliary operator introduced by Schrödingerization
- $D_\eta$: discretized auxiliary operator
- $D^\Delta$: discrete Laplacian

This formula uses the `exp(-iHT)` convention. The SDK's `schro_classical` instead constructs `K=-D_eta⊗A1+I⊗A2` and applies `exp(+iKT)`, where `A1=(A+A†)/2` and `A2=-i(A-A†)/2`; thus `H=-K`. Preserve that sign distinction. Source and boundary forcing are additional terms, with the wrapper path limitations described above.

------

## Understanding the Key Quantum Components
- PDE Type: 1D parabolic diffusion
- Boundary Conditions: **Dirichlet**, **Periodic**
- Initial Conditions: sine, Gaussian, custom
- Solvers: Classical matrix exponentiation, Trotter splitting
- Automatic finite-difference Laplacian assembly
- Visualization + quantum circuit export

------

## Implementation Architecture
### Step 1: Parse Input Parameters

```python
from unitarylab.library.equation import parse_equation

eq = parse_equation(params)          # params: dict loaded from setup.json
method = eq.solver.type.lower()      # "classical" | "trotter" | "block"

# Common coefficients
L, T, source, nx, na, R, point, order, f0 = eq.get_common_coefficients()
bd     = eq.boundary.type            # "dirichlet" | "periodic" | "neumann"
scheme = eq.discrete.type
a      = eq.get_parameter('a')       # diffusion coefficient

# Grid
Nx = 2**nx
# Dirichlet (default)
dx = L / (Nx + 1)
x  = np.arange(dx, L, dx)
# Periodic override
if bd == "periodic":
    dx = L / Nx
    x  = np.arange(0, L, dx)
# Neumann override
elif bd == "neumann":
    dx = L / (Nx - 1)
    x  = np.arange(0, L + dx, dx)

u0 = f0(x)  # initial condition
```

------

### Step 2: Discretization

Construct 2nd-order differential operator using `CDiff` (classical) or `TDiff` (Trotter):

#### Classical

```python
from unitarylab.library.equation.differential_operator import CDiff

A = a * CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()
b = eq.get_rhs_1d(Nx, dx, scheme=scheme) + source(x)
```

#### Trotter

```python
from unitarylab.library.equation.differential_operator import TDiff

dt = eq.solver.dt
Nt = int(T / dt)

func1, func2 = (a * TDiff(nx, dx, 2, scheme=scheme, boundary=bd)).data()
H1 = func1(dt / R)   # auxiliary-space term
H2 = func2(dt)       # spatial-Laplacian term
```

------

### Step 3: Schrödingerization

Transform non-unitary diffusion equation:
$$
\frac{du}{dt} = A u + b \quad \longrightarrow \quad \frac{d\psi}{dt} = -i H \psi
$$

- Produces a Hermitian lifted evolution problem with a separate physical-solution recovery step
- Diffusion is non-unitary even with periodic, source-free boundaries; source and boundary terms require their own handling

------

### Step 4: Time Evolution

#### Classical Schrödinger Solver

```python
from unitarylab.library.equation.schrodingerization import schro_classical as schro
from unitarylab.library.equation.schrodingerization import circuit_classical

u  = schro(A, u0, T=T, na=na, R=R, order=order, point=point, b=b)
qc = circuit_classical(nx, na)
```

#### Trotterized Quantum Evolution

Split Hamiltonian $H = H_1 + H_2$ and apply Trotter decomposition:
$$
e^{-iHt} \approx \left(e^{-i H_1 \Delta t} e^{-i H_2 \Delta t}\right)^{N_t}
$$

```python
from unitarylab.library.equation.schrodingerization import schro_trotter as schro

u, qc = schro(
    u0=u0,
    H1=H1,
    H2=H2,
    Nt=Nt,
    na=na,
    R=R,
    order=order,
    point=point
)
```

------

### Step 5: Visualization

Internally called via `self._generate_solution_plot(name, x, u)` where `name` encodes solver metadata:

```python
# Classical
name = f"1D Heat Classical nx={nx} na={na} T={T}"
# Trotter
name = f"1D Heat Lie-Trotter nx={nx} na={na} T={T} dt={dt}"

solution_plot_path = self._generate_solution_plot(name, x, u)
```

- Generates 1D solution SVG
- Path returned in the `plot` field of the result dict

------

### Step 6: Quantum Circuit Export

Internally called via `self._generate_circuit_plots()`:

```python
# Classical  (qc only)
circuit_plot_paths = self._generate_circuit_plots(name, qc)

# Trotter  (qc + sub-Hamiltonians)
circuit_plot_paths = self._generate_circuit_plots(name, qc, H1, H2)
```

- Paths returned in the `circuit` field of the result dict
- Includes Trotter sub-circuit diagrams when H1/H2 are provided

------

## Boundary Conditions
- **Dirichlet:** $u(0,t)=0, u(L,t)=0$
- **Periodic:** $u(0,t) = u(L,t)$

------

## Initial Conditions
- Sine: $u(x,0) = \sin(2 \pi x / L)$
- Gaussian: $u(x,0) = \exp(-x^2)$
- Custom: user-defined $f_0(x)$

------

## Finite-Difference Formula
- **Central Difference** (2nd-order Laplacian):

$$
\partial_{xx}u_i \approx \frac{u_{i+1} - 2 u_i + u_{i-1}}{\Delta x^2}
$$

- Automatically assembled into matrix $A$

------

## Prerequisites

SDK documentation: [UnitaryLab](simulator-unitarylab.md#sources-and-version).

- PDE fundamentals: 1D heat/diffusion equation $\partial u/\partial t = a \partial^2 u/\partial x^2 + f(x)$
- Numerical methods: finite-difference Laplacian, Dirichlet/Periodic/Neumann BCs
- Schrödingerization theory: transforming non-unitary diffusion into unitary evolution via auxiliary variables
- Python: `numpy`, `unitarylab` core

## Core Parameters Explained

These are concepts resolved by the equation parser, not separate `run()` keyword arguments. The packaged setup schema (`unitarylab_algorithms/schrodingerization/equation_heat/setup.json`) uses a top-level `params` array with equation, method, `par_fix`, and `par_func` objects; `solver.type` below belongs to the parsed equation object.

| Parameter | Type | Description |
|---|---|---|
| `nx` | `int` | Grid exponent; $N_x = 2^{n_x}$ spatial points |
| `a` | `float` | Diffusion coefficient ($a > 0$) |
| `L` | `float` | Domain length $[0, L]$ |
| `T` | `float` | Final simulation time |
| `boundary` | `str` | `"dirichlet"`, `"periodic"`, or `"neumann"` |
| `scheme` | `str` | Discretization scheme (`"central"`) |
| `solver.type` | `str` | `"classical"`, `"trotter"`, or `"block"` (classical fallback) |
| `na` | `int` | Auxiliary qubit count for Schrödingerization |
| `R` | `int` | Schrödingerization scaling parameter |
| `order` | `int` | Warped-phase initialization order passed to `initial_schro_fp`; not a selectable Suzuki product-formula order |
| `dt` | `float` | Time step (Trotter only) |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` in a returned result; parsing, unsupported-method, and solver exceptions propagate |
| `x` | `list[float]` | Spatial grid coordinates |
| `u` | `list[float]` | Final solution $u(x, T)$ |
| `grid.n_points` | `int` | $N_x = 2^{n_x}$ |
| `grid.dx` | `float` | $\Delta x$ (depends on BC type) |
| `grid.dt` | `float` | Time step (Trotter only) |
| `grid.nt` | `int` | Number of time steps (Trotter only) |
| `circuit` | `list[str]` | Circuit SVG paths |
| `plot` | `dict` | `{"format":"svg", "filename":...}` for the solution plot; not the BaseAlgorithm list contract |

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Heat equation | `parse_equation(params)` extracts $a$, $L$, $T$, BC, initial condition |
| 2nd-order Laplacian matrix | `CDiff(Nx, dx, 2, scheme, boundary).get_matrix()` (classical) or `TDiff(nx, dx, 2, ...).data()` (Trotter) |
| Schrödingerized Hamiltonian $H = a D_\eta \otimes D^\Delta$ | $H_1$ (auxiliary) and $H_2$ (spatial Laplacian) from Trotter decomposition |
| Classical solver | `schro_classical(A, u0, T, na, R, order, point, b)` |
| Trotter evolution $e^{-iHt} \approx (e^{-iH_1\Delta t}e^{-iH_2\Delta t})^{N_t}$ | `schro_trotter(u0, H1, H2, Nt, na, R, order, point)` |
| Initial condition $u(x,0) = f_0(x)$ | `u0 = f0(x)` |

## Minimal Manual Implementation

```python
import numpy as np

def heat1d_skeleton(Nx, dx, a, T, boundary='dirichlet'):
    """Simplified 1D heat: build Laplacian and simulate.

    In practice, the implementation uses CDiff/TDiff and
    schro_classical/schro_trotter from unitarylab.library.
    """
    # Build 2nd-order Laplacian (central difference)
    A = np.zeros((Nx, Nx))
    for i in range(Nx):
        A[i, i] = -2.0
        if i > 0:
            A[i, i - 1] = 1.0
        if i < Nx - 1:
            A[i, i + 1] = 1.0

    if boundary == 'periodic':
        A[0, -1] += 1.0
        A[-1, 0] += 1.0

    A = a * A / (dx ** 2)

    # Dense matrix exponentiation (classical reference)
    from scipy.linalg import expm
    return expm(A * T)
```

## Debugging Tips

1. **Grid depends on BC**: Dirichlet: $dx = L/(N_x+1)$; Periodic: $dx = L/N_x$; Neumann: $dx = L/(N_x-1)$. Using the wrong grid formula shifts the solution domain.
2. **Source term handling**: Classical forcing is `eq.get_rhs_1d(...) + source(x)`. The current Trotter wrapper omits it; do not claim equivalent forced or nonzero-boundary solves across both paths.
3. **Trotter time step**: `Nt=int(T/dt)` gives effective duration `Nt*dt`; a non-integral ratio truncates the interval. The SDK composes the supplied component circuits in a fixed step; `order` controls warped-phase initialization rather than selecting a higher-order product formula. Grid, lifting, and recovery errors remain separate.
4. **Diffusion coefficient sign**: $a$ must be positive for physical diffusion. Negative $a$ produces anti-diffusion (exponentially unstable). Verify $a > 0$ in `setup.json`.
5. **Auxiliary dimension**: `na` controls a `2**na` auxiliary space. Compare recovery error under refinement with `R`, `point`, and initialization order fixed; increasing dimension alone does not establish convergence.
