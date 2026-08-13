---
name: heat-1d-schrodingerization
description: "A quantum-compatible solver for the 1D Heat Equation using Schrödingerization to transform the non-unitary diffusion equation into a unitary evolution problem. Supports Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution with automatic circuit generation and solution visualization. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---


# Skill: Quantum Simulation of 1D Heat Equation

## Overview

This skill provides a Schrödingerization-based solver for the 1D Heat Equation $\partial u/\partial t = a \partial^2 u/\partial x^2 + f(x)$. The non-unitary diffusion dynamics are transformed into unitary evolution via auxiliary-variable lifting, enabling both classical matrix exponentiation and Trotterized quantum simulation. The pipeline covers:

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

> Note: General source or non-periodic BC requires full Schrödingerization.

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

- Ensures unitary evolution
- Required for general source or non-periodic BC

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
\Delta u_i \approx \frac{u_{i+1} - 2 u_i + u_{i-1}}{\Delta x^2} \Delta t
$$

- Automatically assembled into matrix $A$

------

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Quantum Simulation of 1D Heat Equation.

- Quantum simulation of 1D heat equation
- Schrödingerization-based solver for diffusion PDE
- Trotter quantum evolution of parabolic PDE

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.
- **PDE configuration:** Confirm the spatial domain, grid size, time interval, diffusion coefficients, initial condition, and boundary condition before running the solver.
- **Solver path:** Distinguish classical Schrödingerization from Trotterized quantum simulation and preserve the selected execution path.
- **One-dimensional shape:** Preserve the mapping between the one-dimensional grid and the returned state or solution vector.
- **Result validation:** Verify grid dimensions, solution shape, finite numerical values, and boundary-condition behavior before interpreting plots or circuit outputs.

## Hands-On Example
- 1D diffusion and conduction problems
- Heat transfer simulations
- Benchmarking quantum PDE algorithms
- Educational demonstrations of Schrödingerization

------

### Summary

- Standardized **step-by-step quantum simulation** pipeline for 1D Heat Equation
- Handles Dirichlet/Periodic BCs and custom initial conditions
- Supports classical and Trotterized quantum evolution
- Automates Laplacian assembly, visualization, and circuit export
- Fully consistent with **Advection / Burgers / General Linear PDE / Elastic Wave** skill style

## Prerequisites

- PDE fundamentals: 1D heat/diffusion equation $\partial u/\partial t = a \partial^2 u/\partial x^2 + f(x)$
- Numerical methods: finite-difference Laplacian, Dirichlet/Periodic/Neumann BCs
- Schrödingerization theory: transforming non-unitary diffusion into unitary evolution via auxiliary variables
- Python: `numpy`, `unitarylab` core

## Core Parameters Explained

| Parameter | Type | Description |
|---|---|---|
| `nx` | `int` | Grid exponent; $N_x = 2^{n_x}$ spatial points |
| `a` | `float` | Diffusion coefficient ($a > 0$) |
| `L` | `float` | Domain length $[0, L]$ |
| `T` | `float` | Final simulation time |
| `boundary` | `str` | `"dirichlet"`, `"periodic"`, or `"neumann"` |
| `scheme` | `str` | Discretization scheme (`"central"`) |
| `solver.type` | `str` | `"classical"` or `"trotter"` |
| `na` | `int` | Auxiliary qubit count for Schrödingerization |
| `R` | `int` | Schrödingerization scaling parameter |
| `order` | `int` | Trotter expansion order |
| `dt` | `float` | Time step (Trotter only) |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` or `"failed"` |
| `x` | `list[float]` | Spatial grid coordinates |
| `u` | `list[float]` | Final solution $u(x, T)$ |
| `grid.n_points` | `int` | $N_x = 2^{n_x}$ |
| `grid.dx` | `float` | $\Delta x$ (depends on BC type) |
| `grid.dt` | `float` | Time step (Trotter only) |
| `grid.nt` | `int` | Number of time steps (Trotter only) |
| `circuit` | `list[str]` | Circuit SVG paths |
| `plot` | `list[dict]` | Saved plot metadata |

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
        A[0, -1] = 1.0
        A[-1, 0] = 1.0

    A = a * A / (dx ** 2)

    # Dense matrix exponentiation (classical reference)
    from scipy.linalg import expm
    return expm(A * T)
```

## Debugging Tips

1. **Grid depends on BC**: Dirichlet: $dx = L/(N_x+1)$; Periodic: $dx = L/N_x$; Neumann: $dx = L/(N_x-1)$. Using the wrong grid formula shifts the solution domain.
2. **Source term handling**: Non-zero $f(x)$ requires full Schrödingerization even for periodic BC. If results are unexpected with a source term, verify `b = source(x)` is correctly assembled.
3. **Trotter time step**: $\Delta t = T/N_t$. The Trotter error scales as $O(\Delta t^2)$ for order-2. Too-large $\Delta t$ causes instability in the diffusion solve.
4. **Diffusion coefficient sign**: $a$ must be positive for physical diffusion. Negative $a$ produces anti-diffusion (exponentially unstable). Verify $a > 0$ in `setup.json`.
5. **Auxiliary dimension**: `na` controls the Schrödingerization ancilla register. The effective auxiliary space scales as $2^{na}$ — increase if the solution shows truncation artifacts.
