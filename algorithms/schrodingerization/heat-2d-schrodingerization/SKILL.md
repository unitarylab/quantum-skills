---
name: heat-2d-schrodingerization
description: "A quantum-compatible solver for the 2D Heat Equation using Schrödingerization to transform the non-unitary diffusion equation into a unitary evolution problem. Supports anisotropic diffusion, Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution with automatic 2D circuit generation and 3D temperature field visualization. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---


# Skill: 2D Heat Equation (Schrödingerization-based Solver)

## Overview

### 2D Heat Equation

$$
\frac{\partial u}{\partial t} = a_1 \frac{\partial^2 u}{\partial x^2} + a_2 \frac{\partial^2 u}{\partial y^2} + f(x,y)
$$

- $a_1, a_2$: diffusion coefficients in x/y directions
- $u(x,y,t)$: temperature field
- $f(x,y)$: source term

### Schrödingerized Hamiltonian (Periodic Form)

$$
H = - \hat{\eta} \otimes \left( a_1 \hat{p}_x^2 + a_2 \hat{p}_y^2 \right)
$$

Valid if:

1. Discrete derivative operator is Hermitian
2. Periodic BC in both x/y directions
3. Source term $f(x,y) = 0$

Otherwise, full Schrödingerization procedure is required.

------

## Reference Implementation Example

Use the provided algorithm class first. It owns parameter parsing, solver dispatch, plotting, and circuit export.

```python
from unitarylab_algorithms.schrodingerization.equation_heat2d.algorithm import Heat2dEquationAlgorithm

algo = Heat2dEquationAlgorithm()
result = algo.run()  # loads equation_heat2d/setup.json when params is None

print(result["status"])
print(result["grid"])
print(result["plot"]["filename"])
```

To provide parameters explicitly, pass a `params` dict using the same schema as `unitarylab_algorithms/schrodingerization/equation_heat2d/setup.json`:

```python
import json
from pathlib import Path

from unitarylab_algorithms.schrodingerization.equation_heat2d.algorithm import Heat2dEquationAlgorithm

setup_path = Path("unitarylab_algorithms/schrodingerization/equation_heat2d/setup.json")
params = json.loads(setup_path.read_text(encoding="utf-8"))

algo = Heat2dEquationAlgorithm()
result = algo.run(params=params, backend="torch", device="cpu")
```


## Inputs and Outputs

`Heat2dEquationAlgorithm.run()` dispatches to `_solve_classical`, `_solve_trotter`, or `_solve_block`. The block path currently logs a fallback message and calls the classical solver. Classical and Trotter paths return dictionaries with this shape:

```python
{
    "status": "ok",
    "message": "2D Heat equation solved",
    "grid": {
        "n_points": 2**nx,
        "dx": dx,
        # Trotter only:
        "dt": dt,
        "nt": Nt,
    },
    "x": [...],
    "y": [...],
    "u": [[...], ...],
    "circuit": circuit_plot_paths,
    "plot": {
        "format": "svg",
        "filename": "<solution_plot_filename>",
    },
}
```

Key fields:
- `grid.n_points`: number of grid points per spatial dimension
- `grid.dx`: spatial step size
- `grid.dt` and `grid.nt`: present for the Trotter solver
- `x`, `y`: spatial grids
- `u`: final 2D solution array serialized as nested lists
- `circuit`: filenames returned by `_generate_circuit_plots(name, qc, ...)`
- `plot.filename`: filename returned by `_generate_solution_plot(name, x, y, u)`

------

## Implementation Architecture

### Step 0: Import the Algorithm Class

For normal use, import and call the implemented class:

```python
from unitarylab_algorithms.schrodingerization.equation_heat2d.algorithm import Heat2dEquationAlgorithm

result = Heat2dEquationAlgorithm().run()
```

The lower-level imports below are implementation details used inside `Heat2dEquationAlgorithm`:

```python
# import parser
from unitarylab.library.equation import parse_equation

# import circuit builder
from unitarylab import Circuit

# import classical solver and circuit generator
from unitarylab.library.equation.schrodingerization import schro_classical
from unitarylab.library.equation.schrodingerization import circuit_classical

# import Trotter solver
from unitarylab.library.equation.schrodingerization import schro_trotter

# import finite-difference operators
from unitarylab.library.equation.differential_operator import CDiff   # classical matrix
from unitarylab.library.equation.differential_operator import TDiff   # Trotter unitary

import scipy.sparse as sp
import numpy as np
```

### Step 1: Parse 2D Domain & Parameters

Extract coefficients, domain, grid, qubits, boundary type. `eq` is the object returned by `parse_equation(params)`:

```python
# Parse equation object from params dict / JSON path
eq = parse_equation(params)

a1 = eq.get_parameter('a1')
a2 = eq.get_parameter('a2')
L, T, source, nx, na, R, point, order, f0 = eq.get_common_coefficients()
bd   = eq.boundary.type
scheme = eq.discrete.type

Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
y = np.arange(dx, L, dx)

# Periodic BC uses different grid
if bd == 'periodic':
    dx = L / Nx
    x = np.arange(0, L, dx)
    y = np.arange(0, L, dx)
```

------

### Step 2: Initialize 2D Temperature Field

```python
u0 = f0(x[:, None], y[None, :])  # 2D initial condition
u0 = u0.flatten()                # flatten for solver
```

------

### Step 3: Assemble 2D Laplacian Matrix (Kronecker Product)

```python
A0 = CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()
A = a1 * sp.kron(A0, sp.eye(Nx)) + a2 * sp.kron(sp.eye(Nx), A0)
```

- `kron(A0, I)`: x-direction Laplacian
- `kron(I, A0)`: y-direction Laplacian

------

### Step 4: Build Source Term

```python
b0 = source(x)
b = a1 * np.kron(b0, np.ones(Nx)) + a2 * np.kron(np.ones(Nx), b0)
```

------

### Step 5: Schrödingerization Solver (Classical Method)

```python
u = schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point, b=b)
u = u.reshape((Nx, Nx))          # reshape back to 2D field
qc = circuit_classical(nx, na, dim=2)
```

`point` selects the Fourier-basis evaluation point for the ancilla register. The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

### Step 6: Trotter Quantum Circuit (Optional)

```python
# Trotter-specific extra parameters
dt = eq.solver.dt
Nt = int(T / dt)

# Neumann BC uses yet another grid
if bd == 'neumann':
    dx = L / (Nx - 1)
    x = np.arange(0, L + dx, dx)
    y = np.arange(0, L + dx, dx)

# Build per-direction Trotter unitary factory
func1 = TDiff(nx, dx, 2, scheme=scheme, boundary=bd).data()[0]
D1 = lambda a: func1(a * dt / R)

# Compose 2D Hamiltonian (2*nx qubits: first nx for x, next nx for y)
H1 = Circuit(2 * nx)
H1.append(D1(a1), range(nx))           # x-direction block
H1.append(D1(a2), range(nx, 2 * nx))   # y-direction block
H2 = None
```

------

### Step 7: Run Trotter Evolution

```python
u, qc = schro_trotter(
    u0=u0, H1=H1, H2=H2,
    Nt=Nt, na=na, R=R,
    order=order, point=point,
    device=device,          # e.g. 'cpu'
)
u = u.reshape((Nx, Nx))
```

------

### Step 8: Visualization via `_generate_solution_plot`

```python
# Called internally; returns the saved filename (relative to algo_dir)
name = f"2D Heat Classical nx={nx} na={na} T={T}"
solution_plot_path = self._generate_solution_plot(name, x, y, u)

# The method builds a 3D surface using meshgrid:
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='viridis')
```

------

## Understanding the Key Quantum Components

- 2D anisotropic heat conduction
- Boundary conditions: **Dirichlet**, **Periodic**
- Initial conditions: 2D sine wave, custom
- Quantum solvers: Classical matrix exponentiation, Trotter splitting, Block encoding (fallback)
- Tensor-product finite-difference discretization
- 3D surface / contour visualization
- Automatic quantum circuit generation

------

## Mathematical Deep Dive

- **Dirichlet**: $u=0$ on domain boundary
- **Periodic**: $u(0,y)=u(L,y), u(x,0)=u(x,L)$
- **Initial Condition**: 2D sine wave:

$$
u(x,y,0) = \sin\left(\frac{\pi x}{L_x}\right) \sin\left(\frac{\pi y}{L_y}\right)
$$

------

2D central difference for Laplacian:
$$
\Delta u_{i,j} = \Delta t \left[ a_1 \frac{u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}}{\Delta x^2} + a_2 \frac{u_{i,j+1} - 2 u_{i,j} + u_{i,j-1}}{\Delta y^2} \right]
$$

------

## Hands-On Example

- Chip / PCB thermal simulation
- 2D material heat conduction
- Thin-plate temperature analysis
- Quantum PDE benchmarking

------

### Summary

This skill provides a **complete quantum solution for the 2D heat equation**:

- Uses Kronecker product for 2D Laplacian
- Supports anisotropic diffusion in x/y directions
- Works with classical, Trotter, and block solvers
- Automatically reshapes to 2D field
- Generates professional 3D visualizations
- Fully aligned with your implementation and mathematical framework

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the 2D heat-equation Schrödingerization solver.

- Solve 2D heat equation using quantum Schrödingerization
- Quantum simulation for planar heat conduction
- 2D thermal analysis
- Quantum PDE solver for heat conduction

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.
- **PDE configuration:** Confirm both spatial domains, grid sizes, time interval, diffusion coefficients, initial field, and boundary conditions before running the solver.
- **Solver path:** Distinguish classical Schrödingerization from Trotterized quantum simulation and preserve the selected execution path.
- **Two-dimensional shape:** Preserve the flattening, axis ordering, and reshape conventions between the 2D field and the solver state vector.
- **Operator construction:** Preserve the Kronecker-product construction of the 2D Laplacian and verify its dimensions.
- **Result validation:** Verify grid dimensions, reshaped solution, finite numerical values, and boundary-condition behavior before interpreting plots or circuit outputs.

## Prerequisites

- PDE fundamentals: 2D heat equation $\partial u/\partial t = a_1 \partial^2 u/\partial x^2 + a_2 \partial^2 u/\partial y^2 + f(x,y)$
- Numerical methods: 2D finite-difference Laplacian via Kronecker products
- Schrödingerization theory: extending the 1D auxiliary-variable method to 2D tensor-product domains
- Python: `numpy`, `scipy.sparse`, `unitarylab` core

## Core Parameters Explained

| Parameter | Type | Description |
|---|---|---|
| `nx` | `int` | Grid exponent per dimension; $N_x = N_y = 2^{n_x}$ |
| `a1`, `a2` | `float` | Diffusion coefficients in x/y directions (anisotropic) |
| `L` | `float` | Domain length $[0, L] \times [0, L]$ |
| `T` | `float` | Final simulation time |
| `boundary` | `str` | `"dirichlet"` or `"periodic"` |
| `scheme` | `str` | Discretization scheme (`"central"`) |
| `solver.type` | `str` | `"classical"`, `"trotter"`, or `"block"` (block encoder falls back to classical) |
| `na` | `int` | Auxiliary qubit count |
| `R` | `int` | Schrödingerization scaling parameter |
| `order` | `int` | Trotter order |
| `dt` | `float` | Time step (Trotter only) |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` or `"failed"` |
| `message` | `str` | Descriptive status message |
| `grid.n_points` | `int` | Grid points per dimension |
| `grid.dx` | `float` | $\Delta x = \Delta y$ |
| `grid.dt` | `float` | Time step (Trotter only) |
| `grid.nt` | `int` | Number of time steps (Trotter only) |
| `x` | `list[float]` | x-coordinate grid |
| `y` | `list[float]` | y-coordinate grid |
| `u` | `list[list[float]]` | Final 2D solution $u(x, y, T)$ as nested lists |
| `circuit` | `list[str]` | Circuit SVG paths |
| `plot.filename` | `str` | 3D surface plot SVG path |

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| 2D heat PDE | `parse_equation(params)` — extracts $a_1$, $a_2$, BC |
| 1D Laplacian $A_0$ | `CDiff(Nx, dx, 2, scheme, boundary).get_matrix()` |
| 2D Laplacian via Kronecker | `A = a1 * kron(A0, I) + a2 * kron(I, A0)` |
| 2D initial condition | `u0 = f0(x[:, None], y[None, :]).flatten()` |
| Classical solver | `schro_classical(A, u0, T, na, R, order, point, b)` |
| Trotter 2D Hamiltonian | `H1` assembled from per-direction `TDiff` blocks on $2 n_x$ qubits |
| Solution reshape | `u.reshape((Nx, Nx))` after solver |

## Minimal Manual Implementation

```python
import numpy as np
from scipy.sparse import kron, eye
from scipy.linalg import expm

def heat2d_skeleton(Nx, dx, a1, a2, T, boundary='dirichlet'):
    """Simplified 2D heat: Kronecker Laplacian and dense evolution."""
    # Build 1D Laplacian
    A0 = np.zeros((Nx, Nx))
    for i in range(Nx):
        A0[i, i] = -2.0
        if i > 0: A0[i, i-1] = 1.0
        if i < Nx-1: A0[i, i+1] = 1.0
    if boundary == 'periodic':
        A0[0, -1] = 1.0; A0[-1, 0] = 1.0
    A0 = A0 / (dx ** 2)

    # 2D Laplacian via Kronecker products
    I = np.eye(Nx)
    A = a1 * np.kron(A0, I) + a2 * np.kron(I, A0)
    return expm(A * T)
```

## Debugging Tips

1. **Kronecker order matters**: `kron(A0, I)` applies the Laplacian in the x-direction; `kron(I, A0)` in y. Swapping them silently transposes the diffusion tensor.
2. **Anisotropic coefficients**: $a_1 \neq a_2$ produces direction-dependent diffusion. If the solution looks unexpectedly directional, verify $a_1$ and $a_2$ are assigned to the correct Kronecker term.
3. **2D reshape**: The solver operates on flattened vectors of length $N_x^2$. Always reshape back to `(Nx, Nx)` before visualization — failing to reshape produces garbled 3D plots.
4. **Block solver fallback**: The `"block"` solver currently falls back to the classical path with a log message. Do not rely on it for quantum speedup — use `"trotter"` for quantum evolution.
5. **Memory scaling**: The 2D Laplacian is $N_x^2 \times N_x^2$. For $n_x = 5$ ($N_x = 32$), the matrix is $1024 \times 1024$. For $n_x = 7$ ($N_x = 128$), it's $16384 \times 16384$ — sparse matrix methods are essential.
