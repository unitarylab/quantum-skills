---
name: heat-2d-schrodingerization
description: A quantum-compatible solver for the 2D Heat Equation using Schrödingerization to transform the non-unitary diffusion equation into a unitary evolution problem. Supports anisotropic diffusion, Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution with automatic 2D circuit generation and 3D temperature field visualization.
---

## Entry Point

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

# Skill: 2D Heat Equation (Schrödingerization-based Solver)

## Mathematical Model

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

## Supported Features

- 2D anisotropic heat conduction
- Boundary conditions: **Dirichlet**, **Periodic**
- Initial conditions: 2D sine wave, custom
- Quantum solvers: Classical matrix exponentiation, Trotter splitting, Block encoding (fallback)
- Tensor-product finite-difference discretization
- 3D surface / contour visualization
- Automatic quantum circuit generation

------

## Full Algorithm Pipeline (Step-by-Step)

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

## Boundary & Initial Conditions

- **Dirichlet**: $u=0$ on domain boundary
- **Periodic**: $u(0,y)=u(L,y), u(x,0)=u(x,L)$
- **Initial Condition**: 2D sine wave:

$$
u(x,y,0) = \sin\left(\frac{\pi x}{L_x}\right) \sin\left(\frac{\pi y}{L_y}\right)
$$

------

## Finite-Difference Scheme

2D central difference for Laplacian:
$$
\Delta u_{i,j} = \Delta t \left[ a_1 \frac{u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}}{\Delta x^2} + a_2 \frac{u_{i,j+1} - 2 u_{i,j} + u_{i,j-1}}{\Delta y^2} \right]
$$

------

## Outputs

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

## Trigger Phrases

- Solve 2D heat equation using quantum Schrödingerization
- Quantum simulation for planar heat conduction
- 2D thermal analysis
- Quantum PDE solver for heat conduction

------

## Use Cases

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
