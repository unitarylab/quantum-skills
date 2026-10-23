
# 2D Heat Equation via Schrödingerization

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The `Heat2dEquationAlgorithm` source contract (`unitarylab_algorithms/schrodingerization/equation_heat2d/algorithm.py`) delegates to UnitaryLab equation SDK helpers. `../scripts/heat_2d_schrodingerization_algorithm.py` shows its wrapper call. Its `block` mode logs a classical fallback.

The wrapper uses a square `[0,L]×[0,L]` domain with the same `Nx=2**nx` points on each axis; it does not expose independent axis domains or grid widths. Preserve C-order flattening, axis assignment, the Kronecker Laplacian, and `(Nx,Nx)` recovery. Inspect grid lengths, finite values, and boundary behavior.

Classical forcing is specifically `a1*kron(source(x),ones(Nx)) + a2*kron(ones(Nx),source(x))`; this is not an arbitrary two-variable source API. The current Trotter path forwards no forcing and passes only `device` from its backend/device/dtype arguments. Neumann grid handling exists only in the Trotter branch; do not assume identical boundary coverage across paths.

There is also a source axis discrepancy: C-order flattening makes the `y` index occupy the lower `nx` qubits, but the Trotter wrapper appends the `a1` block there and the `a2` block on the upper register. The classical Kronecker construction assigns `a1` to the first (`x`) axis. For anisotropic coefficients, report this discrepancy rather than silently swapping inputs or claiming equivalent paths.

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

This is a theoretical periodic, source-free form. The SDK's `schro_classical` builds `K=-D_eta⊗A1+I⊗A2` and evolves with `exp(+iKT)`, so the conventional `exp(-iHT)` Hamiltonian is `H=-K`. The wrapper delegates lifting and recovery to that SDK; diffusion still requires lifting in the periodic source-free case.

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
from importlib.resources import files

from unitarylab_algorithms.schrodingerization.equation_heat2d.algorithm import Heat2dEquationAlgorithm

params = json.loads(files(
    "unitarylab_algorithms.schrodingerization.equation_heat2d"
).joinpath("setup.json").read_text(encoding="utf-8"))

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

### SDK Components

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

`point` selects the Fourier-basis evaluation point for the ancilla register.
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

# Preserve the source register assignment; see the axis discrepancy above.
H1 = Circuit(2 * nx)
H1.append(D1(a1), range(nx))           # lower-register block
H1.append(D1(a2), range(nx, 2 * nx))   # upper-register block
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

This reproduces the source plot. Its default `meshgrid(x,y)` uses `xy` indexing, while the returned field was built with first axis `x` and second axis `y`; for nonsymmetric fields, the plotted axis labels can therefore transpose the returned data's interpretation. Do not use that surface alone to validate axis ordering.

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

For a first-order time increment, the anisotropic central-difference formula is:
$$
u_{i,j}(t+\Delta t)-u_{i,j}(t) \approx \Delta t \left[ a_1 \frac{u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}}{\Delta x^2} + a_2 \frac{u_{i,j+1} - 2 u_{i,j} + u_{i,j-1}}{\Delta y^2} \right]
$$

------

## Prerequisites

SDK documentation: [UnitaryLab](simulator-unitarylab.md#sources-and-version).

- PDE fundamentals: 2D heat equation $\partial u/\partial t = a_1 \partial^2 u/\partial x^2 + a_2 \partial^2 u/\partial y^2 + f(x,y)$
- Numerical methods: 2D finite-difference Laplacian via Kronecker products
- Schrödingerization theory: extending the 1D auxiliary-variable method to 2D tensor-product domains
- Python: `numpy`, `scipy.sparse`, `unitarylab` core

## Core Parameters Explained

These are parsed concepts, not separate `run()` keyword arguments. The packaged setup schema (`unitarylab_algorithms/schrodingerization/equation_heat2d/setup.json`) uses a top-level `params` array with equation/method and `par_fix`/`par_func` records; `solver.type` below is a parsed-equation field.

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
| `order` | `int` | Warped-phase initialization order passed to `initial_schro_fp`; not a selectable Suzuki product-formula order |
| `dt` | `float` | Time step (Trotter only) |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` in a returned result; parsing, unsupported-method, and solver exceptions propagate |
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
        A0[0, -1] += 1.0; A0[-1, 0] += 1.0
    A0 = A0 / (dx ** 2)

    # 2D Laplacian via Kronecker products
    I = np.eye(Nx)
    A = a1 * np.kron(A0, I) + a2 * np.kron(I, A0)
    return expm(A * T)
```

## Debugging Tips

1. **Kronecker order matters**: `kron(A0, I)` applies the Laplacian in the x-direction; `kron(I, A0)` in y. Swapping them silently transposes the diffusion tensor.
2. **Anisotropic coefficients**: Compare classical Kronecker axes, Trotter register placement, and the plot's `xy` meshgrid convention separately. The source discrepancies above can be hidden when `a1==a2` or the field is symmetric.
3. **2D reshape**: The solver operates on flattened vectors of length $N_x^2$. Always reshape back to `(Nx, Nx)` before visualization — failing to reshape produces garbled 3D plots.
4. **Solver comparisons**: `"block"` is a classical fallback. Trotter mode is simulated circuit evolution and provides no demonstrated speedup. It uses `Nt=int(T/dt)`; compare at `Nt*dt` and account for its omitted forcing before comparing classical results.
5. **Memory scaling**: The 2D Laplacian is $N_x^2 \times N_x^2$. For $n_x = 5$ ($N_x = 32$), the matrix is $1024 \times 1024$. For $n_x = 7$ ($N_x = 128$), it's $16384 \times 16384$ — sparse matrix methods are essential.
