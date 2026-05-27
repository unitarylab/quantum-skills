---
name: heat-1d-schrodingerization
description: A quantum-compatible solver for the 1D Heat Equation using Schrödingerization to transform the non-unitary diffusion equation into a unitary evolution problem. Supports Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution with automatic circuit generation and solution visualization.
---

## One-Step Run Example Command
```bash
python ./scripts/algorithm.py
```

## Entry Point

```python
from unitarylab_algorithms.schrodingerization.equation_heat.algorithm import HeatEquationAlgorithm

result = HeatEquationAlgorithm().run()
```

# Skill: Quantum Simulation of 1D Heat Equation

## 1. Mathematical Formulation

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

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

## 2. Supported Features

- PDE Type: 1D parabolic diffusion
- Boundary Conditions: **Dirichlet**, **Periodic**
- Initial Conditions: sine, Gaussian, custom
- Solvers: Classical matrix exponentiation, Trotter splitting
- Automatic finite-difference Laplacian assembly
- Visualization + quantum circuit export

------

## 3. Full Algorithm Pipeline (Step-by-Step)

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

## 4. Boundary Conditions

- **Dirichlet:** $u(0,t)=0, u(L,t)=0$
- **Periodic:** $u(0,t) = u(L,t)$

------

## 5. Initial Conditions

- Sine: $u(x,0) = \sin(2 \pi x / L)$
- Gaussian: $u(x,0) = \exp(-x^2)$
- Custom: user-defined $f_0(x)$

------

## 6. Finite-Difference Scheme

- **Central Difference** (2nd-order Laplacian):

$$
\Delta u_i \approx \frac{u_{i+1} - 2 u_i + u_{i-1}}{\Delta x^2} \Delta t
$$

- Automatically assembled into matrix $A$

------

## 7. Outputs

The `run()` method returns a dict built by `_build_return_dict(success, circuit_path, filepath, circuit)`:

```python
{
    "status":       "ok" | "failed",
    "circuit_path": circuit_path,          # path(s) to circuit diagram file(s)
    "plot": [
        {"format": "svg", "filename": "<solution_plot_path>"},
        ...                                # one entry per filepath
    ],
    "circuit":      circuit_plot_paths,    # list of circuit SVG paths
    # …plus any extra keys from self.output
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

## 8. Trigger Phrases

- Quantum simulation of 1D heat equation
- Schrödingerization-based solver for diffusion PDE
- Trotter quantum evolution of parabolic PDE

------

## 9. Use Cases

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