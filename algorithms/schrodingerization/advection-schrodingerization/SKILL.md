---
name: advection-schrodingerization
description: A Schrödingerization-based solver for the 1D advection equation, supporting both direct unitary evolution (under periodic and central difference discretization) and general transformation for non-unitary cases. Enables classical and quantum simulation through Hamiltonian formulation.
---

## Entry Point

Use the implemented algorithm class first. It handles default parameter loading, equation parsing, solver dispatch, plotting, and circuit export.

```python
from unitarylab_algorithms.schrodingerization.equation_advection.algorithm import AdvectionEquationAlgorithm

algo = AdvectionEquationAlgorithm()
result = algo.run()  # loads equation_advection/setup.json when params is None

print(result["status"])
print(result["grid"])
print(result["plot"]["filename"])
```

To provide parameters explicitly, pass a `params` dict using the same schema as `unitarylab_algorithms/schrodingerization/equation_advection/setup.json`:

```python
import json
from pathlib import Path

from unitarylab_algorithms.schrodingerization.equation_advection.algorithm import AdvectionEquationAlgorithm

setup_path = Path("unitarylab_algorithms/schrodingerization/equation_advection/setup.json")
params = json.loads(setup_path.read_text(encoding="utf-8"))

algo = AdvectionEquationAlgorithm()
result = algo.run(params=params, backend="torch", device="cpu")
```

# Skill: Advection Equation Solver via Schrödingerization

## Skill Objective

Enable the agent to **solve the 1D Advection Equation** using a Schrödingerization-based framework, supporting both:

- Classical simulation
- Quantum (Trotterized) simulation

------

## Using the Provided Implementation

For normal use, import and call the algorithm class:

```python
from unitarylab_algorithms.schrodingerization.equation_advection.algorithm import AdvectionEquationAlgorithm

result = AdvectionEquationAlgorithm().run()
```

`AdvectionEquationAlgorithm.run()` accepts:
- `params`: parsed setup dict; when omitted, the implementation loads `equation_advection/setup.json`
- `algo_dir`: output directory; when omitted, results are written under `results/schrodingerization/equation_advection`
- `backend`, `device`, `dtype`: passed to the Trotter solver path

The lower-level snippets below describe the implementation internals and should not replace the class entry point.

## Mathematical Model

### Advection Equation

$$
\frac{\partial u}{\partial t}
=
a \frac{\partial u}{\partial x}
$$

- $a$: convection velocity
- Models transport phenomena (fluid flow, waves, etc.)

------

## Core Capabilities

### 1. PDE Parsing

Extract problem parameters:

- Domain $[0, L]$
- Final time $T$
- Grid size $N = 2^{n_x}$
- Boundary condition (Periodic / Dirichlet / others)
- Initial condition $u(x,0)$
- Velocity $a$

------

### 2. Spatial Discretization

Construct grid:
$$
x_i = i \Delta x, \quad \Delta x = \frac{L}{N}
$$
Build first-order derivative operator:

```python
A0, b0 = first_order_derivative(
    N=Nx,
    dx=dx,
    boundary_condition=bd,
    scheme=scheme
)
```

Construct system:
$$
A = a A_0, \quad b = a b_0
$$

------

### 3. Unitary Structure Detection

Agent must determine whether Schrödingerization is needed.

#### Case A: Already Unitary

If:

- Central difference scheme
- Periodic boundary

Then:
$$
H = -a \hat{p} \approx \frac{a D^{\pm}}{i}
$$
Properties:

- $A$ is skew-Hermitian
- Evolution is already unitary
- **Skip Schrödingerization**

------

#### Case B: Non-unitary System

If:

- Upwind scheme
- Dirichlet boundary
- Source term $b \neq 0$

Then:

- Apply full Schrödingerization procedure

------

### 4. Schrödingerization Transformation

Transform:
$$
\frac{du}{dt} = Au + b
\quad \Rightarrow \quad
\frac{d\psi}{dt} = -iH\psi
$$

#### If $A$ is Hermitian-compatible:

$$
H = iA
$$

#### Otherwise:

- Use auxiliary variable lifting
- Apply Fourier transform
- Construct Hamiltonian:

$$
H = \eta H_1 + H_2
$$

------

### 5. Hamiltonian Construction

Split operator:
$$
A = H_1 + iH_2
$$
Construct:

- $H_1 = \frac{A + A^\dagger}{2}$
- $H_2 = \frac{A - A^\dagger}{2i}$

Final Hamiltonian:
$$
H = D \otimes H_1 + I \otimes H_2
$$
The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

------

### 6. Time Evolution Methods

#### (A) Classical Schrödinger Solver

```python
from unitarylab.library.equation.schrodingerization import schro_classical as schro
from unitarylab.library.equation.schrodingerization import circuit_classical
from unitarylab.library.equation.differential_operator.classical_matrices import first_order_derivative

# Build derivative operator and system matrices
A0, b0 = first_order_derivative(
    N=Nx, dx=dx,
    boundary_condition=bd, scheme=scheme,
    g1=eq.boundary.left_value, g2=eq.boundary.right_value
)
A = A0 * a
b = b0 * a

# Solve and obtain circuit structure
u = schro(A, u0, T=T, na=na, R=R, order=order, point=point, b=b)
qc = circuit_classical(nx, na)
```

------

#### (B) Quantum Trotter Evolution

Hamiltonian splitting:
$$
H = H_1 + H_2
$$
Trotter approximation:
$$
e^{-iHt}
\approx
\left(e^{-iH_1 \Delta t} e^{-iH_2 \Delta t}\right)^{N_t}
$$

```python
from unitarylab.library.equation.schrodingerization import schro_trotter as schro
from unitarylab.library.equation.differential_operator import TDiff
from unitarylab.library.equation.differential_operator.classical_matrices import first_order_derivative as first_order_derivative_classical

# Build boundary term
A0, b0 = first_order_derivative_classical(
    N=Nx, dx=dx,
    boundary_condition=bd, scheme=scheme,
    g1=eq.boundary.left_value, g2=eq.boundary.right_value
)
b = b0 * a
b = b * T if T > 1 else b
theta = 1 / T if T > 1 else 1

# Build Hamiltonian components
if scheme == 'upwind':
    func1 = (abs(a) * TDiff(nx, dx, 2, boundary=bd)).data()[0]
    H1 = func1(dt / R)
elif scheme == 'central':
    H1 = None
func2 = (a * TDiff(nx, dx, 1, boundary=bd)).data()[1]
H2 = func2(dt)

# Execute Trotter evolution
u, qc = schro(
    u0=u0, H1=H1, H2=H2,
    Nt=Nt, na=na, R=R,
    order=order, point=point,
    b=b, theta=theta * dt
)
```

------

### 7. Output Generation

`AdvectionEquationAlgorithm.run()` dispatches to `_solve_classical` or `_solve_trotter`. Both paths generate the solution plot and circuit diagrams internally, then return a plain dictionary:

```python
{
    "status": "ok",
    "message": "Advection equation solved",
    "grid": {
        "n_points": 2**nx,
        "dx": dx,
        # Trotter only:
        "dt": dt,
        "nt": Nt,
    },
    "x": [...],
    "u": [...],
    "circuit": circuit_plot_paths,
    "plot": {
        "format": "svg",
        "filename": "<solution_plot_filename>",
    },
}
```

Key fields:
- `grid.n_points`: number of spatial grid points
- `grid.dx`: spatial step size
- `grid.dt` and `grid.nt`: present for the Trotter solver
- `x`: spatial grid coordinates
- `u`: final solution values $u(x,T)$ serialized as a list
- `circuit`: filenames returned by `_generate_circuit_plots(name, qc)`
- `plot.filename`: filename returned by `_generate_solution_plot(name, x, u)`

------

## Execution Workflow (Agent Pipeline)

1. Parse input parameters
2. Construct spatial grid
3. Build derivative operator $A_0$
4. Form system $A = aA_0$
5. Detect unitary structure
   - If unitary → direct evolution
   - Else → Schrödingerization
6. Construct Hamiltonian
7. Select solver:
   - Classical / Quantum
8. Perform time evolution
9. Recover solution
10. Output results

------

## Accuracy & Stability Considerations

- Grid size $n_x$ controls spatial accuracy
- Time step $\Delta t$ affects Trotter error
- Scheme choice:
  - Central → high accuracy, unitary
  - Upwind → stable but dissipative
- Schrödingerization introduces:
  - auxiliary dimension cost
  - smoother but higher-dimensional system

------

## Usage Interface

```python
from unitarylab_algorithms.schrodingerization.equation_advection.algorithm import AdvectionEquationAlgorithm

algo = AdvectionEquationAlgorithm()
result = algo.run(params)
```

------

## Skill Trigger Conditions

Activate when:

- PDE contains:
  - first-order spatial derivative
  - transport/advection structure
- Keywords:
  - “advection equation”
  - “transport equation”
  - “hyperbolic PDE”
  - “Schrödingerization + PDE”

------

## Summary

This skill implements a **hybrid computational framework**:
$$
\text{Advection PDE}
\;\rightarrow\;
\text{Discretization}
\;\rightarrow\;
\text{(Optional) Schrödingerization}
\;\rightarrow\;
\text{Unitary Evolution}
$$
It bridges:

- Classical numerical PDE methods
- Quantum Hamiltonian simulation
