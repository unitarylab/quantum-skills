---
name: advection-schrodingerization
description: A Schrödingerization-based solver for the 1D advection equation, supporting both direct unitary evolution (under periodic and central difference discretization) and general transformation for non-unitary cases. Enables classical and quantum simulation through Hamiltonian formulation.
---

## One-Step Run Example Command
```bash
python ./scripts/algorithm.py
```

# Skill: Advection Equation Solver via Schrödingerization

## Skill Objective

Enable the agent to **solve the 1D Advection Equation** using a Schrödingerization-based framework, supporting both:

- Classical simulation
- Quantum (Trotterized) simulation

------

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

After solving, populate `self.output` with result data, then call `_build_return_dict()` to assemble the return value:

```python
# Store solution data into self.output (merged into return dict automatically)
self.output.update({
    "message": "Advection equation solved",
    "grid": {"n_points": Nx, "dx": dx},
    "x": x.tolist(),
    "u": u.tolist(),
})

# Generate plots and circuit diagrams
name = f"1D_Advection_Classical_nx={nx}_na={na}_T={T}"
solution_plot_path = self._generate_solution_plot(name, x, u)
circuit_plot_paths = self._generate_circuit_plots(name, qc)

# Build return dict using base class helper
result = self._build_return_dict(
    success=True,
    circuit_path=circuit_plot_paths,
    filepath=[solution_plot_path],
    circuit=qc
)
```

Return value keys (from `_build_return_dict`):

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` or `'failed'` |
| `circuit_path` | `list` | Paths to circuit diagram files |
| `plot` | `list[dict]` | `[{"format": "svg", "filename": ...}]` |
| `circuit` | object | Circuit object (`qc`) |
| `message` | `str` | Human-readable result message (from `self.output`) |
| `grid` | `dict` | `{"n_points": Nx, "dx": dx}` — spatial grid info (from `self.output`) |
| `x` | `list` | Grid point coordinates (from `self.output`) |
| `u` | `list` | Solution values $u(x,T)$ (from `self.output`) |

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