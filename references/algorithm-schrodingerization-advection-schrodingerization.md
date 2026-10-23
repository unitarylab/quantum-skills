
# Advection Equation Solver via Schrödingerization

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The `AdvectionEquationAlgorithm` source contract (`unitarylab_algorithms/schrodingerization/equation_advection/algorithm.py`) dispatches classical or Trotter paths using `unitarylab.library.equation` SDK helpers. `../scripts/advection_schrodingerization_algorithm.py` shows the wrapper call; lower-level snippets describe its internals.

Establish the domain, boundary values, grid, time, coefficient, initial condition, and selected solver. Preserve one-dimensional grid/solution alignment and inspect finite values and boundary behavior. The wrapper assembles `A=a*A0` and boundary forcing `b=a*b0`; it reads but does not add a general configured `source(x)`. Its module docstring uses the opposite conventional advection sign, so check transport direction against an analytical translated profile before interpreting `a` as a physical velocity.

## Overview

### Advection Equation

$$
\frac{\partial u}{\partial t}
=
a \frac{\partial u}{\partial x}
$$

- $a$: coefficient in the displayed PDE; for $u_t=a u_x$, the translation is $u(x,t)=u_0(x+at)$
- Models transport phenomena (fluid flow, waves, etc.)

------

## Reference Implementation Example

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
from importlib.resources import files

from unitarylab_algorithms.schrodingerization.equation_advection.algorithm import AdvectionEquationAlgorithm

params = json.loads(files(
    "unitarylab_algorithms.schrodingerization.equation_advection"
).joinpath("setup.json").read_text(encoding="utf-8"))

algo = AdvectionEquationAlgorithm()
result = algo.run(params=params, backend="torch", device="cpu")
```


`AdvectionEquationAlgorithm.run()` accepts:
- `params`: parsed setup dict; when omitted, the implementation loads `equation_advection/setup.json`
- `algo_dir`: output directory; when omitted, results are written under `results/schrodingerization/equation_advection`
- `backend`, `device`, `dtype`: passed to the Trotter solver path

------

## Implementation Architecture

1. Load and parse parameters, then dispatch the classical or Trotter path.
2. Construct the boundary-dependent spatial grid and initial vector.
3. On the classical path, build `A=a*A0` and `b=a*b0`; on the Trotter path, construct component circuits and the scaled boundary vector.
4. Call `schro_classical` or `schro_trotter`, which performs evolution and physical-solution recovery.
5. Save solution/circuit diagrams and return grid and one-dimensional solution data.

------

## Understanding the Key Quantum Components

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

For periodic boundaries, construct:
$$
x_i = i \Delta x, \quad \Delta x = \frac{L}{N}
$$
Build first-order derivative operator:

Dirichlet boundaries instead use `dx=L/(N+1)` and interior points `x=arange(dx,L,dx)`. On the classical path, `scheme='upwind'` is translated to `'forward'` when `a>=0` and `'backward'` otherwise; the Trotter path handles its components separately.

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

For the homogeneous semidiscrete system, skew-Hermiticity of `A` determines whether direct evolution is unitary. The wrapper itself always delegates to `schro_classical` or `schro_trotter`; it has no independent public direct-evolution switch.

#### Case A: Already Unitary

If:

- Central difference scheme
- Periodic boundary

Then:
$$
H=iA
$$
Properties:

- $A$ is skew-Hermitian
- Evolution is already unitary
- Direct unitary evolution is mathematically sufficient when `b=0`

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

#### If $A$ is skew-Hermitian and $b=0$:

$$
H = iA
$$

#### Otherwise:

- Use auxiliary variable lifting
- Apply Fourier transform
- Use the SDK's auxiliary-variable Fourier and recovery conventions.

------

### 5. Hamiltonian Construction

Split operator:
$$
A = H_1 + iH_2
$$
Construct:

- $H_1 = \frac{A + A^\dagger}{2}$
- $H_2 = \frac{A - A^\dagger}{2i}$

In the SDK, `schro_classical` builds `K=-D_eta⊗H1+I⊗H2` and applies `exp(+i*K*T)`. In the conventional `exp(-i*H*T)` notation, `H=-K`. In the skew case `H1=0`, so `H=iA=-H2`. Keep the source's sign convention distinct from the conventional notation. The Trotter variables named `H1` and `H2` below are component circuits constructed with `dt/R` and `dt`, not these raw matrices.

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

## Mathematical Deep Dive

- Grid size $n_x$ controls spatial accuracy
- Time step $\Delta t$ affects Trotter error
- Scheme choice:
  - Homogeneous periodic central differences give a skew generator
  - Upwind dissipation and stability depend on the derivative/sign/boundary construction
- Schrödingerization introduces:
  - auxiliary dimension cost
  - a higher-dimensional evolution and a separate recovery step

------

## Prerequisites

SDK documentation: [UnitaryLab](simulator-unitarylab.md#sources-and-version).

- PDE fundamentals: advection/transport equation $\partial u/\partial t = a \partial u/\partial x$
- Numerical methods: finite-difference discretization, central vs upwind schemes
- Schrödingerization theory: transforming non-unitary dynamics into unitary evolution
- Python: `numpy`, `unitarylab` core

## Core Parameters Explained

Parameters are parsed from the packaged setup schema (`unitarylab_algorithms/schrodingerization/equation_advection/setup.json`). Its top-level `params` is an array of equation/method objects with `par_fix`/`par_func` records. The table names resolved concepts, not flat `run()` arguments or a top-level `params['solver']` object. `solver.type` is a field of the parsed equation object.

| Parameter | Type | Description |
|---|---|---|
| `nx` | `int` | Grid size exponent; $N_x = 2^{n_x}$ spatial points |
| `L` | `float` | Domain length $[0, L]$ |
| `T` | `float` | Final simulation time |
| `a` | `float` | Coefficient multiplying the first-derivative operator; check the sign convention described above. |
| `scheme` | `str` | `"central"` or `"upwind"`; skew-Hermiticity and dissipation also depend on boundaries and operator construction. |
| `boundary` | `str` | `"periodic"` or `"dirichlet"` |
| `solver.type` | `str` | `"classical"` or `"trotter"` |
| `na` | `int` | Number of auxiliary qubits for Schrödingerization |
| `R` | `int` | Schrödingerization scaling parameter |
| `order` | `int` | Warped-phase initialization order passed to `initial_schro_fp`; it does not select a Suzuki product-formula order. |
| `dt` | `float` | Time step (Trotter only) |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `"ok"` in a returned result; parsing, unsupported-method, and solver exceptions propagate |
| `message` | `str` | Descriptive status message |
| `grid.n_points` | `int` | $N_x$ spatial grid points |
| `grid.dx` | `float` | Spatial step: periodic `L/Nx`, Dirichlet `L/(Nx+1)` |
| `grid.dt` | `float` | Time step (Trotter only) |
| `grid.nt` | `int` | Number of time steps (Trotter only) |
| `x` | `list[float]` | Spatial coordinate array |
| `u` | `list[float]` | Final solution $u(x, T)$ |
| `circuit` | `list[str]` | Paths to circuit diagram files |
| `plot.filename` | `str` | Path to solution plot SVG |

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Advection PDE | `parse_equation(params)` → `eq` object |
| Derivative operator $A_0$ | `first_order_derivative(N, dx, boundary, scheme)` |
| System matrix $A = a A_0$ | `A = A0 * a` |
| Skew generator and unitary evolution | Homogeneous periodic central differences are mathematically eligible for direct evolution; the wrapper still calls the selected SDK solver, which handles this case internally |
| Hermitian generator split | $H_1 = (A + A^\dagger)/2$, $H_2 = (A - A^\dagger)/(2i)$; SDK Fourier conventions determine lifted evolution |
| Classical solver | `schro_classical(A, u0, T, na, R, order, point, b)` |
| Trotter solver | `schro_trotter(u0, H1, H2, Nt, na, R, order, point)` |

## Minimal Manual Implementation

```python
import numpy as np

def advection_skeleton(Nx, dx, a, T):
    """Homogeneous periodic central-difference reference; returns a propagator.

    In practice, the implementation uses `first_order_derivative()` and
    `schro_classical()`/`schro_trotter()` from unitarylab.library.
    """
    # Build first-order derivative (central difference)
    A0 = np.zeros((Nx, Nx))
    for i in range(Nx):
        A0[i, (i + 1) % Nx] += 1.0      # forward
        A0[i, (i - 1) % Nx] -= 1.0      # backward
    A0 = A0 / (2 * dx)

    A = a * A0

    # Detect unitary structure
    is_skew = np.allclose(A, -A.T)
    if is_skew:
        # Direct unitary evolution: H = i * A
        from scipy.linalg import expm
        H = 1j * A
        U = expm(-1j * H * T)
        return U
    else:
        # Requires full Schrödingerization
        raise NotImplementedError("Full Schrödingerization required for non-unitary case")
```

## Hands-On Example

Prepare separate classical and Trotter configurations using the parser's method-object schema. The packaged file selects `solutionMethod_classical`; a Trotter configuration also needs the corresponding method parameters such as `dt`. Adding `params['solver']['type']` does not modify that packaged schema. Use the wrapper's returned `grid`, `x`, and `u` to compare matched physical configurations.

Compare both paths at the same effective final time and boundary configuration. The source uses `Nt=int(T/dt)`, so the Trotter duration is `Nt*dt`; a non-integral ratio truncates the final interval. Agreement is numerical evidence for the selected case, not a general solver guarantee.

## Debugging Tips

1. **Grid resolution**: $N_x = 2^{n_x}$ must match qubit register size. Too-few points cause dispersion artifacts; verify with a known analytical solution.
2. **Scheme selection**: Homogeneous periodic central differences give a skew generator. Check dissipative signs, boundary forcing, and actual operator construction for upwind cases; stability is not guaranteed by the scheme name alone.
3. **Trotter error**: Time step `dt` controls Trotter approximation accuracy. Halve `dt` and double `Nt` if the solution shows unphysical oscillations.
4. **Auxiliary qubit count `na`**: The auxiliary dimension is `2**na`. Compare recovery error while holding the other controls fixed; more qubits increase circuit/state cost and do not by themselves establish accuracy.
5. **Schrödingerization controls**: `R`, `na`, `point`, and warped-phase `order` govern auxiliary initialization/recovery. `order` does not raise the Trotter formula order. The classical SDK warns when `point<max(lambda_max(A1),0)*T`; inspect recovery error instead of treating packaged defaults as a convergence certificate.
