---
name: traffic-flow-schrodingerization-solver
description: This skill enables the AI agent to solve the 1D Traffic Flow Equation (LWR model) using the Schrödingerization framework, which transforms a nonlinear conservation law into a linear unitary evolution problem suitable for both classical and quantum simulation.
----



# Skill: Traffic Flow Equation (Schrödingerization)

The agent supports:

- Nonlinear scalar conservation laws
- Level-set lifting to high-dimensional linear systems
- Schrödingerization of non-Hermitian operators
- Classical solver (matrix evolution)
- Quantum solver (Trotter decomposition → quantum circuits)

It automatically:

1. Parses equation parameters
2. Constructs level-set representation
3. Builds discrete operators
4. Executes evolution (classical or quantum)
5. Generates solution and circuit visualizations

## **Original PDE**

$$
\frac{\partial u}{\partial t} + f'(u)\frac{\partial u}{\partial x} = 0
$$

with
$$
f(u) = u(1-u)
$$

------

## **Level Set Lifting**

Introduce:
$$
\phi(t,x,p) = 0, \quad p = u(t,x)
$$
Transform into Liouville equation:
$$
\partial_t \phi + \nabla_p f(p)\cdot \nabla_x \phi = 0
$$

------

## **Schrödingerization**

Linear system:
$$
\frac{d\Psi}{dt} = A \Psi
$$
Mapped to Hamiltonian:
$$
H_C = A_1 \otimes D_\eta + A_2 \otimes I
$$

------

## Agent Workflow (Step-by-Step with Code)

------

### Step 1 — Parse Input Parameters

Extract PDE configuration, solver type, discretization, and initial condition.

### Code

```python
def run(self, params: Optional[str] = None) -> Dict[str, Any]:
    self.logger.info("Stage 1: Parsing equation parameters...")
    eq = parse_equation(params)

    method = eq.solver.type
    if method == 'classical':
        return self._solve_classical(eq)
    elif method == 'trotter':
        return self._solve_trotter(eq)
```

------

### Step 2 — Construct Flux Function $f'(u)$

Symbolically compute nonlinear convection speed.

#### Code

```python
import sympy

u = sympy.symbols('u')
flow = sympy.sympify(eq.par_func['g(u)'])
flowx = sympy.diff(flow, u)
fx = sympy.lambdify(u, flowx, 'numpy')
```

------

### Step 3 — Discretize Spatial Domain

Create computational grid.

#### Code

```python
Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)

if bd == 'periodic':
    dx = L / Nx
    x = np.arange(0, L, dx)

u0 = f0(x)
```

------

### Step 4 — Level Set Initialization

Lift nonlinear PDE into linear phase-space formulation.

#### Code

```python
l_vec = np.linspace(-1, 1, nl)
dl = 1 / (nl - 1)

phi = l * np.ones(u0.size) - u0

w = 0.1
psi = np.where(abs(phi) > w, 0,
               (1 / w / 2) * (1 + np.cos(np.pi * phi / w)))
```

------

### Step 5 — Upwind Scheme Selection

Ensure numerical stability.

#### Code

```python
if scheme == 'upwind':
    if fxl > 0:
        scheme = 'forward'
    else:
        scheme = 'backward'
```

------

### Step 6 — Construct Differential Operator

Build discrete convection operator.

#### Code

```python
A0, b0 = first_order_derivative(
    N=Nx,
    dx=dx,
    boundary_condition=bd,
    scheme=scheme
)

A = A0.T * fxl
b = b0 * fxl
```

------

### Step 7 — Classical Schrödinger Evolution

Solve linearized system.

#### Code

```python
u_l = schro(
    A,
    psi,
    T=T,
    na=na,
    R=R,
    order=order,
    point=point,
    b=b
)
```

------

### Step 8 — Reconstruction via Level Set Integration

Recover physical solution $u(x,t)$.

#### Code

```python
v = v + l * u_l * dl
v_norm = v_norm + u_l * dl

u = v / v_norm
```

------

### Step 9 — Quantum (Trotter) Evolution

Simulate Hamiltonian via quantum circuits.

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

#### Code

```python
H1 = func1((abs(fxl) * dx / 2) * dt / R)
H2 = func2(-fxl * dt)

u_l, qc = schro(
    u0=psi,
    H1=H1,
    H2=H2,
    R=R,
    na=na,
    Nt=Nt,
    order=order,
    point=point
)
```

------

### Step 10 — Generate Solution Plot

Visualize final result.

#### Code

```python
fig, ax = plt.subplots()

ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)

ax.set_title(name)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
```

------

### Step 11 — Generate Quantum Circuit

Export circuit representation.

#### Code

```python
qc.draw(filename=circuit_path1, title=f"{name} (Schro)")

H1.decompose().draw(filename=circuit_path2)
H2.decompose().draw(filename=circuit_path3)
```

------

## Agent Capabilities Summary

This skill allows the AI agent to:

- Handle nonlinear PDEs via **linear lifting**
- Automatically construct **flux derivatives**
- Apply **upwind schemes**
- Perform **Schrödingerization**
- Run both:
  - Classical simulation
  - Quantum Trotter simulation
- Generate:
  - Numerical solutions
  - Quantum circuits
  - Visualization plots