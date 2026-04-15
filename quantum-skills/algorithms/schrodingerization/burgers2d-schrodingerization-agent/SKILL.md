---
name: burgers2d-schrodingerization-agent
description: Solve the 2D inviscid Burgers' equation using level-set lifting and Schrödingerization. This agent transforms nonlinear PDEs into linear high-dimensional systems, enabling both classical simulation and quantum-inspired Trotter evolution.
---

## One Step to Run 2D Burgers Example
```bash
python ./scripts/algorithm.py
```

# Skill: 2D Burgers Equation via Schrödingerization + Level-Set

---

## Governing Equation

\[
\begin{cases}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = 0 \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = 0
\end{cases}
\]

---

## Core Idea

### Step 1: Level-set lifting

Introduce:

\[
\phi(t,x,y,\xi,\zeta)
\]

Transform nonlinear PDE → linear Liouville equation:

\[
\partial_t \phi + \xi \partial_x \phi + \zeta \partial_y \phi = 0
\]

---

### Step 2: Schrödingerization

Convert:

\[
\frac{d\Psi}{dt} = A \Psi
\]

to unitary evolution via ancilla system:

\[
i \frac{d}{dt} \Psi = H \Psi
\]

---

### Step 3: Numerical realization

- Spatial discretization: finite difference
- Velocity space: quadrature over level-set variable \( \ell \)
- Reconstruction:
  
\[
u = \frac{\int \ell \, u_\ell d\ell}{\int u_\ell d\ell}
\]

---

## Supported Methods

### 1. Classical
- Matrix exponential via `schro_classical`
- Loop over level-set samples

### 2. Trotter (Quantum Simulation)
- First/second order Trotter splitting
- Gate-based Hamiltonian simulation

---

## Algorithm Workflow

### Stage 0: Import Libraries
```python
# import parser
from engine.library import parse_equation

# import solvers
from engine.library import schro_classical, schro_trotter
from engine.library.differential_operator.classical_matrices import first_order_derivative, second_order_derivative
from engine.library.schrodingerization.classical import circuit_classical
from scipy.integrate import cumulative_trapezoid
```

### Stage 1: Parse Parameters

```python
eq = parse_equation(params)
L, T, source, nx, na, R, point, order, f0 = eq.get_common_coefficients()
nu = eq.get_parameter('ν')
```

---

### Stage 2: Grid Construction

```python
Nx = 2**nx
dx = L / Nx
x = np.arange(0, L, dx)
y = np.arange(0, L, dx)

u0 = f0(x[:, None], y[None, :]).flatten()
```

---

### Stage 3: Level-set Sampling

```python
nl = eq.get_parameter('nl', 15)
l_vec = np.linspace(-1, 1, nl)
dl = 1 / (nl - 1)
```

---

### Stage 4: Delta Approximation (关键!)

```python
phi = l * np.ones(u0.size) - u0
w = 0.1

psi = np.where(
    abs(phi) > w,
    0,
    (1/(2*w)) * (1 + np.cos(np.pi * phi / w))
)
```

---

### Stage 5: Build Operator

#### Classical

```python
A0, _ = first_order_derivative(...)
A1, _ = second_order_derivative(...)

A = (
    kron(A0.T, I) * l
    + kron(A1, I) * nu
    + kron(I, A0.T) * l
    + kron(I, A1) * nu
)
```

---

#### Trotter (Quantum)

```python
D1 = func1((-abs(l)*dx/2 + nu) * dt / R)
D2 = func2(-l * dt)

H1.append(D1, ...)
H2.append(D2, ...)
```

---

### Stage 6: Schrödinger Evolution

```python
u_l = schro(A=A, u0=psi, T=T, ...)
```

或

```python
u_l, qc = schro(u0=psi, H1=H1, H2=H2, Nt=Nt, ...)
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

---

### Stage 7: Reconstruction

```python
v += l * u_l * dl
v_norm += u_l * dl

u = v / v_norm
```

---

### Stage 8: Visualization

```python
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u.reshape(Nx, Nx))
```

---

## Key Features

- Handles **nonlinear PDE via linear lifting**
- Naturally supports **shock formation**
- Compatible with **quantum simulation**
- Modular: easy to extend to higher dimensions

---

## When to Use

Use this skill when:

- Solving nonlinear conservation laws
- Studying shock formation
- Testing Schrödingerization framework
- Comparing classical vs quantum PDE solvers

---

## Limitations

- High dimensional lifting → computational cost
- Requires careful choice of:
  - $nl$ (level-set resolution)
  - $w$ (delta width)
- Upwind scheme needed for stability

---

## Extension Ideas

- Adaptive level-set sampling
- GPU acceleration
- Hybrid classical-quantum solver
- Shock tracking enhancement