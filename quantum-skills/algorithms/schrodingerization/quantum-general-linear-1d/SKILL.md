---
name: quantum-general-linear-1d
description: Quantum solver for 1D general linear evolution equations via Schrödingerization. Supports arbitrary-order derivatives, multiple boundary conditions, and classical/Trotter quantum solvers. Outputs solution fields, plots, and quantum circuit diagrams.
---

# Skill: Quantum Simulation of 1D General Linear PDEs

## Basic Information

- **Skill ID**: `quantum_general_linear_1d`
- **Skill Name**: Quantum Solver for 1D General Linear Evolution Equations
- **Domain**: Quantum Computing, PDEs, Scientific Computing
- **Core Capability**: Solve **1D linear time-dependent PDEs** using Schrödingerization. Supports arbitrary-order derivatives, Dirichlet/Periodic/Neumann boundaries, classical/Trotter solvers, and automatic finite-difference assembly.

---

## 1. Mathematical Formulation

### 1.1 General Linear PDE

\[
\frac{\partial u}{\partial t} = a \frac{\partial u}{\partial x} + b \frac{\partial^2 u}{\partial x^2} + c \frac{\partial^3 u}{\partial x^3} + \dots + f(x)
\]

- \(a,b,c,\dots\) — coefficients  
- \(f(x)\) — source term  

### 1.2 Schrödingerized Hamiltonian

\[
H = - a \eta^{\otimes 1} + b \eta^{\otimes 2} - c \eta^{\otimes 3} + \dots
\]

- Periodic BC + Hermitian symmetric scheme  
- Source-free case \(f=0\)

---
The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

## 2. Algorithm Steps (Pipeline)
### Step 0: Import Libraries
Import necessary modules for parsing, solvers, operators, and circuit generation:
```python
# import parser
from engine.library import parse_equation

# import solvers
from engine.library import schro_classical, schro_trotter
from engine.library.differential_operator.classical_matrices import first_order_derivative, second_order_derivative
from engine.library.schrodingerization.classical import circuit_classical
from scipy.integrate import cumulative_trapezoid
```
### Step 1: Parse Parameters

```python
L, T, nx, na, R, point, order, f0 = eq.get_common_coefficients()
derivatives = eq.get_derivative_1d()  # {1: a, 2: b, 3: c, ...}
bd = eq.boundary.type
scheme = eq.discrete.type

Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
u0 = f0(x)
```

### Step 2: Assemble Finite-Difference Operator

```python
A = ClassicalOperator()
for k, coeff in derivatives.items():
    if coeff != 0:
        D = CDiff(N=Nx, dx=dx, order=int(k), scheme=scheme, boundary=bd)
        A += D * coeff
A = A.get_matrix()
b = eq.get_rhs_1d(x)  # source term
```

### Step 3: Schrödingerization Quantum Solver

```python
u = schro_classical(A, u0, T=T, na=na, R=R, order=order, b=b, scale_b=1.0/T)
```

### Step 4: Trotter Time Splitting (Optional)

```python
A_trotter = TrotterOperator()
func1, func2 = A_trotter.data()
H1 = func1(dt / R)
H2 = func2(dt)
u, qc = schro_trotter(u0, H1, H2, Nt=Nt, na=na, b=b, theta=dt/T)
```

### Step 5: Visualization

```python
ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

### Step 6: Quantum Circuit Export

```python
qc.draw(filename="general_linear_1d_circuit.svg")
```

------

## 3. Boundary Conditions

- **Dirichlet:** $u(0,t)=0, u(L,t)=0$
- **Periodic:** $u(0,t)=u(L,t)$
- **Neumann:** $\partial_x u = 0$

------

## 4. Initial Conditions

- Sine: $u(x,0)=\sin(2\pi x/L)$
- Discontinuous step: for shock test
- Custom: user-defined function $f_0(x)$

------

## 5. Trigger Phrases

- Quantum simulation of 1D general linear PDE
- Schrödingerization for arbitrary-order linear PDE
- Solve advection-diffusion or wave equation via quantum method

------

## 6. Use Cases

- 1D advection-diffusion problems
- Heat equation
- Linear wave propagation
- General linear PDE benchmarking

------

## Summary

- Standardized **step-by-step quantum simulation** pipeline for 1D linear PDEs
- Arbitrary derivatives, multiple boundaries, and initial conditions
- Schrödingerization quantum solver + Trotter option
- Automatic FD matrix assembly and circuit export
- Fully consistent with **Burgers / Advection / Elastic Wave** skill style