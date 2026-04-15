---
name: helmholtz-1d-schrodingerization
description: A quantum-compatible solver for the 1D Helmholtz equation using Schrödingerization via damped dynamical system reformulation. Supports radiation boundary conditions, preconditioning, and complex-valued steady-state wave solutions with quantum circuit generation.
---

## One Step to Run 1D Helmholtz Example
```bash
python ./scripts/algorithm.py
```

# Skill: Quantum Simulation for 1D Helmholtz Equation

## 1. Mathematical Background

### 1.1 1D Helmholtz Equation

$$
- \Delta u - k^2 u = f, \quad x \in (0, L)
$$

- $k$: wave number, proportional to wave frequency  
- $u(x)$: complex-valued steady-state wave field  
- $f(x)$: source term  
- When $k=0$, reduces to **Laplace equation**

### 1.2 Schrödingerization via Damped Dynamical System

1. Discretize to linear system:

$$
A x = b
$$

2. Rewrite as second-order damped ODE:

$$
\frac{d}{dt} V = M V, \quad V = \begin{bmatrix} v \\ w \end{bmatrix}
$$

3. Transform to homogeneous form:

$$
\frac{d}{dt} V_f = M_f V_f
$$

4. Split into Hermitian parts:

$$
M_f = H_1 + i H_2
$$

5. Apply warped phase transformation and Fourier discretization, yielding Hamiltonian:

$$
H = D_p \otimes H_1 - I_{N_p} \otimes H_2
$$

------

## 2. Supported Features

- 1D time-harmonic wave propagation simulation  
- **Sommerfeld radiation boundary condition**  
- Central finite-difference discretization with shifted wave number  
- Complex-valued wave field solution  
- Quantum solver:
  - Classical matrix exponentiation  
- Preconditioning for numerical stability  
- Real / imaginary part visualization  
- Quantum circuit diagram output  

------

## 3. Full Algorithm Pipeline (Step-by-Step)

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

### Step 1: Parse Wave & Domain Parameters

```python
k = eq.get_parameter('k')
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type

Nx = 2**nx
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
```

### Step 2: Build Discrete Helmholtz Matrix

Central difference with radiation boundary:

```python
A0 = np.zeros((Nx, Nx), dtype=complex)
for i in range(1, Nx-1):
    A0[i, i-1] = -1
    A0[i, i]   = 2 - 2*(1 - np.cos(k*dx))
    A0[i, i+1] = -1

# Sommerfeld boundary at x=L
A0[0, 0] = 1
A0[-1, -1] = -(1 - 1j*k*dx)
A0[-1, -2] = 1
```

Implements discretization of:
$$
- \frac{1}{\Delta x^2} (u_{i-1} - 2 u_i + u_{i+1}) - k^2 u_i = f(x_i)
$$
with shifted wave number:
$$
\hat{k}^2 = \frac{2}{\Delta x^2} (1 - \cos(k \Delta x))
$$

### Step 3: Construct Source Term & Preconditioning

```python
b0 = dx**2 * source(x)
b0[0] = 0
b0[-1] = 0

# Preconditioner
Pinv = ...  # build preconditioning matrix
P = np.linalg.inv(Pinv)
A0 = P @ A0
b0 = P @ b0
```

### Step 4: Form Damped Dynamical System Matrix

```python
gam = 2 * m
b = np.concatenate((np.zeros(Nx), -b0))
A = np.block([
    [np.zeros((Nx,Nx)), -A0.conj().T],
    [A0,               -gam*np.eye(Nx)]
])
```

### Step 5: Quantum Schrödingerization Solver

```python
u0 = np.zeros_like(b)
u = schro_classical(A, u0, T=T, na=na, b=b, scale_b=1/T)
u = u[:Nx]  # extract physical solution
qc = circuit_classical(nx, na)
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

### Step 6: Visualize Real & Imaginary Parts

```python
ax.plot(x, np.real(u), "b-", label="Re(u)")
ax.plot(x, np.imag(u), "r--", label="Im(u)")
```

------

## 4. Boundary & Initial Conditions

### Boundary Condition

**Sommerfeld Radiation Condition**:
$$
u(0) = 0, \quad u'(L) - i k u(L) = 0
$$

### Initial Condition

Not applicable — Helmholtz equation is **steady-state elliptic PDE**.

------

## 5. Finite-Difference Scheme

Central difference:
$$
- \frac{1}{\Delta x^2} (u_{i-1} - 2 u_i + u_{i+1}) - k^2 u_i = f(x_i)
$$
with shifted wave number:
$$
\hat{k}^2 = \frac{2}{\Delta x^2} (1 - \cos(k \Delta x))
$$

------

## 6. Outputs

- Complex-valued steady-state wave field $u(x)$
- Real / imaginary part distribution plot
- Full quantum circuit diagram
- Preconditioned quantum operator diagrams

------

## 7. Trigger Phrases

- Solve 1D Helmholtz equation using quantum method
- Quantum simulation for time-harmonic acoustic waves
- Quantum solver for wave propagation with radiation BC
- Schrödingerization for Helmholtz PDE

------

## 8. Use Cases

- Acoustic cavity resonance analysis
- Electromagnetic waveguide mode calculation
- Quantum mechanical bound state simulation
- Steady-state wave scattering problems

------

### Summary

This skill provides a **complete quantum solution for the 1D Helmholtz equation**:

- Based on damped dynamical system reformulation and Schrödingerization
- Supports Sommerfeld radiation boundary for outgoing waves
- Uses preconditioning to improve numerical stability
- Computes complex wave fields and visualizes real/imaginary parts
- Generates quantum circuits and professional solution plots
- Fully aligned with your implementation and mathematical framework