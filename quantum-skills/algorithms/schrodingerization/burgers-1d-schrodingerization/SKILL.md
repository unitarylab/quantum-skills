---
name: burgers-1d-schrodingerization
description: A quantum-compatible solver for the 1D viscous Burgers’ equation using Schrödingerization and Cole-Hopf transformation to linearize the nonlinear PDE. Supports shock wave formation, viscous smoothing, and both classical and quantum Trotter-based evolution with circuit generation.
---

# Skill: Quantum Simulation for 1D Burgers’ Equation

## Basic Information

- **Skill ID**: `quantum_burgers_1d`
- **Skill Name**: Quantum Solver for 1D Viscous Burgers’ Equation
- **Domain**: Quantum Computing, Computational Fluid Dynamics, Nonlinear PDEs
- **Core Purpose**: Solve the **1D viscous Burgers’ equation** using quantum Schrödingerization + **Cole-Hopf transformation** to linearize the nonlinear PDE. Supports shock wave capturing, viscous smoothing, and both classical matrix exponential and Trotter time-splitting quantum solvers.

---

## 1. Mathematical Background

### 1.1 Burgers’ Equation

\[
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, L], \, t \ge 0
\]

- \(\nu\): kinematic viscosity (diffusion coefficient)  
- Nonlinear convection + linear diffusion  
- Models shock wave formation

### 1.2 Linearization via Cole-Hopf Transformation

The Cole-Hopf transformation converts the nonlinear Burgers’ equation into a **linear heat equation**:

\[
u(x,t) = -2 \nu \frac{\partial_x \phi(x,t)}{\phi(x,t)}
\]

where \(\phi(x,t)\) satisfies the linear PDE:

\[
\frac{\partial \phi}{\partial t} = \nu \frac{\partial^2 \phi}{\partial x^2}
\]

### 1.3 Schrödingerized Hamiltonian

\[
H = l_p^{-} \nu p^2
\]

Valid for **periodic boundary conditions** and **symmetric (Hermitian) difference schemes**.

The Schrödingerization framework can be referred to in "./Schr_skills.markdown"
---

## 2. Supported Features

- 1D viscous / inviscid Burgers’ equation  
- **Cole-Hopf linearization**  
- Boundary conditions: Dirichlet, Periodic  
- Initial conditions: sine, discontinuous, Gaussian  
- Quantum solvers:
  - Classical matrix exponentiation  
  - Trotter–Lie splitting  
- Shock wave and smooth solution visualization  
- Quantum circuit diagram output

---

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Parameters

```python
nu = eq.get_parameter('ν')
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type
Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
u0 = f0(x)
```

### Step 2: Cole-Hopf Transformation

$$
\phi_0(x) = \exp\Bigg(-\frac{1}{2\nu} \int_0^x u_0(s) \, ds \Bigg)
$$

```python
I = cumulative_trapezoid(u0, x, initial=0.0)
exponent = -I / (2.0 * nu)
exponent -= np.max(exponent)  # avoid overflow
phi0 = np.exp(exponent)
phi0 = np.clip(phi0, 1e-300, np.inf)
```

### Step 3: Build Diffusion Matrix

$$
\mathbf{A} = \nu \mathbf{L}, \quad \mathbf{L} \text{ is the discrete Laplacian matrix}
$$

```python
from engine.library.differential_operator.classical_matrices import second_order_derivative
A0, b0 = second_order_derivative(N=Nx, dx=dx, boundary_condition=bd)
A = A0 * nu
b = b0 * nu
```

### Step 4: Quantum Schrödingerization Solver

```python
from engine.library import schro_classical
phi = schro_classical(A, phi0, T=T, na=na, R=R, order=order, b=b)
qc = circuit_classical(nx, na)
```
The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

### Step 5: Invert Cole-Hopf to Recover u

$$
u(x,t) = -2 \nu \frac{\partial_x \phi(x,t)}{\phi(x,t)}
$$

```python
phi_x_over_phi = np.gradient(phi, dx) / phi
u = -2.0 * nu * phi_x_over_phi
```

### Step 6: Optional Trotter Quantum Circuit

```python
from engine.library import schro_trotter
func1, func2 = (nu * TDiff(nx, dx, 2, boundary=bd)).data()
H1 = func1(dt / R)
phi, qc = schro_trotter(u0=phi0, H1=H1, H2=None, Nt=Nt, na=na)
```

### Step 7: Visualization

```python
ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

------

## 4. Boundary & Initial Conditions

### Boundary Conditions

- Dirichlet: $u(0,t)=g_1, \quad u(L,t)=g_2$
- Periodic: $u(0,t)=u(L,t)$

### Initial Conditions

- Sine: $u(x,0) = b \sin\Big(2\pi (x+c)/L\Big)$
- Discontinuous: piecewise step
- Gaussian: smooth profile

------

## 5. Discretization Schemes

### Central Difference

$$
\Delta u_i \approx - \frac{l}{2} \Delta x \Delta t (u_{i+1} - u_{i-1}) + \nu (\Delta x)^2 \Delta t (u_{i+1} - 2 u_i + u_{i-1})
$$

### Upwind Difference

$$
\Delta u_i \approx - l \Delta x \Delta t (u_i - u_{i-1}) + \nu (\Delta x)^2 \Delta t (u_{i+1} - 2 u_i + u_{i-1})
$$

------

## 6. Outputs

- Shock / smooth solution $u(x,T)$
- 1D solution plot
- Full quantum circuit diagram
- H1 decomposed circuit (Trotter)

------

## 7. Trigger Phrases

- Solve 1D Burgers’ equation using quantum method
- Quantum simulation for viscous shock waves
- Quantum PDE solver with Cole-Hopf transformation
- Compute nonlinear convection–diffusion with quantum algorithm

------

## 8. Use Cases

- Shock wave simulation
- Fluid dynamics benchmarking
- Nonlinear PDE quantum acceleration
- Viscous flow modeling

------

### Summary

This skill provides a **complete quantum solution for 1D Burgers’ equation**:

- Uses **Cole-Hopf transformation** to linearize the nonlinear PDE
- Solves the linear heat equation via quantum Schrödingerization
- Supports both classical and Trotter quantum solvers
- Automatically reconstructs the original nonlinear solution
- Captures shock waves and viscous smoothing
- Fully aligned with your implementation and mathematical framework