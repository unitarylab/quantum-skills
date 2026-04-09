---
name: maxwell-1d-schrodingerization
description: A quantum-compatible solver for the 1D Maxwell equations using Schrödingerization with staggered-grid discretization. Supports impedance/periodic boundary conditions, outputs electric and magnetic fields, visualizations, and quantum circuit diagrams.
---

# Skill: Quantum Simulation for 1D Maxwell's Equations

## Basic Information

- **Skill ID**: `quantum_maxwell_1d`
- **Skill Name**: Quantum Simulation of 1D Maxwell's Equations Based on Schrödingerization
- **Domain**: Quantum Computing, Electromagnetics, PDE Numerical Simulation
- **Core Capability**: Implements quantum solution for **1D Maxwell's equations** using Schrödingerization and twisted phase transformation. Supports impedance and periodic boundary conditions, and outputs electric/magnetic field solutions, visualizations, and quantum circuit diagrams.

------

## 1. Mathematical Background

### 1.1 Governing Equations

Maxwell's equations in 1D (assuming non-dispersive medium and no charges) are:
$$
\begin{aligned}
\frac{\partial D}{\partial t} &= \nabla \times H - J,\\
\frac{\partial B}{\partial t} &= -\nabla \times E,\\
\nabla \cdot B &= 0,\\
\nabla \cdot D &= \rho,
\end{aligned}
$$
where $E(x,t)$ is the electric field, $B(x,t)$ is the magnetic field, $D = \epsilon E$, $H = B/\mu$, $J$ is current density, and $\rho$ is charge density.

### 1.2 Schrödingerization

- Transform Maxwell’s equations into a **high-dimensional Schrödinger system** using twisted phase transformation.
- Apply Trotter splitting and Fourier methods for time evolution.
- Convert non-unitary dynamics into quantum-compatible unitary evolution.
- Preserve divergence-free property of $B$ and energy conservation.

------

## 2. Supported Features

- Spatial dimension: **1D**
- Boundary conditions: **Periodic**, **Impedance**
- Discretization: Staggered Yee-like grid
- Quantum solver: Classical matrix exponentiation (Schrödingerization-based)
- Outputs: Electric field $E(x,t)$, Magnetic field $B(x,t)$, field plots, quantum circuit diagrams

------

## 3. Algorithm Pipeline (Conceptual)

1. **Parameter Parsing**: wave speed, boundary type, initial fields, qubit numbers
2. **Matrix Construction**: staggered-grid differential matrices and Hamiltonian
3. **Quantum Evolution**: Schrödingerization solver with source term $J$
4. **Visualization**: Electric and magnetic field distribution
5. **Circuit Export**: Quantum circuit diagram generation

------

## 4. Core Methodology

### 4.1 System Matrix Construction

For impedance boundary conditions:
$$
\begin{aligned}
E_{\text{block}}[0,0] &= -\frac{2}{\Delta x},\\
EB_{\text{block}}[0,0] &= -2\alpha,
\end{aligned}
$$
and assemble the full evolution matrix:
$$
A = 
\begin{bmatrix}
E_{\text{block}} & EB_{\text{block}} \\
BE_{\text{block}} & B_{\text{block}}
\end{bmatrix}.
$$

```python
# Build block matrix for 1D Maxwell's equations
E_block = lil_matrix((Ne, Ne))
EB_block = lil_matrix((Ne, Nb))
BE_block = lil_matrix((Nb, Ne))

# Boundary treatment for impedance boundaries
E_block[0, 0] = -2.0 / dx
EB_block[0, 0] = -2.0 * alpha

# Assemble full evolution matrix
A = vstack([
    hstack([E_block, EB_block]),
    hstack([BE_block, B_block])
])
```

### 4.2 Staggered-Grid Differential Operator

For the magnetic field $B$:
$$
D_B[i,i-1] = -1, \quad D_B[i,i] = 1, \quad i = 1,2,\dots,M_x
$$
with electric field updates applied on staggered grid points.

```python
# Staggered difference for E and B fields
D_B = lil_matrix((Mx, Mx))
for i in range(1, Mx):
    D_B[i, i - 1] = -1.0
    D_B[i, i] = 1.0
Dx = -alpha * D_B
```

### 4.3 Schrödingerization Solver

- Solve the system $u(t) = e^{-i H t} u_0$ in the Schrödingerized form.
- Extract physical fields:

$$
E = u[:N_E], \quad B = u[N_E:]
$$

```python
# Schrödingerization solver for Maxwell system
u = schro(A, u, bE, T=T, R=R, na=na, order=order, point=point, scale=1e-3)

# Extract electric and magnetic fields
E = u[:len(xE)]
B = u[len(xE):]
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

## 5. Boundary & Initial Conditions

- **Periodic BC**: $E(0) = E(L)$, $B(0) = B(L)$
- **Impedance BC**: Accounts for partial reflection at boundaries
- Initial condition: user-defined $E_0(x), B_0(x)$ (e.g., $E_0(x) = \cos(2\pi x / L)$, $B_0(x) = 0$)

------

## 6. Outputs

- Electric field $E(x,t)$ and magnetic field $B(x,t)$
- Field distribution plots (real-time or final)
- Quantum circuit diagrams
- Numerical verification of divergence-free property

------

## 7. Trigger Phrases

- Simulate 1D Maxwell equations via Schrödingerization
- Quantum computation of EM wave propagation
- Solve Maxwell equations with impedance boundaries using quantum method
- Generate electric and magnetic field solutions via quantum simulation

------

## 8. Use Cases

- EM wave propagation in waveguides or conductors
- PDE-based quantum algorithm demonstration
- Physics education: visualizing electromagnetic fields
- Research: energy-preserving, divergence-free EM simulation

------

### Summary

This skill provides a **complete quantum solution for 1D Maxwell’s equations**:

- Schrödingerization framework with staggered-grid discretization
- Supports impedance and periodic boundary conditions
- Outputs $E$ and $B$ fields, visualizations, and quantum circuits
- Ensures physical consistency and energy preservation