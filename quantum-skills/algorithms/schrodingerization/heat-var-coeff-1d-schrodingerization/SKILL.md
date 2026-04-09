---
name: heat-var-coeff-1d-schrodingerization
description: A quantum-compatible solver for the 1D heat equation with spatially varying diffusion coefficient using Schrödingerization. Supports periodic boundaries, variable a(x)=1+cos(2πx/L), classical and block quantum solvers, and generates 1D temperature profiles along with quantum circuit diagrams.
---

# Skill: Quantum Simulation for 1D Variable-Coefficient Heat Equation

## Basic Information

- **Skill ID**: `quantum_heat_var_coeff_1d`
- **Skill Name**: Quantum Solver for 1D Variable-Coefficient Forward Heat Equation
- **Domain**: Quantum Computing, Heat Conduction, PDEs, Nonhomogeneous Materials
- **Core Purpose**: Solve the **1D heat equation with spatially varying thermal conductivity** using Schrödingerization quantum methods. Supports periodic boundaries, variable diffusivity $a(x)$, and classical/block solvers.

------

## 1. Mathematical Background

### 1.1 Variable-Coefficient Heat Equation

$$
\frac{\partial u}{\partial t} = a(x) \frac{\partial^2 u}{\partial x^2} + f(x)
$$

where the diffusion coefficient is:

$$
a(x) = 1 + \cos\Big(\frac{2 \pi x}{L}\Big)
$$

- $a(x)$: spatially varying thermal conductivity  
- $u(x,t)$: temperature field  
- $f(x)$: heat source  

### 1.2 Schrödingerized Hamiltonian (Periodic Form)

$$
H \approx D_\eta \otimes a(X) D_\Delta
$$

- $D_\eta$: auxiliary operator discretization  
- $a(X)$: diagonal coefficient matrix  
- $D_\Delta$: discrete Laplacian  

Valid for:

- Hermitian symmetric difference  
- Periodic boundary conditions  
- Source-free case $f=0$

------

## 2. Supported Features

- 1D heat conduction with **spatially varying diffusivity**  
- Diffusion coefficient: $a(x) = 1 + \cos(2\pi x / L)$  
- Boundary condition: **Periodic**  
- Initial conditions: sine wave, discontinuous step  
- Quantum solvers:
  - Classical matrix exponentiation  
  - Block encoding (fallback to classical)  
- Central finite-difference discretization  
- 1D temperature profile visualization  
- Quantum circuit diagram output  

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Domain & Physical Parameters

Extract domain length, time, qubits, and boundary type.

```python
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type
scheme = eq.discrete.type

Nx = 2**nx
dx = L / Nx
x = np.arange(0, L, dx)
u0 = f0(x)
```

### Step 2: Initialize Temperature Field

Load initial condition (sine or discontinuous).

```python
u0 = f0(x)  # initial temperature distribution
```

### Step 3: Build Variable-Coefficient Operator

Construct Laplacian and multiply by spatially varying coefficient $a(x)$.

```python
# Build discrete second-derivative operator
p2_matrix = generate_compact_p_2_normal(nx, 0, L)

# Build diagonal matrix for a(x) = 1 + cos(2πx/L)
ax = 1 + np.cos(2 * np.pi * x / L)
ax_matrix = np.diag(ax)

# Assemble full operator: A = a(x)·Δ
A = ax_matrix @ p2_matrix
```

### Step 4: Set Source Term

```python
b = source(x)
```

### Step 5: Quantum Schrödingerization Solver

```python
u = schro(A, u0, T=T, na=na, R=R, order=order, b=b)
qc = circuit_classical(nx, na)
```

### Step 6: Visualization

```python
ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

### Step 7: Output Results & Quantum Circuit

```python
circuit_files = _generate_circuit_plots(name, qc)
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Discrete Laplacian Construction

```python
# Generate second-order spatial derivative matrix
p2_matrix = generate_compact_p_2_normal(nx, 0, L)
```

### 4.2 Variable-Coefficient Diagonal Matrix

```python
# Build a(x) as a diagonal matrix
ax = 1 + np.cos(2 * np.pi * x / L)
ax_matrix = np.diag(ax)
```

### 4.3 Full PDE Operator Assembly

```python
# Variable-coefficient heat operator: A = a(x)·Δ
A = ax_matrix @ p2_matrix
```

### 4.4 Quantum Solver Call

```python
# Core quantum evolution for variable-coefficient PDE
u = schro(A, u0, T=T, na=na, b=b)
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

## 5. Boundary & Initial Conditions

### Boundary Condition

**Periodic BC**:
$$
u(0,t) = u(L,t)
$$

### Initial Conditions

- Sine wave:

$$
u(x,0) = \sin\Big(\frac{2\pi x}{L}\Big)
$$

- Discontinuous step:

$$
u(x,0) =
\begin{cases}
0, & x \in [0, L/2) \\
1, & x \in [L/2, L]
\end{cases}
$$

------

## 6. Finite-Difference Scheme

Central difference for variable-coefficient diffusion:
$$
a(x_i) \frac{\Delta u_i}{\Delta x^2} \approx a(x_i) \frac{u_{i+1}^n - 2 u_i^n + u_{i-1}^n}{\Delta x^2}
$$

------

## 7. Outputs

- Temperature field $u(x,T)$
- 1D solution plot
- Full quantum circuit diagram
- Circuit decomposition files

------

## 8. Trigger Phrases

- Solve 1D variable-coefficient heat equation using quantum method
- Quantum simulation for nonhomogeneous heat conduction
- Variable diffusivity thermal PDE with Schrödingerization
- Quantum PDE solver for spatially varying materials

------

## 9. Use Cases

- Heat conduction in composite materials
- Thermal simulation with spatially varying conductivity
- Quantum acceleration for variable-coefficient PDEs
- Thermal analysis of periodic structures

------

### Summary

This skill provides a **complete quantum solution for the 1D variable-coefficient heat equation**:

- Uses $a(x)=1+\cos(2\pi x/L)$ for spatially varying diffusion
- Builds the PDE operator as $A = \text{diag}(a(x)) \cdot \Delta$
- Supports classical and block quantum solvers
- Automatically generates visualizations and quantum circuits
- Fully aligned with your implementation and mathematical framework