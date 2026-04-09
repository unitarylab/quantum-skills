---
name: quantum-multiscale-elliptic
description: Quantum simulation for 1D highly oscillatory multiscale elliptic equations using Schrödingerization. Supports both original fine-scale model and two-scale homogenization model, constructs FEM stiffness matrices, computes macro and micro solutions, and outputs quantum circuit diagrams.
---

# Skill: Quantum Algorithm for Multiscale Elliptic PDEs

## Basic Information

- **Skill ID**: `quantum_multiscale_elliptic`
- **Skill Name**: Quantum Simulation for Multiscale Elliptic Equations (Homogenization + Schrödingerization)
- **Domain**: Quantum Computing, Multiscale Modeling, Scientific Computing
- **Core Capability**: Solve **1D highly oscillatory multiscale elliptic PDEs** using quantum Schrödingerization, supporting **original fine-scale model** and **two-scale homogenization model**.

------

## 1. Theoretical Background

### 1.1 Multiscale Elliptic PDE

$$
-\nabla \cdot \big(A(x) \nabla u(x)\big) = f(x), \quad u|_{\partial D} = 0
$$

where $A(x) = A(x, x/\epsilon)$ is a highly oscillatory coefficient.

### 1.2 Two‑Scale Homogenization Model

$$
\begin{cases}
-\nabla_y \cdot A(x,y) (\nabla_x u_0 + \nabla_y u_1) = 0,\\[2mm]
-\nabla_x \cdot \int_Y A(x,y) (\nabla_x u_0 + \nabla_y u_1)\, dy = f(x)
\end{cases}
$$

### 1.3 Quantum Solver: Schrödingerization

- Convert linear system $S u = f$ into a dynamical system
- Introduce auxiliary variables
- Transform into Schrödinger form for quantum simulation

------

## 2. Supported Features

- 1D multiscale elliptic PDE with highly oscillatory coefficients
- Two-scale homogenization solver
- FEM stiffness matrix construction
- Quantum Schrödingerization solver
- Visualization of $u_0$ (macro) and $u_1$ (micro)
- Quantum circuit diagram output

------

## 3. Algorithm Execution Pipeline

### Step 1: Parameter Parsing

- Parse scale parameter $\epsilon$, domain, quantum bits, solver type

### Step 2: Define Oscillatory Coefficient

$$
A_\epsilon(x) = \frac{1}{2 + \cos\left(\frac{2 \pi x}{\epsilon L}\right)}
$$

### Step 3: Stiffness Matrix Construction

#### Fine-Scale FEM

$$
S = \text{triangle\_csc}(a,b,c),\quad \text{rhs} = h \, f(x)
$$

#### Homogenization Model (Two-Scale)

$$
\begin{aligned}
S &= 
\begin{bmatrix}
\text{kron}(M, A_0) & \text{kron}(B_0, \cdot) \\
\cdots & \cdots
\end{bmatrix}, \\
\text{rhs} &= \text{macro/micro source term}
\end{aligned}
$$

### Step 4: Build Hamiltonian for Quantum Solver

- Choose **original** or **homogenization** model

### Step 5: Quantum Schrödingerization Solver

$$
u = \text{schro}(-S, u_0, na, R, T, b=\text{rhs}, \text{scale\_b}=1)
$$

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

### Step 6: Extract Macro/Micro Solutions

$$
u_1 = u[:-N_x].\text{reshape}(N_x, N_x), \quad u_0 = u[-N_x:]
$$

### Step 7: Visualization & Circuit Export

- Plot $u_0$ and $u_1$
- Export quantum circuit diagram

------

## 4. Core Code Snippets

### 4.1 Tridiagonal Stiffness Matrix

```python
def triangle_csc(a,b,c,N=None):
    # Build symmetric tridiagonal FEM matrix
    return sparse_csc(a, range(N), range(N), N) + \
           sparse_csc(b, range(N-1), range(1,N), N) + \
           sparse_csc(c, range(1,N), range(N-1), N)
```

### 4.2 Two-Scale Solution Extraction

```python
u1, u0 = u[:-Nx].reshape(Nx, Nx), u[-Nx:]
u0 = get_u0(u0, x)
u1 = get_u1(u1, x, L)
```

### 4.3 Quantum Solver Call

```python
u = schro(-S, np.zeros_like(rhs), na=na, b=rhs)
```

------

## 5. Outputs

- Macro solution $u_0$, micro solution $u_1$
- 1D solution plot
- Quantum circuit diagram
- Grid and computation statistics

------

## 6. Trigger Phrases

- Solve multiscale elliptic equation using quantum method
- Simulate 1D highly oscillatory elliptic PDE
- Two-scale homogenization with quantum Schrödingerization
- Compute macro/micro solutions for multiscale PDE
- Quantum simulation for composite material PDE

------

## 7. Use Cases

- Multiscale simulation in composite materials
- High-contrast conduction/permeability problems
- Quantum acceleration for fine-grid multiscale PDEs
- Homogenization model validation

------

### Summary

This skill provides a **complete quantum solution for multiscale elliptic PDEs**:

- Supports both **original** and **two-scale homogenization** models
- Converts linear systems into quantum-solvable Schrödinger form
- Constructs FEM stiffness matrices automatically
- Outputs macro/micro solutions and quantum circuits
- Fully aligned with your algorithm and research