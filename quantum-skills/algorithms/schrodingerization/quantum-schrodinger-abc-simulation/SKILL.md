---
name: quantum-schrodinger-abc-simulation
description: Quantum simulation of 2D Schrödinger equation with artificial (absorbing) boundary conditions (ABC) using Schrödingerization. Restores unitary evolution for non-unitary dynamics, outputs wavefunction, 3D visualization, and quantum circuit diagrams.
---

# Skill: Quantum Dynamics Simulation with Artificial Boundary Conditions (2D Schrödinger)

## Basic Information

- **Skill ID**: `quantum_schrodinger_abc_simulation`
- **Skill Name**: Quantum Dynamics Simulation with Artificial Boundary Conditions (ABC)
- **Domain**: Quantum Computing, Quantum Dynamics, Scientific Computing, PDEs
- **Core Capability**: Solve the **2D Schrödinger equation with non-unitary ABC** using Schrödingerization, enabling quantum simulation of open systems.

------

## 1. Theoretical Foundation

### 1.1 Background & Challenge

- Artificial boundary conditions (ABC) allow **finite-domain simulation of open systems**.
- ABC introduces **non-unitary dynamics**, which cannot be directly simulated on quantum computers.
- **Schrödingerization** converts the non-unitary evolution into a unitary Hamiltonian system suitable for quantum computation.

### 1.2 Governing Equation

Time-dependent Schrödinger equation:
$$
i \frac{\partial \psi}{\partial t} = H \psi, \quad H = -\frac{1}{2}\Delta + V(x)
$$
**Artificial Boundary Condition** (example):
$$
\psi(L,t) = \alpha \psi(L-\Delta x, t) + \beta \partial_x \psi|_{x=L}
$$

------

### 1.3 Computational Complexity

Hamiltonian simulation complexity (sparse Hamiltonian):
$$
O\Big(s \|H\|_\text{max,1} \log\log(s \|H\|_\text{max,1} / \epsilon) \log(s \|H\|_\text{max,1}/\epsilon)\Big)
$$

------

## 2. Core Algorithm & Implementation

### Step 1: Absorbing Potential Construction

- Smooth potential near boundaries to suppress reflections:

```python
# Absorbing potential near edges
sigma[x_mask] += amplitude * (np.abs(X[x_mask]) - rcut)**2
sigma[y_mask] += amplitude * (np.abs(Y[y_mask]) - rcut)**2
```

### Step 2: Hamiltonian Matrix Assembly

- Laplacian + internal potential + ABC:

```python
# 2D Laplacian operator
D = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(Nx, Nx))
Z_mat = kron(D, I) + kron(I, D)

# Full Hamiltonian with ABC
H0 = Z_mat / (2*dx**2) - diags(V, 0)
H1 = diags(sigma, 0)  # Absorbing boundary potential
A = H0 * 1j - H1
```

### Step 3: Schrödingerization Quantum Solver

```python
from engine.library import schro_classical
# Solve 2D Schrödinger with ABC
u = schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point)
u = u.reshape(Nx, Nx)
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

## 3. Execution Pipeline

1. **Parameter Parsing**: domain size, ABC parameters `rcut`/`amplitude`, qubits
2. **Grid & Potential Setup**: generate 2D grid, internal potential, absorbing boundary
3. **Hamiltonian Construction**: Laplacian + potential + ABC terms
4. **Quantum Simulation**: Schrödingerization solver
5. **Visualization**: 3D wavefunction + quantum circuit diagram

------

## 5. Supported Features

- 2D Schrödinger equation simulation with **ABC / absorbing boundaries**
- Converts non-unitary dynamics to unitary via **Schrödingerization**
- Adjustable boundary potential strength and range
- Classical matrix exponentiation for quantum solver
- Automatic 3D wavefunction visualization
- Quantum circuit generation

------

## 6. Trigger Phrases

- Simulate 2D quantum dynamics with artificial boundary conditions
- Solve Schrödinger equation with absorbing boundaries using quantum method
- Run ABC Schrödinger simulation and visualize wavefunction
- Quantum simulation for open quantum systems on finite grids

------

### Summary

This skill enables **quantum simulation of open quantum systems**:

- Handles non-unitary artificial boundary conditions with Schrödingerization
- Implements smooth absorbing potentials for finite-domain simulation
- Provides full 2D simulation with wavefunction, visualization, and quantum circuit outputs
- Matches the complexity and numerical framework of state-of-the-art ABC quantum solvers