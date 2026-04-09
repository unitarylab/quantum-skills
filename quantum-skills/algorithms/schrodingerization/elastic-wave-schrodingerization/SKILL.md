---
name: elastic-wave-schrodingerization
description: Quantum solver for 1D/2D isotropic elastic wave equations using Schrödingerization. Supports homogeneous or variable coefficient media, Dirichlet/periodic boundaries, and classical/Trotter quantum solvers. Outputs physical fields, plots, and quantum circuit diagrams.
---

# Skill: Quantum Simulation of 1D/2D Elastic Wave Equations

## Basic Information

- **Skill ID**: `quantum_elastic_wave_simulation`
- **Skill Name**: Quantum Solver for 1D/2D Elastic Wave Equations via Schrödingerization
- **Domain**: Quantum Computing, Numerical PDEs, Geophysics, Material Mechanics
- **Core Capability**: Implements quantum solution for **1D/2D isotropic elastic wave equations**. Supports homogeneous/variable coefficient media, Dirichlet/periodic boundaries, and classical/Trotter solvers. Automatically outputs field solutions, plots, and quantum circuits.

---

## 1. Mathematical Background

### 1.1 Governing Equation

\[
\rho(\mathbf{x}) \frac{\partial^2 \mathbf{u}(\mathbf{x},t)}{\partial t^2} = \nabla \cdot \boldsymbol{\sigma}(\mathbf{x},t) + \mathbf{f}(\mathbf{x},t)
\]

- \(\mathbf{u}(\mathbf{x},t)\) — displacement  
- \(\boldsymbol{\sigma}(\mathbf{x},t)\) — stress tensor  
- \(\rho(\mathbf{x})\) — density  
- \(\mathbf{f}(\mathbf{x},t)\) — external force  

### 1.2 Velocity-Stress Form

\[
\begin{cases}
\frac{\partial \mathbf{v}}{\partial t} = \rho^{-1} \nabla \cdot \boldsymbol{\sigma} + \rho^{-1} \mathbf{f} \\
\frac{\partial \boldsymbol{\sigma}}{\partial t} = \mathbf{C} : \nabla \mathbf{v}
\end{cases}
\]

- \(\mathbf{v} = \partial_t \mathbf{u}\) — velocity  
- \(\mathbf{C}\) — stiffness tensor  

### 1.3 Schrödingerization

Transforms the first-order velocity-stress system into a **Hamiltonian evolution** suitable for quantum simulation:

\[
i \frac{d}{dt} |\psi(t)\rangle = H |\psi(t)\rangle
\]

Handles inhomogeneous media, forces, and boundaries.

---

## 2. Algorithm Steps (Pipeline)

### Step 1: Parse Parameters

Extract medium properties, domain, time, qubits, boundary type, and discretization scheme.

```python
rho = eq.get_parameter('ρ')
lam, mu = eq.get_parameter('λ'), eq.get_parameter('μ')
L, T, nx, ny, dx, dy = eq.get_domain()
bd = eq.boundary.type
Nx, Ny = 2**nx, 2**ny
```

### Step 2: Initial Conditions

Set velocity and stress initial states.

```python
v0 = eq.initial.v0(x, y)
sigma0 = eq.initial.sigma0(x, y)
```

### Step 3: Build Differential / System Matrices

Construct first-order derivative matrices and system block matrices for velocity-stress form.

```python
A_x, A_y = first_order_derivative_2D(Nx, Ny, dx, dy, bd)
zero_block = sp.csc_matrix((Nx*Ny, Nx*Ny))
# 5x5 block for 2D velocity-stress system
A = sp.bmat([[zero_block, A_x/rho, A_y/rho],
             [A_x*(lam+2*mu), zero_block, zero_block],
             [A_y*(lam+2*mu), zero_block, zero_block]], format='csc')
```

### Step 4: Quantum Schrödingerization Solver

Perform quantum evolution for the constructed Hamiltonian.

```python
psi_final = schro(A, initial_state=(v0, sigma0), T=T, na=na, R=R, order=order)
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
### Step 5: Trotter Quantum Circuit (Optional)

Apply Trotter decomposition for large systems.

```python
H1, H2 = split_Hamiltonian(A)
psi_final, qc = schro_trotter(H1, H2, initial_state=(v0, sigma0), Nt=Nt, na=na)
```

### Step 6: Visualization & Output

Reconstruct displacement/velocity/stress fields and generate plots.

```python
plot_fields(x, y, psi_final.v, psi_final.sigma)
export_circuit(qc, filename='elastic_wave_circuit.svg')
```

------

## 3. Trigger Phrases

- Quantum simulation of 1D/2D elastic waves
- Solve elastic wave equation via Schrödingerization
- Generate quantum circuit for velocity/stress fields
- Variable-coefficient elastic wave quantum solver

------

## 4. Use Cases

- Material mechanics: internal stress wave simulation
- Geophysics: quantum seismic wave modeling
- Quantum computing education: PDE-to-quantum encoding
- High-dimensional quantum-accelerated elastic wave simulation

------

## Summary

This skill provides a **step-by-step quantum simulation** framework for 1D/2D elastic wave equations, fully aligned with **Burgers’ / Advection / Black–Scholes style steps**:

- Standardized parameter parsing
- System matrix construction
- Schrödingerization solver with optional Trotter decomposition
- Visualization and standardized input/output JSON
- Supports homogeneous / variable media, multiple solvers, and quantum circuit export