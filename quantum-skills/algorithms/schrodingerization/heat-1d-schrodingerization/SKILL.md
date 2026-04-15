---
name: heat-1d-schrodingerization
description: A quantum-compatible solver for the 1D Heat Equation using Schrödingerization to transform the non-unitary diffusion equation into a unitary evolution problem. Supports Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution with automatic circuit generation and solution visualization.
---

## One Step to Run 1D Heat Equation Example
```bash
python ./scripts/algorithm.py
```

# Skill: Quantum Simulation of 1D Heat Equation

## 1. Mathematical Formulation

### 1.1 1D Heat Equation

$$
\frac{\partial u}{\partial t} = a \frac{\partial^2 u}{\partial x^2} + f(x)
$$

- $a > 0$: diffusion coefficient
- $f(x)$: source term

### 1.2 Schrödingerized Hamiltonian

$$
\frac{d\psi}{dt} = -i H \psi
$$

with simplified Hamiltonian for periodic, source-free cases:
$$
H \approx -a \hat{\eta} \otimes \hat{p}^2 \approx a D_{\eta} \otimes D^\Delta
$$

- $\hat{\eta}$: auxiliary operator introduced by Schrödingerization
- $D_\eta$: discretized auxiliary operator
- $D^\Delta$: discrete Laplacian

> Note: General source or non-periodic BC requires full Schrödingerization.

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

## 2. Supported Features

- PDE Type: 1D parabolic diffusion
- Boundary Conditions: **Dirichlet**, **Periodic**
- Initial Conditions: sine, Gaussian, custom
- Solvers: Classical matrix exponentiation, Trotter splitting
- Automatic finite-difference Laplacian assembly
- Visualization + quantum circuit export

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Input Parameters

```python
L, T, nx, na, R, order, point, f0 = eq.get_common_coefficients()
bd = eq.boundary.type

Nx = 2**nx
dx = L / Nx
x = np.arange(0, L, dx)
u0 = f0(x)  # initial condition
```

------

### Step 2: Discretization

Construct 2nd-order differential operator:

```python
A0, b0 = second_order_derivative(
    N=Nx,
    dx=dx,
    boundary_condition=bd
)

A = a * A0
b = a * b0 + f(x)
```

------

### Step 3: Schrödingerization

Transform non-unitary diffusion equation:
$$
\frac{du}{dt} = A u + b \quad \longrightarrow \quad \frac{d\psi}{dt} = -i H \psi
$$

- Ensures unitary evolution
- Required for general source or non-periodic BC

------

### Step 4: Time Evolution

#### Classical Schrödinger Solver

```python
u = schro(
    A,
    u0,
    T=T,
    na=na,
    R=R,
    order=order,
    point=point,
    b=b
)
```

#### Trotterized Quantum Evolution (Optional)

Split Hamiltonian $H = H_1 + H_2$ and apply Trotter decomposition:
$$
e^{-iHt} \approx \left(e^{-i H_1 \Delta t} e^{-i H_2 \Delta t}\right)^{N_t}
$$

```python
u, qc = schro(
    u0=u0,
    H1=H1,
    H2=H2,
    Nt=Nt,
    na=na,
    R=R,
    order=order,
    point=point,
    b=b,
    theta=theta
)
```

------

### Step 5: Visualization

```python
ax.plot(x, u, "r-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

- Generates 1D solution plot
- Optional animation for time evolution

------

### Step 6: Quantum Circuit Export

```python
qc.draw(filename="heat_1d_circuit.svg")
```

- Full circuit for Hamiltonian simulation
- Includes Trotter decomposition if used

------

## 4. Boundary Conditions

- **Dirichlet:** $u(0,t)=0, u(L,t)=0$
- **Periodic:** $u(0,t) = u(L,t)$

------

## 5. Initial Conditions

- Sine: $u(x,0) = \sin(2 \pi x / L)$
- Gaussian: $u(x,0) = \exp(-x^2)$
- Custom: user-defined $f_0(x)$

------

## 6. Finite-Difference Scheme

- **Central Difference** (2nd-order Laplacian):

$$
\Delta u_i \approx \frac{u_{i+1} - 2 u_i + u_{i-1}}{\Delta x^2} \Delta t
$$

- Automatically assembled into matrix $A$

------

## 7. Outputs

- Solution array $u(x,T)$
- 1D plot of solution
- Full quantum circuit diagram
- Optional Trotter decomposition diagrams

------

## 8. Trigger Phrases

- Quantum simulation of 1D heat equation
- Schrödingerization-based solver for diffusion PDE
- Trotter quantum evolution of parabolic PDE

------

## 9. Use Cases

- 1D diffusion and conduction problems
- Heat transfer simulations
- Benchmarking quantum PDE algorithms
- Educational demonstrations of Schrödingerization

------

### Summary

- Standardized **step-by-step quantum simulation** pipeline for 1D Heat Equation
- Handles Dirichlet/Periodic BCs and custom initial conditions
- Supports classical and Trotterized quantum evolution
- Automates Laplacian assembly, visualization, and circuit export
- Fully consistent with **Advection / Burgers / General Linear PDE / Elastic Wave** skill style