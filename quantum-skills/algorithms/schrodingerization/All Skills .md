# Skills: Advection Equation (Schrödingerization-based Solver)

## Description

This skill solves the Advection Equation using a Schrödingerization algorithm.

The core idea is to transform the classical PDE into a unitary evolution problem and solve it via Hamiltonian simulation.

## Mathematical Model

### Advection Equation

$$
\frac{\partial u}{\partial t}
\;=\;
a\,\frac{\partial u}{\partial x}
$$

- **a**: Convection velocity
- Describes transport phenomena in fluids, atmosphere, and porous media

------

## Schrödingerization Framework

The Schrödinger framework can be referred to in ～/Schr.markdown.

------

## Special Case: Direct Unitary Form

### Periodic + Central Difference

$$
H \;=\; -a \hat{p}
\;\approx\;
a\,\frac{D^{\pm}}{i}
$$

This form is valid if:

1. Central difference (symmetric matrix)
2. Periodic boundary conditions

In this case:

- The discretized system is already unitary
- No Schrödingerization is required

------

## Algorithm Pipeline

### Stage 1: Parameter Parsing

Extract:

- Domain $[0, L]$
- Final time $T$
- Grid size $N = 2^{n_x}$
- Boundary condition
- Initial condition $u(x,0)$
- Velocity $a$

------

### Stage 2: Discretization

Construct grid:

```
Nx = 2**nx
dx = L / Nx
x = np.arange(0, L, dx)
```

Construct the difference matrix and right-hand side terms of the first derivative up to the difference scheme you use:

```
A0, b0 = first_order_derivative(
    N=Nx,
    dx=dx,
    boundary_condition=bd,
    scheme=scheme
)
```

Form system:

```
A = a * A0
b = a * b0
```

------

### Stage 3: Schrödingerization

Transform:
$$
\frac{du}{dt} = A u + b
\quad \longrightarrow \quad
\frac{d\psi}{dt} = -i H \psi
$$

- If $A$ is Hermitian:
   $H = iA$
- Otherwise:
   Apply a general Schrödingerization procedure 

------

### Stage 4: Time Evolution

#### Classical Schrödinger Solver

```
from engine.library import schro_classical as schro
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

------

#### Trotterized Quantum Evolution

Split the Hamiltonian:
$$
H = H_1 + H_2
$$
Apply Trotter decomposition:
$$
e^{-iHt}
\approx
\left(e^{-iH_1 \Delta t} e^{-iH_2 \Delta t}\right)^{N_t}
$$
Code:

```
from engine.library import schro_trotter as schro
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

### Stage 5: Output

- Solution $u(x,T)$
- Grid information
- Quantum circuit
- Visualization results

------

## Inputs

```
{
  "a": 1.0,
  "L": 1.0,
  "T": 1.0,
  "nx": 6,
  "na": 2,
  "boundary": "periodic | dirichlet",
  "scheme": "central | upwind",
  "solver": {
    "type": "classical | trotter",
    "dt": 0.01
  }
}
```

------

## Outputs

```
{
  "status": "ok",
  "x": [...],
  "u": [...],
  "grid": {
    "n_points": int,
    "dx": float
  },
  "circuit": [...],
  "plot": "solution.svg"
}
```

------

## Notes

- Central difference with periodic boundary leads to a unitary system
- Upwind schemes or Dirichlet boundary conditions require Schrödingerization
- Accuracy depends on:
  - grid size $n_x$
  - time step $\Delta t$
  - decomposition order

------

## Example

```
algo = AdvectionEquationAlgorithm()
result = algo.run(params)
```

------

## Summary

This skill provides a unified framework:

Classical PDE → Discretization → Schrödingerization → Quantum evolution

It connects numerical PDE methods with quantum simulation techniques.

# Skills:  **ill-posed PDEs**(1D backward heat)

This agent is designed to solve **ill-posed PDEs** using **Schrödingerization**, focusing on backward heat equations and similar problems. It integrates classical numerical methods with quantum-inspired algorithms and provides visualization of both solution and quantum circuits.

---

## 1. Solving Ill-Posed Problems

**Description:**  
Numerically solve ill-posed or unstable problems such as backward heat equations. Uses **warped phase transformation** and Schrödingerization to transform exponential growth problems into well-behaved evolution equations.

**Example Equation:**

$$
\partial_t u = H u, \quad u(0) = u_0
$$

Transformed via Schrödingerization:

$$
w(t,p) = e^{-p} u(t), \quad \frac{d}{dt} w = - H \partial_p w
$$

**Code Example:**

```python
# Transform and solve using Schrödingerization
w0 = np.exp(-p) * u0
u_t = np.exp(p) * w_t  # recover solution
```

## 2. Classical Matrix Exponentiation Method

**Description:**
 Solve backward heat equations using classical linear algebra.

**Workflow:**

1. Construct finite-difference matrix `A`.
2. Compute solution with matrix exponentiation.
3. Visualize the solution.

**Code Example:**

```
from engine.library import CDiff, schro_classical

# Construct finite-difference matrix
A = -a * CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()
u = schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point)
```

------

## 3. Lie-Trotter Decomposition Method

**Description:**
 Time evolution using Trotter decomposition for quantum-inspired simulation.

**Workflow:**

1. Split Hamiltonian into H1 and H2.
2. Apply discrete time evolution in steps.
3. Reconstruct solution after `Nt` steps.

**Code Example:**

```
from engine.library import TDiff, schro_trotter

func1, func2 = (a * TDiff(nx, dx, 2, scheme=scheme, boundary=bd)).data()
H1 = func1(dt / R)
H2 = func2(dt)

u, qc = schro_trotter(u0=u0, H1=H1, H2=H2, Nt=Nt, na=na, R=R, order=order, point=point)
```

------

## 4. Block Encoding (Future Support)

**Description:**
 Prepare for block encoding quantum simulation of PDEs. Currently defaults to classical method.

**Code Example:**

```
# Placeholder for block encoding
u = backHeatEquationAlgorithm._solve_block(eq)
```

------

## 5. Boundary Conditions Handling

- **Dirichlet:** `u(0,t) = u(L,t) = 0`
- **Periodic:** `u(0,t) = u(L,t)`
- **Neumann:** derivative at boundary

**Code Snippet:**

```
if bd == "periodic":
    x = np.arange(0, L, dx)
elif bd == "neumann":
    x = np.arange(0, L+dx, dx)
```

------

## 6. Visualization

- **Solution Plot:**

```
fig, ax = plt.subplots()
ax.plot(x, u)
ax.set_title("1D Heat Equation Solution")
fig.savefig("solution.svg")
```

- **Quantum Circuit Plot:**

```
qc.draw(filename="circuit_full.svg", title="Quantum Circuit")
H1.decompose().draw(filename="circuit_H1.svg")
H2.decompose().draw(filename="circuit_H2.svg")
```

------

## 7. Workflow Summary

1. **Parse parameters** from JSON input.
2. **Choose solver method**: classical, Trotter, or block.
3. **Construct matrices or quantum circuits**.
4. **Run solver** and compute `u(t)`.
5. **Generate visualizations** of solution and circuits.
6. **Return results** including solution grid, values, and images.

**Example Usage:**

```
algo = backHeatEquationAlgorithm()
result = algo.run(params=json_params)
print(result["u"])  # solution array
```

# Skill: Solve 2D Backward Heat Equation (Schrödingerization)

**Description:**  
This skill allows the agent to solve the **two-dimensional backward heat equation** using classical matrix exponentiation, Lie-Trotter decomposition, or future block encoding. It supports **Dirichlet, Periodic, and Neumann boundary conditions** and can generate solution plots and quantum circuit diagrams.

---

## Equation

$$
\frac{\partial u}{\partial t} = -a_1 \frac{\partial^2 u}{\partial x^2} - a_2 \frac{\partial^2 u}{\partial y^2} + \text{source}(x, y)
$$

**Parameters:**

- `a1`, `a2`: Diffusion coefficients along `x` and `y`.
- `L`: Domain length in each dimension.
- `T`: Final time.
- `nx`: Number of spatial qubits (log2 of grid points in x/y).
- `na`: Number of auxiliary qubits.
- `dt`: Time step for Trotter method.
- `boundary`: `"dirichlet"`, `"periodic"`, or `"neumann"`.
- `source(x, y)`: Source function.
- `f0(x, y)`: Initial condition function.
- `solver.type`: `"classical"`, `"trotter"`, or `"block"`.

---

## Skill Steps

### 1. Parse Parameters

- Extract PDE coefficients, domain, initial condition, source function, boundary conditions, solver type, and numerical parameters from JSON input.

### 2. Choose Solver Method

- **Classical**: Classical matrix exponentiation.
- **Trotter**: Lie-Trotter decomposition method.
- **Block**: Placeholder for block encoding (currently fallback to classical).

---

### 3. Construct Spatial Grid

- Compute `Nx = 2^nx` grid points per dimension.
- Compute `dx = L / (Nx + 1)` (or `dx = L / Nx` for periodic).
- Generate `x` and `y` arrays.
- Evaluate initial condition `u0 = f0(x[:, None], y[None, :])` and flatten to 1D.

---

### 4. Build Differential Operators

**Classical Method:**

1. Build 1D difference matrix: `A0 = CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()`.
2. Construct 2D Laplacian using Kronecker products:  
   `A = a1 * kron(A0, I) + a2 * kron(I, A0)`.
3. Prepare RHS vector:  
   `b = a1 * kron(source(x), ones(Nx)) + a2 * kron(ones(Nx), source(x))`.

**Trotter Method:**

1. Construct 1D derivative operator: `func1 = TDiff(nx, dx, 2, scheme=scheme, boundary=bd)[0]`.
2. Build H1 circuit: `H1.append(D1(a1), range(nx))`, `H1.append(D1(a2), range(nx, 2*nx))`.
3. H2 is `None` (for backward heat equation).

---

### 5. Execute Solver

- **Classical**: `u = schro(A, u0, T=T, na=na, R=R, order=order, point=point, b=b)`.
- **Trotter**: `u, qc = schro(u0=u0, H1=H1, H2=H2, Nt=Nt, na=na, R=R, order=order, point=point)`.

---

### 6. Reshape Solution

- Reshape `u` back to 2D: `u = u.reshape((Nx, Nx))`.

---

### 7. Generate Visualization

- 3D surface plot of `u(x, y, T)`.
- Save as `.svg` in algorithm folder.
- Use `matplotlib` non-interactive backend.

**Example Plot Code:**

```python
fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='viridis')
fig.savefig("solution.svg")
```

### 8. Generate Quantum Circuit Diagrams

- Full circuit: `qc.draw(filename="circuit_full.svg")`.
- H1 circuit: `H1.decompose().draw(filename="circuit_H1.svg")`.
- H2 circuit: `H2.decompose().draw(filename="circuit_H2.svg")`.
- Returns array of plot file information.

------

### 9. Return Results

```
{
  "status": "ok",
  "message": "2D Heat equation solved",
  "grid": {"n_points": 2^nx, "dx": dx, "dt": dt, "nt": Nt},
  "x": [...],
  "y": [...],
  "u": [...],
  "circuit": [...],
  "plot": {"format": "svg", "filename": "solution.svg"}
}
```

------

**Notes for Agent:**

- Use **Classical** for simple, small grid calculations.
- Use **Trotter** for quantum-inspired step-wise evolution.
- **Block encoding** will be supported in future.
- Ensure proper boundary handling (`dirichlet`, `periodic`, `neumann`).
- Flatten 2D grid for solver input, then reshape output for visualization.
- Generate plots and circuits after computation to provide full result package.



# Skill: Quantum Simulation for Black–Scholes Equation

## Basic Information

- **Skill ID**: `quantum_black_scholes_1d`
- **Skill Name**: Quantum Solver for 1D Black–Scholes Equation (European Option Pricing)
- **Domain**: Quantum Computing, Computational Finance, PDEs
- **Core Purpose**: Solve the **Black–Scholes option pricing PDE** using Schrödingerization for quantum simulation. Supports log-price formulation, boundary treatment for far-field conditions, and both classical matrix exponential and Trotter time-splitting quantum solvers.

------

## 1. Mathematical Background

### 1.1 Log-Price Black–Scholes Equation

∂t∂u=a(r−2σ2)∂x∂u+2σ2∂x2∂2u−ru

- x=logS: log stock price
- r: risk-free rate
- σ: volatility
- K: strike price

### 1.2 Initial Condition (European Put Payoff)

u(x,0)=max(−K,−ex)

### 1.3 Boundary Conditions

u→−ex as x→−∞,u→−Ke−rt as x→+∞

### 1.4 Schrödingerized Hamiltonian

H=η^⊗(∣0⟩⟨0∣⊗H1−∣1⟩⟨1∣⊗rI+X⊗∣1⟩⟨1∣n)+I⊗(∣0⟩⟨0∣⊗H2+Y⊗∣1⟩⟨1∣n)

------

## 2. Supported Features

- 1D Black–Scholes equation in log-price coordinates

- European put option payoff

- Far-field boundary treatment via auxiliary variables

- Quantum solvers:

  - Classical matrix exponentiation
  - Trotter–Lie splitting

  

- Automatic finite-difference discretization

- Option price visualization (in real stock price S)

- Quantum circuit diagram output

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Financial & Numerical Parameters

Extract risk-free rate, volatility, strike price, domain, and quantum resources.

```
r = eq.get_parameter('r')
sigma = eq.get_parameter('σ')
K = eq.get_parameter('K')
L1 = np.log(1e-4)
L2 = np.log(10 * K)
Nx = 2**nx
dx = (L2 - L1) / (Nx + 1)
x = np.linspace(L1 + dx, L2 - dx, Nx)
```

### Step 2: Set Initial Condition & Augmented State

Construct initial option payoff and extend state for boundary treatment.

```
u0 = f0(x)  # European put payoff
scale_b = K * (0.5 * sigma**2 / dx**2 + 0.25 * (-sigma**2 + 2*r) / dx)
u0 = np.concatenate((u0, np.ones(Nx) * scale_b))
```

### Step 3: Assemble Black–Scholes FD Matrix

Build drift, diffusion, and discount terms.

```
A1, b1 = first_order_derivative(N=Nx, dx=dx)
A2, b2 = second_order_derivative(N=Nx, dx=dx)
A = (r - 0.5*sigma**2) * A1 + 0.5*sigma**2 * A2
A = A - r * eye(2 * Nx)
```

### Step 4: Quantum Schrödingerization Solver

```
u = schro(A, u0, T=T, na=na, R=R, order=order)
u = u[:Nx] + np.exp(x)  # convert back to option price
```

### Step 5: Trotter Quantum Circuit (Optional)

Construct decomposed Hamiltonian blocks.

```
H1 = GateSequence(nx+1)
func1 = (sigma**2/2 * TDiff(nx, dx, order=2)).data()[0]
H1.append(func1(dt/R), target=range(nx), control=nx)

H2 = GateSequence(nx+1)
func2 = ((r-sigma**2/2) * TDiff(nx, dx, order=1)).data()[1]
H2.append(func2(dt), target=range(nx), control=nx)
```

### Step 6: Run Time Evolution

```
u, qc = schro(u0=u0, H1=H1, H2=H2, Nt=Nt, na=na)
```

### Step 7: Visualization & Output

Plot option price against real stock price S.

```
ax.plot(np.exp(x[:end]), u[:end])
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Augmented State for Boundary Treatment

```
# Augment state to handle far-field boundary condition
u0 = np.concatenate((u0, np.ones(Nx) * scale_b))
```

### 4.2 Black–Scholes Operator Assembly

```
# BS operator = drift + diffusion - interest rate
A = (r - 0.5*sigma**2) * A1 + 0.5*sigma**2 * A2 - r * eye(2*Nx)
```

### 4.3 Quantum Solver Call

```
# Core quantum evolution for BS PDE
u = schro(A, u0, T=T, na=na, R=R)
```

### 4.4 Convert to Real Stock Price

```
# Map log-price x back to real price S = exp(x)
S = np.exp(x[:end])
option_price = u[:end]
```

------

## 5. Boundary & Initial Conditions

### Initial Condition

European put payoff in log-price:

u(x,0)=max(−K,−ex)

### Boundary Conditions

- Left: u→−ex
- Right: u→−Ke−rt (handled via auxiliary variable)

------

## 6. Discretization Scheme

Central difference for drift + diffusion:

Δui=(r−2σ2)2ΔxΔt(ui+1−ui−1)+2σ2Δx2Δt(ui+1−2ui+ui−1)−rΔtui

------

## 7. Outputs

- European option price V(S,T)
- 1D plot: option price vs stock price
- Full quantum circuit diagram
- H1 / H2 decomposed circuits (Trotter)

------

## 8. Trigger Phrases

- Solve Black–Scholes equation using quantum method
- Quantum simulation for European option pricing
- Compute option price with Schrödingerization
- Quantum PDE solver for Black–Scholes model
- Financial quantum computing for option pricing

------

## 9. Use Cases

- European option pricing
- Quantum acceleration for computational finance
- PDE benchmarking in financial modeling
- Risk analysis and volatility simulation

------

### Summary

This skill provides a **complete quantum solution for the Black–Scholes equation**:

- Uses log-price formulation and far-field boundary treatment
- Supports classical and Trotter quantum solvers
- Automatically maps results to real stock prices
- Outputs professional financial visualizations and quantum circuits
- Fully aligned with your implementation and mathematical framework



# Skill: Quantum Simulation for 1D Burgers’ Equation

## Basic Information

- **Skill ID**: `quantum_burgers_1d`
- **Skill Name**: Quantum Solver for 1D Viscous Burgers’ Equation
- **Domain**: Quantum Computing, Computational Fluid Dynamics, Nonlinear PDEs
- **Core Purpose**: Solve the **1D viscous Burgers’ equation** using quantum Schrödingerization + **Cole-Hopf transformation** to linearize the nonlinear PDE. Supports shock wave capturing, viscous smoothing, and both classical matrix exponential and Trotter time-splitting quantum solvers.

------

## 1. Mathematical Background

### 1.1 Burgers’ Equation

∂t∂u+u∂x∂u=ν∂x2∂2u

- ν: kinematic viscosity (diffusion strength)
- Nonlinear convection + linear diffusion
- Shock wave formation model

### 1.2 Linearization via Cole-Hopf Transformation

The key step: convert nonlinear Burgers’ equation into **linear heat equation**:

u=−2νϕϕx

where ϕ satisfies:

∂t∂ϕ=ν∂x2∂2ϕ

### 1.3 Schrödingerized Hamiltonian

H=lp^−νp^2

Valid for **periodic BC + symmetric difference (Hermitian)**.

------

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

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Parameters

Extract viscosity, domain, time, qubits, boundary, scheme.

```
nu = eq.get_parameter('ν')
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type
Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
u0 = f0(x)
```

### Step 2: Cole-Hopf Transformation (Linearization)

Transform initial condition u0→ϕ0.

```
I = cumulative_trapezoid(u0, x, initial=0.0)
exponent = -I / (2.0 * nu)
exponent = exponent - np.max(exponent)
phi0 = np.exp(exponent)
phi0 = np.clip(phi0, 1e-300, np.inf)
```

### Step 3: Build Diffusion Matrix

Assemble Laplacian for linear heat equation.

```
A0, b0 = second_order_derivative(N=Nx, dx=dx, boundary_condition=bd)
A = A0 * nu
b = b0 * nu
```

### Step 4: Quantum Schrödingerization Solver

```
phi = schro(A, phi0, T=T, na=na, R=R, order=order, b=b)
qc = circuit_classical(nx, na)
```

### Step 5: Invert Cole-Hopf to Recover u

Compute derivative and reconstruct solution.

```
phi_x_over_phi[1:-1] = (phi[2:] - phi[:-2]) / (2 * dx) / phi[1:-1]
u = -2.0 * nu * phi_x_over_phi
```

### Step 6: Trotter Quantum Circuit (Optional)

```
func1, func2 = (nu * TDiff(nx, dx, 2, boundary=bd)).data()
H1 = func1(dt / R)
phi, qc = schro(u0=phi0, H1=H1, H2=None, Nt=Nt, na=na)
```

### Step 7: Visualization & Output

Plot shock / smooth solution u(x,T).

```
ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Cole-Hopf Transformation (Linearize Nonlinear PDE)

```
# Convert nonlinear Burgers to linear heat equation
I = cumulative_trapezoid(u0, x, initial=0.0)
phi0 = np.exp(-I / (2.0 * nu))
```

### 4.2 Laplacian Assembly for Diffusion

```
# Build linear diffusion operator
A0, b0 = second_order_derivative(N=Nx, dx=dx, boundary_condition=bd)
A = A0 * nu
```

### 4.3 Quantum PDE Solver Call

```
# Core quantum evolution for linear heat equation
phi = schro(A, phi0, T=T, na=na, R=R)
```

### 4.4 Reconstruct Burgers Solution

```
# Invert Cole-Hopf: u = -2ν ∂xφ / φ
phi_x_over_phi = np.gradient(phi, dx) / phi
u = -2.0 * nu * phi_x_over_phi
```

------

## 5. Boundary & Initial Conditions

### Boundary Conditions

- **Dirichlet**: u(0,t)=g1, u(L,t)=g2
- **Periodic**: u(0,t)=u(L,t)

### Initial Conditions

- Sine: u(x,0)=bsin(2π(x+c)/L)
- Discontinuous: piecewise step for shock testing
- Gaussian: smooth initial profile

------

## 6. Discretization Schemes

### Central Difference

Δui≈−l2ΔxΔt(ui+1−ui−1)+νΔx2Δt(ui+1−2ui+ui−1)

### Upwind Difference

For convection stability:

Δui≈−lΔxΔt(ui−ui−1)+νΔx2Δt(ui+1−2ui+ui−1)

------

## 7. Outputs

- Shock / smooth solution u(x,T)
- 1D solution plot
- Full quantum circuit diagram
- H1 decomposed circuit (Trotter)

------

## 8. Trigger Phrases

- Solve 1D Burgers’ equation using quantum method
- Quantum simulation for viscous shock waves
- Quantum PDE solver with Cole-Hopf transformation
- Compute nonlinear convection–diffusion with quantum algorithm

------

## 9. Use Cases

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
- 

# Quantum Simulation of Elastic Wave Equations (1D/2D)

## Basic Information

- **Skill ID**: `quantum_elastic_wave_simulation`
- **Skill Name**: Quantum Simulation of Elastic Wave Equations Based on Schrödingerization Method
- **Domain**: Quantum Computing, Numerical PDE Solutions, Geophysics / Material Mechanics
- **Core Capability**: Implements quantum solution for **1D/2D isotropic elastic wave equations** based on the paper *Quantum Simulation of Elastic Wave Equations* (Shi Jin, Chundan Zhang). Supports homogeneous/variable coefficient media, Dirichlet/periodic boundary conditions, and two quantum solvers (classical matrix exponentiation, Trotter decomposition). Automatically outputs physical field solutions, visualizations, and quantum circuit diagrams.

## Theoretical Foundation

### 1. Governing Equation

Elastic wave equations describe the propagation of strain and stress inside an elastic medium under external disturbance:

ρ(x)∂t2∂2u(x,t)=∇⋅σ(x,t)+f(x,t)

### 2. Key Mathematical Forms

1. **Symmetric Matrix Form (Velocity-Stress System)**

   

   First-order hyperbolic system for homogeneous media:

⎩⎨⎧ρdtdv=i=1,2,3∑kiLiσ+fdtdσ=Ci=1,2,3∑kiLiTv

1. **Hyperbolic Displacement System**

   

   Reformulated wave equation for spectral / central difference discretization.

### 3. Core Method: Schrödingerization

Solves the non-unitary dynamics challenge (inhomogeneous media, forces, boundary conditions) for quantum simulation. Gate complexity bounds from the paper:

- SMF with force term:

  

  NGate=O((⌈log(d2+3d)⌉+rdlogϵ1)ϵ−1T)

- Variable-coefficient velocity-stress:

  

  NGate=O((⌈log(d2+3d)⌉+2dlogϵ1)ϵ−23T)

## Supported Features

表格

|        Dimension        |                   Supported Configurations                   |
| :---------------------: | :----------------------------------------------------------: |
|    **Spatial Dims**     |                            1D, 2D                            |
|       **Medium**        |     Homogeneous, spatially variable coefficients (ρ,λ,μ)     |
| **Boundary Conditions** |                     Dirichlet, Periodic                      |
|   **Quantum Solvers**   |            1D: Classical, Trotter; 2D: Classical             |
|   **Discretization**    |             Spectral method, Central difference              |
|       **Outputs**       | Velocity/stress fields, plots, quantum circuit diagrams, performance logs |

## Input Parameters (JSON Schema)

```
{
  "equation": {
    "type": "elastic_wave_1d / elastic_wave_2d",
    "parameters": {
      "ρ": 1.0,
      "λ": 1.0,
      "μ": 1.0
    },
    "boundary": {
      "type": "dirichlet / periodic"
    },
    "discrete": {
      "type": "spectral / central",
      "dt": 0.01
    },
    "initial": {
      "σ0(x)": "np.exp(-x**2)",
      "v0(x)": 0.0
    },
    "source": null
  },
  "solver": {
    "type": "classical / trotter",
    "L": 10.0,
    "T": 1.0,
    "nx": 3,
    "na": 2,
    "R": 10,
    "order": 1,
    "point": 0
  }
}
```

## Core Implementation Code Snippets

### 1. 1D System Matrix Construction

```
# Build first-order derivative matrix
A1, b1 = first_order_derivative(N=Nx, dx=dx, boundary_condition=bd, scheme=scheme)
zero_block = sp.csc_matrix((Nx, Nx))
# Elastic wave equation system matrix
A = sp.bmat([[zero_block, A1/rho], [A1*(lamb + 2*mu), zero_block]], format='csc')
```

### 2. 1D Trotter Hamiltonian Circuit

```
# Controlled differential quantum gate
H2 = GateSequence(nx+1)
H2.h(nx)
func1 = (kappa * TDiff(n=nx, dx=dx, order=1)).data()[1]
H2.append(func1(dt), target=range(nx), control=nx, control_sequence='0')
H2.append(func1(-dt), target=range(nx), control=nx, control_sequence='1')
H2.h(nx)
```

### 3. 2D Block Matrix Assembly

```
# 2D staggered grid derivatives
Dx = build_periodic_diff_matrix(Nx)
Dy = build_periodic_diff_matrix(Ny)
D1 = (sp.kron(Dx, I_Ny)) / dx
D2 = (sp.kron(I_Nx, Dy)) / dy
# 5x5 block system for velocity-stress fields
A_blocks = [[O for _ in range(5)] for _ in range(5)]
A_blocks[0][2] = R1 @ D1
A_blocks[2][0] = -C11 @ D1.T
A1 = sp.bmat(A_blocks, format='csc')
```

## Execution Pipeline (5 Standard Stages)

1. **Parameter Parsing**: Parse medium properties, qubits, boundaries, initial conditions
2. **Matrix Construction**: Build differential, system, and material coefficient matrices
3. **Quantum Simulation**: Run Schrödingerization solver and execute quantum circuits
4. **Visualization**: Generate 2D/3D velocity/stress field plots and animations
5. **Circuit Export**: Render full quantum circuit and Hamiltonian decomposition diagrams

## Output Format

```
{
  "status": "ok",
  "message": "Elastic wave equation solved",
  "grid": {"n_points": 8, "dx": 1.25, "dt": 0.01},
  "x": [0, 1.25, 2.5, ...],
  "v": [0.1, 0.23, ...],
  "sigma": [0.5, 0.9, ...],
  "circuit": [{"format": "svg", "filename": "xxx_circuit_full.svg"}],
  "plot": {"format": "svg", "filename": "xxx_solution.svg"}
}
```

## Trigger Phrases

- Simulate 1D elastic wave equation using quantum Schrödingerization
- Compute 2D variable-coefficient elastic wave propagation with periodic boundaries
- Solve elastic wave equation via Trotter decomposition and output quantum circuit
- Quantum simulation of elastic waves in homogeneous medium, generate velocity and stress field plots

## Use Cases

- Material mechanics: internal stress wave simulation
- Geophysics: quantum numerical solution for seismic wave propagation
- Quantum computing education: PDE-to-quantum-circuit encoding demonstration
- Scientific research: high-dimensional elastic wave simulation with quantum acceleration

------

### Summary

This is a **production-ready AI Skill** fully aligned with your paper and source code:

- Matches theoretical formulation and gate complexity bounds
- Supports 1D/2D, homogeneous/variable media, multiple solvers
- Standardized input/output, triggers, and visualization pipeline
- Embeds real code snippets from your implementation for direct integration



# Skill: 1D General Linear PDE via Schrödingerization

## Basic Information

- **Skill ID**: `quantum_general_linear_1d`
- **Skill Name**: Quantum Solver for 1D General Linear Evolution Equations
- **Domain**: Quantum Computing, Scientific Computing, PDEs
- **Core Purpose**: Solve the **general 1D linear time-dependent PDE** using quantum Schrödingerization. Supports arbitrary-order derivatives, multiple boundary conditions, and multiple quantum solvers.

------

## 1. Mathematical Formulation

### 1.1 General Linear Equation

∂t∂u=a∂x∂u+b∂x2∂2u+c∂x3∂3u+⋯+f(x)

### 1.2 Schrödingerized Hamiltonian (Periodic Form)

H=−aη^⊗p^+bη^⊗p^2+⋯

Valid only for:

- Hermitian discretization
- Periodic BC + symmetric scheme
- Source-free case f=0

------

## 2. Supported Features

- Arbitrary-order linear PDE: advection, diffusion, high-order derivatives

- Boundary conditions: **Dirichlet**, **Periodic**, **Neumann**

- Initial conditions: sine, discontinuous, custom

- Solvers:

  - Classical matrix exponentiation
  - Trotter splitting
  - Block encoding (fallback to classical)

  

- Automatic finite-difference matrix assembly

- Visualization + quantum circuit export

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Input Parameters

Extract domain, time, qubits, boundary, scheme, and derivatives.

```
L, T, source, nx, na, R, point, order, f0 = eq.get_common_coefficients()
derivative = eq.get_derivative_1d()    # e.g., {1: a, 2: b, 3: c}
bd = eq.boundary.type
scheme = eq.discrete.type

Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
u0 = f0(x)    # initial condition
```

### Step 2: Assemble Finite-Difference Matrix

Build differential operator for arbitrary-order derivatives.

```
A = ClassicalOperator()
for key, value in derivative.items():
    order = int(key)
    coeff = float(value)
    if coeff != 0:
        D = CDiff(N=Nx, dx=dx, order=order, scheme=scheme, boundary=bd)
        A += D * coeff
A = A.get_matrix()
b = eq.get_rhs_1d(...) + source(x)    # source term
```

### Step 3: Quantum Schrödingerization Solver

```
u = schro(A, u0, T=T, na=na, R=R, order=order, b=b, scale_b=1.0/T)
```

### Step 4: Trotter Solver (Optional)

For time-evolution splitting:

```
A = TrotterOperator()
# add derivative terms
func1, func2 = A.data()
H1 = func1(dt / R)
H2 = func2(dt)
u, qc = schro(u0, H1, H2, Nt=Nt, na=na, b=b, theta=dt/T)
```

### Step 5: Visualization

Plot solution u(x,T).

```
ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

### Step 6: Quantum Circuit Export

```
qc.draw(filename="..._circuit_full.svg")
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Arbitrary-Order Differential Operator

```
# Build D^k for any order k
D = CDiff(N=Nx, dx=dx, order=D_order, scheme=scheme, boundary=bd)
```

Supports 1st, 2nd, 3rd, ... derivatives.

### 4.2 General Linear System Assembly

```
# Sum all derivative terms: aD + bD² + cD³ + ...
A = ClassicalOperator()
for order, coeff in derivative.items():
    A += CDiff(..., order=order) * coeff
A = A.get_matrix()
```

### 4.3 Quantum Solver Call

python

```
# Core quantum evolution
u = schro(A, u0, T=T, na=na, b=b)
```

### 4.4 Trotter Time Splitting

python

```
H1 = func1(dt/R)
H2 = func2(dt)
u, qc = schro(u0, H1, H2, Nt=Nt)
```

------

## 5. Boundary Conditions

### Dirichlet

u(0,t)=0,u(L,t)=0

### Periodic

u(0,t)=u(L,t)

### Neumann

∂xu=0

------

## 6. Initial Conditions

- Sine: u(x,0)=sin(2πx/L)
- Discontinuous: piecewise step function
- Custom: user-defined f0(x)

------

## 7. Finite-Difference Scheme

### Central Difference

Δui≈Δx2Δt(ui+1−2ui+ui−1)

------

## 8. Outputs

- Solution array u(x,T)
- 1D solution plot
- Full quantum circuit diagram
- H1 / H2 decomposition diagrams (Trotter)

------

## 9. Trigger Phrases

- Solve 1D general linear PDE using quantum method
- Quantum simulation for advection–diffusion equation
- Run Schrödingerization for arbitrary-order linear PDE
- Compute time evolution for linear PDE with quantum solver

------

## 10. Use Cases

- Advection–diffusion problems
- Wave equations
- Heat equations
- General linear evolution systems
- Quantum PDE benchmarking

------

### Summary

This is a **universal quantum PDE skill** for **all 1D linear equations**:

- Supports arbitrary derivatives, boundaries, initial conditions
- Uses Schrödingerization for quantum simulation
- Provides classical, Trotter, and block solvers
- Automatically assembles FD matrices and exports circuits
- Fully aligned with your code and mathematical framework



# Skills: 1D Heat Equation (Schrödingerization-based Solver)

## Description

This skill solves the 1D Heat Equation using a Schrödingerization-based algorithm.

The core idea is to transform the diffusion equation into a unitary evolution problem via Schrödingerization, and then solve it using Hamiltonian simulation.

------

## Mathematical Model

### 1D Heat Equation

$$
\frac{\partial u}{\partial t}
\;=\;
a\,\frac{\partial^{2}u}{\partial x^{2}}
\;+\; f(x)
$$

- **a**: Diffusion coefficient ($a > 0$)
- **f(x)**: Source term

This equation describes heat conduction and diffusion processes in physical systems.

------

## Hamiltonian after Schrödingerization

### Periodic (Fourier) form

$$
H
\;=\;
-\,a\,\hat{\eta}\,\otimes\,\hat{p}^2
\;\;\approx\;\;
a\,D_{\eta}\,\otimes\,D^{\Delta}
$$

where:

- $\hat{\eta}$: auxiliary operator introduced by Schrödingerization
- $D_{\eta}$: discretized auxiliary operator
- $D^{\Delta}$: discrete Laplacian

------

### Validity of simplified form

This simplified Hamiltonian applies only if:

1. The discretized derivative operator is Hermitian
2. Periodic boundary conditions are used
3. Source term $f(x) = 0$

Otherwise, a general Schrödingerization procedure is required.

------

## Algorithm Pipeline

### Stage 1: Parameter Parsing

Extract:

- Domain $[0, L]$
- Final time $T$
- Grid size $N = 2^{n_x}$
- Boundary condition
- Initial condition $u(x,0)$
- Diffusion coefficient $a$
- Source term $f(x)$

------

### Stage 2: Discretization

Construct grid:

```
Nx = 2**nx
dx = L / Nx
x = np.arange(0, L, dx)
```

Build second-order differential operator:

```
A0, b0 = second_order_derivative(
    N=Nx,
    dx=dx,
    boundary_condition=bd
)
```

Form system:

```
A = a * A0
b = a * b0 + f(x)
```

------

### Stage 3: Schrödingerization

Transform:
$$
\frac{du}{dt} = A u + b
\quad \longrightarrow \quad
\frac{d\psi}{dt} = -i H \psi
$$

- Diffusion operator is not unitary
- Schrödingerization is always required (except special periodic source-free case)

------

### Stage 4: Time Evolution

#### Classical Schrödinger Solver

```
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

------

#### Trotterized Quantum Evolution

Split Hamiltonian:
$$
H = H_1 + H_2
$$
Apply Trotter decomposition:
$$
e^{-iHt}
\approx
\left(e^{-iH_1 \Delta t} e^{-iH_2 \Delta t}\right)^{N_t}
$$
Code:

```
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

### Stage 5: Output

- Solution $u(x,T)$
- Grid information
- Quantum circuit
- Visualization

------

## Inputs

```
{
  "a": 1.0,
  "L": 1.0,
  "T": 1.0,
  "nx": 6,
  "na": 2,
  "boundary": "periodic | dirichlet",
  "solver": {
    "type": "classical | trotter",
    "dt": 0.01
  }
}
```

------

## Outputs

```
{
  "status": "ok",
  "x": [...],
  "u": [...],
  "grid": {
    "n_points": int,
    "dx": float
  },
  "circuit": [...],
  "plot": "solution.svg"
}
```

------

## Notes

- Heat equation is inherently non-unitary
- Schrödingerization is required in general
- Periodic + source-free case allows simplified Hamiltonian
- Accuracy depends on:
  - grid size $n_x$
  - time step $\Delta t$
  - Trotter order

------

## Example

```
algo = HeatEquationAlgorithm()
result = algo.run(params)
```

------

## Summary

This skill provides a framework:

Diffusion PDE → Discretization → Schrödingerization → Quantum evolution

It enables solving parabolic equations using quantum simulation techniques.

# Skill: Quantum Simulation for 2D Forward Heat Equation

## Basic Information

- **Skill ID**: `quantum_heat_2d`
- **Skill Name**: Quantum Solver for 2D Forward Heat Equation
- **Domain**: Quantum Computing, Heat Conduction, PDEs, Thermal Simulation
- **Core Purpose**: Solve the **2D heat conduction PDE** on a planar domain using Schrödingerization quantum methods. Supports anisotropic diffusion, multiple boundary conditions, and classical/Trotter/block solvers.

------

## 1. Mathematical Background

### 1.1 2D Heat Equation

∂t∂u=a1∂x2∂2u+a2∂y2∂2u+f(x,y)

- a1,a2: diffusion coefficients in x/y directions
- u(x,y,t): temperature field
- f(x,y): heat source term

### 1.2 Schrödingerized Hamiltonian (Periodic Form)

H=−η^⊗(a1p^x2+a2p^y2)

Valid for:

- Hermitian symmetric difference
- Periodic BC in both directions
- Source-free case f=0

------

## 2. Supported Features

- 2D anisotropic heat conduction
- Boundary conditions: Dirichlet, Periodic
- Initial condition: 2D sine wave
- Quantum solvers:
  - Classical matrix exponentiation
  - Trotter–Lie splitting
  - Block encoding (fallback to classical)
- Tensor-product finite-difference discretization
- 3D surface / contour visualization
- Quantum circuit diagram output

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse 2D Domain & Parameters

Extract diffusion coefficients, domain size, qubits, boundary.

```
a1 = eq.get_parameter('a1')
a2 = eq.get_parameter('a2')
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type

Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
y = np.arange(dx, L, dx)
```

### Step 2: Initialize 2D Temperature Field

Create initial condition and flatten to 1D array.

```
u0 = f0(x[:, None], y[None, :])  # 2D initial condition
u0 = u0.flatten()                # flatten for solver
```

### Step 3: Assemble 2D Laplacian Matrix (Kronecker Product)

Construct x and y second derivatives using Kronecker product.

```
A0 = CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()

# 2D operator = a1·Lap_x + a2·Lap_y
A = a1 * sp.kron(A0, sp.eye(Nx)) + a2 * sp.kron(sp.eye(Nx), A0)
```

### Step 4: Build Source Term

```
b0 = source(x)
b = a1 * np.kron(b0, np.ones(Nx)) + a2 * np.kron(np.ones(Nx), b0)
```

### Step 5: Quantum Schrödingerization Solver

```
u = schro(A, u0, T=T, na=na, R=R, order=order, b=b)
u = u.reshape((Nx, Nx))  # reshape back to 2D
qc = circuit_classical(nx, na, dim=2)
```

### Step 6: Trotter Quantum Circuit (2D)

Construct decomposed 2D diffusion gates.

```
func1 = TDiff(nx, dx, 2, boundary=bd).data()[0]
D1 = lambda a: func1(a * dt/R)

H1 = GateSequence(2*nx)
H1.append(D1(a1), range(nx))      # x-direction
H1.append(D1(a2), range(nx, 2*nx))# y-direction
```

### Step 7: Run Trotter Evolution

```
u, qc = schro(u0=u0, H1=H1, H2=None, Nt=Nt, na=na)
u = u.reshape((Nx, Nx))
```

### Step 8: 3D Visualization

```
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='viridis')
```

------

## 4. Core Code Snippets with Explanation

### 4.1 2D Laplacian via Kronecker Product

```
# Build 2D Laplacian from 1D operators
A0 = CDiff(N=Nx, dx=dx, order=2).get_matrix()
A = a1 * kron(A0, I) + a2 * kron(I, A0)
```

- `kron(A0, I)`: Laplacian along x
- `kron(I, A0)`: Laplacian along y

### 4.2 2D Initial Condition

```
u0 = f0(x[:, None], y[None, :])
u0 = u0.flatten()
```

- Create 2D mesh initial condition
- Flatten for quantum solver

### 4.3 Quantum Solver Call (2D)

```
u = schro(A, u0, T=T, na=na, b=b)
u = u.reshape((Nx, Nx))
```

- Solve linear system
- Reshape back to 2D temperature field

### 4.4 2D Trotter Circuit Construction

```
H1 = GateSequence(2*nx)
H1.append(D1(a1), range(nx))
H1.append(D1(a2), range(nx, 2*nx))
```

- Separate x/y diffusion gates
- Apply to respective qubit registers

------

## 5. Boundary & Initial Conditions

### Boundary Conditions

- **Dirichlet**: u=0 on domain boundary
- **Periodic**: u(0,y)=u(Lx,y), u(x,0)=u(x,Ly)

### Initial Condition

2D sine wave:

u(x,y,0)=sin(πx/Lx)sin(πy/Ly)

------

## 6. Finite-Difference Scheme

Central difference for 2D Laplacian:

Δui,j=Δt[a1Δx2ui+1,j−2ui,j+ui−1,j+a2Δy2ui,j+1−2ui,j+ui,j−1]

------

## 7. Outputs

- 2D temperature field u(x,y,T)
- 3D surface plot
- Full quantum circuit diagram
- H1 decomposed circuit (Trotter)

------

## 8. Trigger Phrases

- Solve 2D heat equation using quantum method
- Quantum simulation for planar heat conduction
- 2D thermal analysis with Schrödingerization
- Quantum PDE solver for heat conduction

------

## 9. Use Cases

- Chip / PCB thermal simulation
- 2D material heat conduction
- Thin-plate temperature analysis
- Quantum PDE benchmarking

------

### Summary

This skill provides a **complete quantum solution for the 2D heat equation**:

- Uses Kronecker product for 2D Laplacian assembly
- Supports anisotropic diffusion in x/y directions
- Works with classical, Trotter, and block solvers
- Automatically reshapes to 2D temperature field
- Generates professional 3D visualizations
- Fully aligned with your implementation and mathematical framework



# Skill: Quantum Simulation for 1D Variable-Coefficient Heat Equation

## Basic Information

- **Skill ID**: `quantum_heat_var_coeff_1d`
- **Skill Name**: Quantum Solver for 1D Variable-Coefficient Forward Heat Equation
- **Domain**: Quantum Computing, Heat Conduction, PDEs, Nonhomogeneous Materials
- **Core Purpose**: Solve the **1D heat equation with spatially varying thermal conductivity** using Schrödingerization quantum methods. Supports periodic boundaries, variable diffusivity a(x), and classical/block solvers.

------

## 1. Mathematical Background

### 1.1 Variable-Coefficient Heat Equation

∂t∂u=a(x)∂x2∂2u+f(x)

where the diffusion coefficient is:

a(x)=1+cos(L2πx)

- a(x): spatially varying thermal conductivity
- u(x,t): temperature field
- f(x): heat source

### 1.2 Schrödingerized Hamiltonian (Periodic Form)

H≈Dη⊗a(X)DΔ

- Dη: auxiliary operator discretization
- a(X): diagonal coefficient matrix
- DΔ: discrete Laplacian

Valid for:

- Hermitian symmetric difference
- Periodic boundary conditions
- Source-free case f=0

------

## 2. Supported Features

- 1D heat conduction with **spatially varying diffusivity**

- Diffusion coefficient: a(x)=1+cos(2πx/L)

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

```
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

```
u0 = f0(x)  # initial temperature distribution
```

### Step 3: Build Variable-Coefficient Operator

Construct Laplacian and multiply by spatially varying coefficient a(x).

```
# Build discrete second-derivative operator
p2_matrix = generate_compact_p_2_normal(nx, 0, L)

# Build diagonal matrix for a(x) = 1 + cos(2πx/L)
ax = 1 + np.cos(2 * np.pi * x / L)
ax_matrix = np.diag(ax)

# Assemble full operator: A = a(x)·Δ
A = ax_matrix @ p2_matrix
```

### Step 4: Set Source Term

```
b = source(x)
```

### Step 5: Quantum Schrödingerization Solver

```
u = schro(A, u0, T=T, na=na, R=R, order=order, b=b)
qc = circuit_classical(nx, na)
```

### Step 6: Visualization

Plot temperature distribution u(x,T).

```
ax.plot(x, u, "b-", linewidth=2)
ax.fill_between(x, u, alpha=0.3)
```

### Step 7: Output Results & Quantum Circuit

```
circuit_files = _generate_circuit_plots(name, qc)
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Discrete Laplacian Construction

```
# Generate second-order spatial derivative matrix
p2_matrix = generate_compact_p_2_normal(nx, 0, L)
```

This builds the symmetric discrete Laplacian DΔ.

### 4.2 Variable-Coefficient Diagonal Matrix

```
# Build a(x) as a diagonal matrix
ax = 1 + np.cos(2 * np.pi * x / L)
ax_matrix = np.diag(ax)
```

Converts the spatially varying function a(x) into a diagonal matrix for fast multiplication.

### 4.3 Full PDE Operator Assembly

```
# Variable-coefficient heat operator: A = a(x)·Δ
A = ax_matrix @ p2_matrix
```

Combines the variable conductivity and Laplacian into one matrix.

### 4.4 Quantum Solver Call

```
# Core quantum evolution for variable-coefficient PDE
u = schro(A, u0, T=T, na=na, b=b)
```

Solves the linear PDE using Schrödingerization.

------

## 5. Boundary & Initial Conditions

### Boundary Condition

**Periodic BC**:

u(0,t)=u(L,t)

### Initial Conditions

- Sine:

u(x,0)=sin(L2πx)

- Discontinuous step:

u(x,0)={0,1,x∈[0,L/2)x∈[L/2,L]

------

## 6. Finite-Difference Scheme

Central difference for variable-coefficient diffusion:

a(xi)Δui≈a(xi)Δx2Δt(ui+1n−2uin+ui−1n)

------

## 7. Outputs

- Temperature field u(x,T)
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

- Uses a(x)=1+cos(2πx/L) for spatially varying diffusion
- Builds the PDE operator as A=diag(a(x))⋅Δ
- Supports classical and block quantum solvers
- Automatically generates visualizations and quantum circuits
- Fully aligned with your implementation and mathematical framework



# Skill: Quantum Simulation for 1D Helmholtz Equation

## Basic Information

- **Skill ID**: `quantum_helmholtz_1d`
- **Skill Name**: Quantum Solver for 1D Helmholtz Equation
- **Domain**: Quantum Computing, Wave Propagation, Acoustics, Electromagnetism, PDEs
- **Core Purpose**: Solve the **1D time-harmonic Helmholtz equation** using Schrödingerization via damped dynamical system (DDS) reformulation and quantum matrix exponential solver. Supports radiation boundary conditions and central difference discretization.

------

## 1. Mathematical Background

### 1.1 1D Helmholtz Equation

−Δu−k2u=f,x∈(0,L)

- k: wave number, proportional to wave frequency
- u(x): complex-valued steady-state wave field
- f(x): source term
- When k=0, reduces to **Laplace equation**

### 1.2 Schrödingerization via Damped Dynamical System

1. Discretize to linear system:

Ax=b

1. Rewrite as second-order damped ODE:

dtdV=MV+F,V=[v;w]

1. Transform to homogeneous form:

dtdVf=MfVf

1. Split into Hermitian parts:

Mf=H1+iH2

1. Apply warped phase transformation and Fourier discretization, yielding Hamiltonian:

H=Dp⊗H1−INp⊗H2

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

### Step 1: Parse Wave & Domain Parameters

Extract wave number k, domain length, qubits, and boundary type.

```
k = eq.get_parameter('k')
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type

Nx = 2**nx
dx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
```

### Step 2: Build Discrete Helmholtz Matrix

Construct central difference scheme with radiation boundary.

```
A0 = np.zeros((Nx, Nx), dtype=complex)
for i in range(1, Nx-1):
    A0[i, i-1] = -1
    A0[i, i]    = 2 - 2*(1 - np.cos(k*dx))
    A0[i, i+1] = -1

# Sommerfeld boundary at x=L
A0[0, 0] = 1
A0[-1, -1] = -(1 - 1j*k*dx)
A0[-1, -2] = 1
```

### Step 3: Construct Source Term & Preconditioning

```
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

```
gam = 2 * m
b = np.concatenate((np.zeros(Nx), -b0))
A = np.block([
    [np.zeros((Nx,Nx)), -A0.conj().T],
    [A0,               -gam*np.eye(Nx)]
])
```

### Step 5: Quantum Schrödingerization Solver

```
u0 = np.zeros_like(b)
u = schro(A, u0, T=T, na=na, b=b, scale_b=1/T)
u = u[:Nx]  # extract physical solution
qc = circuit_classical(nx, na)
```

### Step 6: Visualize Real & Imaginary Parts

```
ax.plot(x, np.real(u), "b-", label="Re(u)")
ax.plot(x, np.imag(u), "r--", label="Im(u)")
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Discrete Helmholtz Operator (Central Difference)

```
A0 = np.zeros((Nx, Nx), dtype=complex)
for i in range(1, Nx-1):
    A0[i, i-1] = -1
    A0[i, i]    = 2 - 2*(1 - np.cos(k*dx))
    A0[i, i+1] = -1
```

Implements the scheme:

−Δx21(ui−1−2ui+ui+1)−k^2ui=f(xi)

with modified wave number:

k^=Δx22(1−cos(kΔx))

### 4.2 Sommerfeld Radiation Boundary

```
A0[0, 0] = 1
A0[-1, -1] = -(1 - 1j*k*dx)
A0[-1, -2] = 1
```

Enforces:

u(0)=0,u′(L)−iku(L)=0

### 4.3 Damped Dynamical System Block Matrix

```
A = np.block([
    [np.zeros((Nx,Nx)), -A0.conj().T],
    [A0,               -gam*np.eye(Nx)]
])
```

Converts linear PDE system into a first-order ODE suitable for Schrödingerization.

### 4.4 Quantum PDE Solver Call

```
u = schro(A, u0, T=T, na=na, b=b, scale_b=1/T)
u = u[:Nx]
```

Solves the dynamical system via quantum Schrödingerization and extracts the physical wave field.

------

## 5. Boundary & Initial Conditions

### Boundary Condition

**Sommerfeld Radiation Condition**:

u(0)=0,u′(L)−iku(L)=0

Ensures outgoing waves at boundary with no unphysical reflection.

### Initial Condition

Not applicable — Helmholtz equation is **steady-state elliptic PDE**.

------

## 6. Finite-Difference Scheme

Central difference discretization:

−Δx21(ui−1−2ui+ui+1)−k^2ui=f(xi)

where shifted wave number:

k^=Δx22(1−cos(kΔx))

------

## 7. Outputs

- Complex-valued steady-state wave field u(x)
- Real / imaginary part distribution plot
- Full quantum circuit diagram
- Preconditioned quantum operator diagrams

------

## 8. Trigger Phrases

- Solve 1D Helmholtz equation using quantum method
- Quantum simulation for time-harmonic acoustic waves
- Quantum solver for wave propagation with radiation BC
- Schrödingerization for Helmholtz PDE

------

## 9. Use Cases

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



# Quantum Simulation of Maxwell's Equations (1D)

## Basic Information

- **Skill ID**: `quantum_maxwell_simulation`
- **Skill Name**: Quantum Simulation of Maxwell's Equations Based on Schrödingerization Method
- **Domain**: Quantum Computing, Electromagnetics, PDE Numerical Simulation
- **Core Capability**: Implements quantum solution for **1D Maxwell's equations** based on the paper *Quantum Simulation of Maxwell's Equations* (Shi Jin, Nana Liu, Chuwen Ma). Supports impedance / periodic boundary conditions, uses Schrödingerization and twisted phase transformation to convert electromagnetic field evolution into quantum computable form, and outputs electric/magnetic field solutions, visualizations, and quantum circuit diagrams.

## Theoretical Foundation

### 1. Governing Equation

Maxwell's equations describe the spatiotemporal evolution of electromagnetic fields:

∂t∂D−∇×H∂t∂B+∇×E∇⋅B∇⋅D=−J,=0,=0,=ρ.

### 2. Core Methodology: Schrödingerization

- Convert Maxwell’s equations into a **high-dimensional Schrödinger system** using twisted phase transformation.
- Achieve time evolution via **Trotter splitting** and Fourier methods.
- Handle non-homogeneous boundaries using **auxiliary variables** and unitary transformations.
- Preserve divergence-free magnetic field and energy conservation during quantum simulation.

### 3. Key Features

- Supports **perfect conductor** and **impedance boundary conditions**.
- Transforms non-unitary dynamics into quantum-compatible unitary evolution.
- Numerically verified for physical consistency and gate-resource efficiency.

## Supported Features

|        Category         |                  Supported Configurations                  |
| :---------------------: | :--------------------------------------------------------: |
|  **Spatial Dimension**  |                             1D                             |
| **Boundary Conditions** |                    Periodic, Impedance                     |
|   **Quantum Solvers**   |              Classical matrix exponentiation               |
|   **Discretization**    |                  Staggered Yee-like grid                   |
|    **Fields Output**    |          Electric field E(x), Magnetic field B(x)          |
|       **Outputs**       | Numerical solutions, field plots, quantum circuit diagrams |

## Input Parameters (JSON Schema)

json











```
{
  "equation": {
    "type": "maxwell_1d",
    "parameters": {
      "v": 1.0
    },
    "boundary": {
      "type": "periodic / impedance"
    },
    "discrete": {
      "type": "central"
    },
    "initial": {
      "E0(x)": "np.cos(2 * np.pi * x / 5)",
      "B0(x)": "np.zeros_like(x)"
    },
    "source": "-(2*np.pi/5)*t*np.cos(2*np.pi*x/5)"
  },
  "solver": {
    "type": "classical",
    "L": 10.0,
    "T": 1.0,
    "nx": 3,
    "na": 2,
    "R": 10,
    "order": 1,
    "point": 0
  }
}
```

## Core Implementation Code Snippets

### 1. System Matrix Construction (Impedance Boundary)

```
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

### 2. Quantum Simulation Core

```
# Schrödingerization solver for Maxwell system
u = schro(A, u, bE, T=T, R=R, na=na, order=order, point=point, scale=1e-3)

# Extract electric and magnetic fields
E = u[:len(xE)]
B = u[len(xE):]
```

### 3. Staggered Grid Differential Operator

```
# Staggered difference for E and B fields
D_B = lil_matrix((Mx, Mx))
for i in range(1, Mx):
    D_B[i, i - 1] = -1.0
    D_B[i, i] = 1.0
Dx = -alpha * D_B
```

## Execution Pipeline (5 Standard Stages)

1. **Parameter Parsing**: Parse wave speed, boundaries, initial fields, quantum bits
2. **Matrix Construction**: Build staggered-grid differential matrices and system Hamiltonian
3. **Quantum Evolution**: Execute Schrödingerization solver with source term J
4. **Visualization**: Generate electric/magnetic field distribution plots
5. **Circuit Export**: Render full quantum circuit diagram

## Output Format

```
{
  "status": "ok",
  "message": "Maxwell equation solved",
  "grid": { "n_points": 8, "dx": 1.25 },
  "x": [0, 1.25, 2.5, ...],
  "E": [0.12, 0.35, ...],
  "B": [0.02, 0.08, ...],
  "circuit": [
    { "format": "svg", "filename": "xxx_circuit_full.svg" }
  ],
  "plot": {
    "format": "svg",
    "filename": "xxx_solution.svg"
  }
}
```

## Trigger Phrases

- Simulate 1D Maxwell equations using quantum Schrödingerization
- Compute electromagnetic field evolution with impedance boundaries
- Solve Maxwell equations with quantum circuit simulation
- Generate electric and magnetic field solutions via quantum method
- Quantum simulation of 1D electromagnetic wave propagation

## Use Cases

- Electromagnetic simulation: wave propagation in waveguide/conductors
- Quantum computing: PDE-based quantum algorithm demonstration
- Physics education: visualizing EM fields via quantum simulation
- Scientific research: divergence-free and energy-preserving EM simulation

------

### Summary

This is a **complete, production-ready AI Skill** fully aligned with your paper and implementation:

- Strictly follows the Schrödingerization framework for Maxwell’s equations
- Supports impedance/periodic boundaries and staggered-grid discretization
- Standardized input/output, visualization, and quantum circuit export
- Code snippets directly match your algorithm for seamless integration



# Skill: Quantum Algorithm for Multiscale Elliptic PDEs

## Basic Information

- **Skill ID**: `quantum_multiscale_elliptic`
- **Skill Name**: Quantum Simulation for Multiscale Elliptic Equations (Homogenization + Schrödingerization)
- **Domain**: Quantum Computing, Multiscale Modeling, Scientific Computing
- **Core Purpose**: Solve **1D highly oscillatory multiscale elliptic equations** using quantum Schrödingerization, supporting both **original fine-scale model** and **two-scale homogenization model** for composite materials and high-contrast media.

------

## 1. Theoretical Background

### 1.1 Multiscale Elliptic Equation

The second-order multiscale elliptic PDE with oscillatory coefficients:

−∇⋅(A(x)∇u(x))=f(x),u∣∂D=0

where A(x)=A(x,x/ε) has **high oscillation** and multiple scales.

### 1.2 Two‑Scale Homogenization Model

{−divyA(x,y)(∇xu0+∇yu1)=0−divx(∫YA(x,y)(∇xu0+∇yu1)dy)=f(x)

### 1.3 Quantum Solver: Schrödingerization

1. Convert linear system Su=f to dynamic system
2. Introduce auxiliary variables
3. Transform into Schrödinger form for quantum simulation

------

## 2. Supported Features

- 1D multiscale elliptic equation with high-oscillation coefficients
- Two-scale homogenization solver
- Finite element stiffness matrix construction
- Quantum Schrödingerization solver
- Visualization of u0 (macro) and u1 (micro) solutions
- Quantum circuit diagram output

------

## 3. Algorithm Execution Pipeline (Full Steps)

### Step 1: Parameter Parsing

Parse scale parameter ε, domain, quantum bits, and solver type.

```
L, T, source, nx, na, R, point, order, f0 = eq.get_common_coefficients()
epsilon = eq.get_parameter('ε', 2**-6)  # small scale parameter
Nx = 2**nx - 1
dx = L / 2**nx
```

### Step 2: Define Oscillatory Coefficient

```
def A_epsilon(x):
    return 1.0 / (2 + np.cos(2 * np.pi * x / epsilon / L))
```

### Step 3: Stiffness Matrix Construction

#### Original Fine-Scale FEM

```
def fem(k, A=A_epsilon, f=f, L=L):
    N = 1 + 2**k
    h = L / (N-1)
    Amean = [quad(A, x[j], x[j+1])[0]/h for j in range(N-1)]
    S = triangle_csc(Amean[:-1]+Amean[1:], -Amean[1:-1], -Amean[1:-1]) / h
    rhs = h * f(x[1:-1])
    return S, rhs
```

#### Homogenization Model (Two-Scale)

```
def multiFEM(k, A=A_y, L=L):
    A0 = triangle_csc(...)  # macro matrix
    B0 = sparse_csc(...)    # coupling matrix
    left = vstack([kron(M,A0), kron(B0,...)])
    S = hstack([left, left.T, K])
    return S, rhs
```

### Step 4: Build Hamiltonian for Quantum Solver

Select model: **original** or **homogenization**:

```
if hamiltonian_type == 'homogenization':
    S, rhs = multiFEM(nx, A_y, L)
else:
    S, rhs = fem(nx, A_epsilon, f, L)
```

### Step 5: Quantum Schrödingerization Solver

```
u = schro(-S, np.zeros_like(rhs), na=na, R=R, T=10, b=rhs, scale_b=1)
```

### Step 6: Extract Macro/Micro Solutions

For homogenization model:

```
u1 = u[:-Nx].reshape(Nx, Nx)  # micro solution
u0 = u[-Nx:]                  # macro solution
```

### Step 7: Visualization & Quantum Circuit Output

- Plot u0 and u1
- Export full quantum circuit diagram

------

## 4. Core Code Snippets with Explanation

### 4.1 Tridiagonal Stiffness Matrix

```
def triangle_csc(a, b, c, N=None):
    # Build symmetric tridiagonal matrix for FEM
    return sparse_csc(a, range(N), range(N), N) + \
           sparse_csc(b, range(N-1), range(1,N), N) + \
           sparse_csc(c, range(1,N), range(N-1), N)
```

### 4.2 Two-Scale Solution Extraction

```
u1, u0 = u[:-Nx].reshape(Nx, Nx), u[-Nx:]
u0 = get_u0(u0, x)          # macro solution
u1 = get_u1(u1, x, L)      # micro oscillatory solution
```

### 4.3 Quantum Solver Call

```
# Solve linear system via quantum Schrödingerization
u = schro(-S, np.zeros_like(rhs), na=na, b=rhs)
```

------

## 5. Outputs

- Solution: macro field u0, micro field u1
- 1D solution plot
- Quantum circuit diagram
- Grid and computation statistics

------

## 6. Trigger Phrases (AI Understanding)

- Solve multiscale elliptic equation using quantum method
- Simulate 1D highly oscillatory elliptic PDE
- Run two-scale homogenization with quantum Schrödingerization
- Compute macro and micro solutions for multiscale elliptic problem
- Quantum simulation for composite material PDE

------

## 7. Use Cases

- Multiscale simulation in composite materials
- High-contrast conduction / permeability problems
- Quantum acceleration for fine-grid multiscale PDEs
- Homogenization model validation

------

### Summary

This skill provides a **complete quantum solution for multiscale elliptic equations**:

- Supports both **original** and **two-scale homogenization** models
- Uses **Schrödingerization** to convert linear systems into quantum-solvable form
- Automatically constructs FEM stiffness matrices
- Outputs macro/micro solutions and quantum circuits
- Fully aligned with your algorithm and research paper

# Skill: Quantum Simulation for 1D Multiscale Transport Equation

## Basic Information

- **Skill ID**: `quantum_multiscale_transport_1d`
- **Skill Name**: Quantum Solver for 1D Multiscale Kinetic Transport Equation
- **Domain**: Quantum Computing, Radiative Transfer, Neutron Transport, Multiscale PDEs
- **Core Purpose**: Solve the **1D multiscale particle transport equation** using Schrödingerization, damped dynamical systems, and quantum time evolution. Supports small scaling parameter ϵ, inflow boundary conditions, and velocity-space discretization.

------

## 1. Mathematical Background

### 1.1 1D Multiscale Transport Equation

ϵ∂tf(v)+v⋅∇xf(v)=ϵ1(SσS∫Ωf(v′)dv′−σf(v))+ϵQ

- ϵ: ratio of mean free path to system scale
- σS: scattering cross-section
- σ: total cross-section
- f(x,v,t): particle distribution function
- As ϵ→0, reduces to a **diffusion equation**

### 1.2 Schrödingerization Hamiltonian

1. Discretize to linear ODE:

dtdu=−Au+F

1. Rewrite as homogeneous damped dynamical system:

dtdu~=−A~u~,u~=[u;v]

1. Split into Hermitian/anti-Hermitian parts:

H1=2A~+A~T,H2=2iA~−A~T

1. Warped phase transformation + Fourier discretization:

idtdw=(H1⊗D+H2⊗I)w

1. Final Hamiltonian:

H=H1⊗D+H2⊗I

------

## 2. Supported Features

- 1D multiscale particle/radition transport
- Small parameter ϵ for stiff kinetic regimes
- 4-point Gaussian velocity quadrature
- **Inflow characteristic boundary conditions**
- Zero initial particle density
- Central finite-difference discretization
- Quantum solver: classical matrix exponentiation
- Density & gradient visualization
- Quantum circuit output

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse Multiscale & Physical Parameters

Extract ϵ, cross-sections, domain, qubits, and velocity quadrature.

```
L, T, source, nx, na, R, order, f0 = eq.get_common_coefficients()

Nx = 2**nx
x = np.linspace(0, L, Nx+2)
dx = x[1]-x[0]

Q = eq.get_parameter('Q')
eps1 = eq.get_parameter('ε')
sigma_S = eq.get_parameter('σS')
sigma_A = eq.get_parameter('σA')
ssigma = sigma_S + eps1**2 * sigma_A

# 4-point Gaussian velocity quadrature
Nv = 4
gauss_points = [...]
gauss_weights = [...]
```

### Step 2: Initialize Boundary & State Vectors

```
F_L = 0.5 * np.ones((Nv,1))
F_R = np.zeros((Nv,1))
r0 = 0.25*(f0+f0)
j0 = 1/(2*eps1)*(f0-f0)
R0 = np.kron(np.ones((Nv,1)), r0)
J0 = np.kron(np.ones((Nv,1)), j0)
```

### Step 3: Build Spatial & Velocity Discrete Operators

Construct Laplacian, advection, and boundary terms.

```
L2 = np.diag(-2*np.ones(Nx)) + np.diag(np.ones(Nx-1),1) + np.diag(np.ones(Nx-1),-1)
M1 = np.diag(np.zeros(Nx)) + np.diag(np.ones(Nx-1),1) + np.diag(-np.ones(Nx-1),-1)

for i in range(Nv):
    vk = gauss_points[i]
    idx = slice(i*Nx, (i+1)*Nx)
    # Build Delta_x, L1, M2 with boundary conditions
    Delta_x[idx,idx] = ...
    L1[idx,idx] = ...
    M2[idx,idx] = ...
    # Boundary source vectors B_v, f_v, g_v
    B_v[idx] = ...
    f_v[idx] = ...
    g_v[idx] = ...
```

### Step 4: Assemble Evolution Operators

```
V = np.diag(gauss_points)
W = np.kron(np.ones((Nv,1)), gauss_weights.reshape(1,-1))
I_Nx = sp.eye(Nx)

# Build A1, B1, A2, B2, b_r, b_j
A1 = beta1*... + (1-beta1)*... + ...
B1 = -(lamb/2)*beta1*sp.kron(V,M1)
A2 = -beta3*... - beta2*... - ...
B2 = beta1*...
b_r = ...
b_j = ...
```

### Step 5: Preconditioning & Scaling

```
Lambda_ = np.diag(gauss_weights**0.5)
Lambda_inv = np.diag(gauss_weights**-0.5)

hR0 = sp.kron(Lambda_, I_Nx) @ R0
hJ0 = J0 / Nx**2
hA1 = sp.kron(Lambda_, I_Nx) @ A1 @ sp.kron(Lambda_inv, I_Nx)
...
```

### Step 6: Build Block Dynamical System Matrix

```
C_top = sp.hstack([B2, hA2, hb_j/Nx**0.5, 0])
C_middle = sp.hstack([hB1, hA1, 0, hb_r/Nx**0.5])
C_bottom1 = ...
C_bottom2 = ...
C = sp.vstack([C_top, C_middle, C_bottom1, C_bottom2])
```

### Step 7: Classical Time Integration

```
hy0 = np.vstack([hJ0, hR0, Nx**0.5, Nx**0.5])
for k in range(...):
    I_mat = np.eye(C.shape[0])
    mat = I_mat - dt_a*(C-I_mat)
    hy = np.linalg.solve(mat, hy)
```

### Step 8: Schrödingerization & Quantum Evolution

#### Step 8.1: Fourier & Warped Transformation

```
Np = 2**na
p = np.linspace(L, R, Np+1)
Fp = Sp @ QFTmtx(nqp)
iFp = iQFTmtx(nqp) @ Sp
hc0 = np.kron(np.exp(-np.abs(ep)), hy0).flatten()
hc0 = iFp @ hc0
```

#### Step 8.2: Hamiltonian Construction

```
H = C - sp.eye(C.shape[0])
H1 = (H + H.conj().T)/2
H2 = (H - H.conj().T)/(2j)
Dmup = sp.diags(mup.flatten())
HD = sp.kron(Dmup, H1) - sp.kron(sp.eye(Np), H2)
```

#### Step 8.3: Quantum Time Stepping

```
hc = hc0.copy()
for kk in range(...):
    I_HD = sp.eye(HD.shape[0]) + 1j*dt_a*HD
    hc = sp.linalg.spsolve(I_HD, hc)
```

### Step 9: Solution Recovery

```
C_matrix = hc.reshape(len(hy0), Np, order='F')
W = Fp @ C_matrix
# Optimal pp selection & density reconstruction
Urr1_full = ...  # particle density
Ujj1_full = ...  # density gradient
```

### Step 10: Visualization & Circuit Output

```
solution_plot_path = self._generate_solution_plot(...)
circuit_files = self._generate_circuit_plots(...)
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Velocity-Space Gaussian Quadrature

```
Nv = 4
gauss_points = np.array([0.1834..., 0.5255..., 0.7966..., 0.9602...])
gauss_weights = np.array([0.3626..., 0.3137..., 0.2223..., 0.1012...])
```

Discretizes continuous velocity integral into 4-point quadrature.

### 4.2 Spatial Difference Operators with BC

```
Delta_xk[0,0] = (-eps1_vk[i])/eps1_vk_dx[i]
Delta_xk[Nx-1,Nx-1] = eps1_vk[i]/eps1_vk_dx[i]
Delta_x[idx,idx] = 1/(2*dx) * Delta_xk
```

Enforces **inflow characteristic boundary conditions**.

### 4.3 Block Dynamical System Matrix

```
C = sp.vstack([C_top, C_middle, C_bottom1, C_bottom2])
```

Converts transport PDE into a first-order ODE for Schrödingerization.

### 4.4 Quantum Hamiltonian Assembly

```
HD = sp.kron(Dmup, H1) - sp.kron(sp.eye(Np), H2)
```

Final Hamiltonian for quantum simulation of multiscale transport.

### 4.5 Quantum Time Evolution

```
I_HD = sp.eye(HD.shape[0]) + 1j*dt_a*HD
hc = sp.linalg.spsolve(I_HD, hc)
```

Implicit quantum evolution step for the stiff transport system.

------

## 5. Boundary & Initial Conditions

### Boundary Condition

**Inflow Characteristic BC**:

(r+ϵj)xL=FL(v),(r−ϵj)xR=FR(v)

### Initial Condition

Zero initial distribution:

f(x,v,0)=0

------

## 6. Finite-Difference Scheme

Central difference semi-discrete form:

ϵdtdfm,in+vm2Δxfm,i+1n−fm,i−1n=ϵ1(SσS∑k=1Mwkfk,in−σfm,in)+ϵQm,in

------

## 7. Outputs

- Particle density field ρ(x)
- Density gradient ∂xρ(x)
- Dual 1D solution plots
- Full quantum circuit diagram
- Hamiltonian decomposition diagrams

------

## 8. Trigger Phrases

- Solve 1D multiscale transport equation using quantum method
- Quantum simulation for radiative/neutron transport
- Quantum solver for stiff multiscale kinetic PDEs
- Schrödingerization for multiscale particle transport

------

## 9. Use Cases

- Radiation shielding simulation
- Nuclear reactor neutron transport
- Optically thick medium heat transfer
- Multiscale particle diffusion regimes

------

### Summary

This skill provides a **complete, production-grade quantum solution for the 1D multiscale transport equation**:

- Full velocity-space quadrature + spatial discretization
- Inflow characteristic boundary conditions
- Stiff ϵ scaling for kinetic/diffusion regimes
- Damped dynamical system + Schrödingerization
- Hermitian Hamiltonian splitting + quantum evolution
- Automatic density/gradient reconstruction
- Professional visualization & quantum circuit output
- Fully aligned with your code and mathematical framework



# Skill: Quantum Simulation for 1D Ornstein-Uhlenbeck (OU) Process

## Basic Information

- **Skill ID**: `quantum_ou_process_1d`
- **Skill Name**: Quantum Solver for 1D Ornstein-Uhlenbeck Stochastic Process
- **Domain**: Quantum Computing, Stochastic Differential Equations (SDE), Quantitative Finance, Physics
- **Core Purpose**: Solve the **1D Ornstein-Uhlenbeck stochastic differential equation** using Schrödingerization, warped phase transformation, and quantum time evolution. Supports Euler-Maruyama discretization and exact stochastic solution comparison.

------

## 1. Mathematical Background

### 1.1 1D Ornstein-Uhlenbeck (OU) Process

dX(t)=aX(t)dt+rdWt

- a: drift (mean-reversion) coefficient
- r: volatility (diffusion) coefficient
- dWt: standard Brownian motion increment
- Models: mean-reverting stochastic dynamics, financial asset prices, physical Brownian particles

### 1.2 Schrödingerization Hamiltonian

1. Euler-Maruyama → continuous-time ODE:

dtdX~(t)=aX~(t)+ΔtrΔWk

1. Augmented state:

Y~=[X~(t);1/Δt],a~k=[a0rΔWk/Δt0]

1. Hermitian/anti-Hermitian decomposition:

H1,k=2a~k+a~kT,H2,k=2ia~k−a~kT

1. Warped phase + Fourier transform → Hamiltonian system:

dtdw=−iηH1,kw+iH2,kw

1. Final Hamiltonian:

H=ηH1,k−H2,k

------

## 2. Supported Features

- 1D Ornstein-Uhlenbeck stochastic process simulation
- **Euler‑Maruyama discretization** for SDE
- Brownian motion noise generation
- Augmented dynamical system for Schrödingerization
- Warped phase transformation + FFT
- Quantum time evolution (Crank–Nicolson)
- Exact analytical solution for validation
- Quantum circuit visualization
- Solution comparison plot

------

## 3. Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse SDE & Quantum Parameters

Extract drift a, volatility r, time step, initial state, qubits.

```
a = eq.get_parameter('a')
r = eq.get_parameter('r')
dt = eq.discrete.get_parameter('dt', 0.001)
x0 = eq.initial.get_parameter('x0', 1.0)

NT = int(T / dt)
N = 2**(na - 1)
M = 2 * N
dp = (R - L) / M
```

### Step 2: Initialize Fourier & Phase Variables

```
p = np.arange(L, R, dp)
exp_p = np.exp(-np.abs(p))

# Initial augmented state
v0 = np.ones(2 * M)
v0[0::2] = exp_p * x0
v0[1::2] = exp_p / np.sqrt(dt)
```

### Step 3: Generate Brownian Motion

```
xi_BM = np.random.randn(NT)
xi = r * xi_BM / np.sqrt(dt)
xi_scale = r * xi_BM
```

### Step 4: FFT Preprocessing

```
c0 = A * v0
c0[0::2] = np.fft.fft(c0[0::2])
c0[1::2] = np.fft.fft(c0[1::2])
c0 /= M
c[:, 0] = c0
```

### Step 5: Time Evolution with Quantum Hamiltonian

```
for i in range(NT):
    # Build Hermitian parts
    H_1k = np.array([[a, xi_scale[i]/2], [xi_scale[i]/2, 0]])
    H_2k = np.array([[0, -1j*xi_scale[i]/2], [1j*xi_scale[i]/2, 0]])

    # Full quantum Hamiltonian
    H_k = -1j * (np.kron(Du.toarray(), H_1k) - np.kron(Id.toarray(), H_2k))

    # Crank–Nicolson time step
    LHS = I - (h/2)*H_k
    RHS = (I + (h/2)*H_k) @ c[:,i]
    c[:,i+1] = np.linalg.solve(LHS, RHS)
```

### Step 6: Inverse FFT & Solution Recovery

```
v_odd = M * np.fft.ifft(c[0::2], axis=0)
v_even = M * np.fft.ifft(c[1::2], axis=0)
v = A[:,None] * np.vstack([v_odd, v_even])

# Integrate to recover X(t)
y_int = np.real(np.sum(v_wave[n_start:], axis=0) * dp / zz)
```

### Step 7: Exact Solution for Validation



```
# Exact OU solution
y_expli[0] = x0
for k in range(NT):
    y_expli[k+1] = np.exp(a*dt)*y_expli[k] + (np.exp(a*dt)-1)/a * xi[k]
```

### Step 8: Visualization & Quantum Circuit

```
solution_plot_path = self._generate_solution_plot(...)
circuit_files = self._generate_circuit_plots(...)
```

------

## 4. Core Code Snippets with Explanation

### 4.1 Hermitian Hamiltonian Decomposition



```
H_1k = np.array([[a, xi_scale[i]/2], [xi_scale[i]/2, 0]])
H_2k = np.array([[0, -1j*xi_scale[i]/2], [1j*xi_scale[i]/2, 0]])
```

Splits the SDE operator into Hermitian/anti‑Hermitian parts for quantum simulation.

### 4.2 Quantum Hamiltonian Construction

```
H_k = -1j * (np.kron(Du.toarray(), H_1k) - np.kron(Id.toarray(), H_2k))
```

Final Hamiltonian for Schrödingerization of the OU process.

### 4.3 Crank–Nicolson Quantum Time Step

```
LHS = I - (dt/2)*H_k
RHS = (I + (dt/2)*H_k) @ c[:,i]
c[:,i+1] = np.linalg.solve(LHS, RHS)
```

Stable implicit time evolution for the quantum system.

### 4.4 Solution Recovery by Integration

```
y_int = np.real(np.sum(v_wave[n_start:n_end], axis=0) * dp / zz)
```

Recovers the stochastic process X(t) from the quantum wavefunction.

------

## 5. Boundary & Initial Conditions

### Boundary Condition

Not applicable (time-dependent SDE)

### Initial Condition

X(0)=x0

------

## 6. Discretization Scheme

**Euler‑Maruyama scheme**:

Xn+1=Xn+aXnΔt+rΔWn

where ΔWn∼N(0,Δt).

------

## 7. Outputs

- Stochastic trajectory X(t) (quantum solution)
- Exact analytical trajectory for validation
- Comparison plot (quantum vs exact)
- Full quantum circuit diagram
- Hamiltonian decomposition diagrams

------

## 8. Trigger Phrases

- Solve 1D Ornstein-Uhlenbeck process using quantum method
- Quantum simulation for stochastic differential equations
- Quantum SDE solver with Schrödingerization
- Mean-reverting stochastic process quantum simulation

------

## 9. Use Cases

- Financial stochastic modeling
- Physical Brownian particle dynamics
- Mean-reverting asset price simulation
- Quantum acceleration for SDEs

------

### Summary

This skill provides a **complete, correct quantum solution for the 1D Ornstein-Uhlenbeck process**:

- Full SDE → quantum Hamiltonian pipeline
- Warped phase transformation + FFT
- Stable Crank–Nicolson quantum evolution
- Exact solution validation
- Automatic trajectory recovery & plotting
- Professional quantum circuit output
- Fully aligned with your code and mathematical framework



# Quantum Dynamics Simulation with Artificial Boundary Conditions (2D Schrödinger)

## Basic Information

- **Skill ID**: `quantum_schrodinger_abc_simulation`
- **Skill Name**: Quantum Dynamics Simulation with Artificial Boundary Conditions (ABC)
- **Domain**: Quantum Computing, Quantum Dynamics, Scientific Computing, PDEs
- **Core Capability**: Solves the **2D Schrödinger equation with non-unitary artificial boundary conditions** using the Schrödingerization method. Restores unitary evolution for quantum simulation, supports absorbing boundary potentials, and outputs wavefunction solutions, 3D visualizations, and quantum circuit diagrams.

## Theoretical Foundation

### 1. Background & Challenge

Artificial boundary conditions (ABC) are used to simulate **open infinite systems** within a finite computational domain. However, ABC introduces **non-unitary dynamics**, breaking standard quantum simulation frameworks. The Schrödingerization method converts non-unitary evolution into a valid Schrödinger form, enabling quantum computation.

### 2. Governing Equation

The time-dependent Schrödinger equation:

i∂t∂ψ=Hψ,H=−21Δ+V(x)

**Artificial Boundary Condition**:

ψ(L,t)=αψ(L−Δx,t)+β∂x∂ψx=L

### 3. Complexity Bound

Hamiltonian simulation complexity:

O(s∥H∥max,1loglog(s∥H∥max,1/ε)log(s∥H∥max,1/ε))

## Core Algorithm & Code Explanation

### 1. Artificial Potential Construction (Absorbing Boundaries)

Creates a smooth absorbing potential near domain boundaries to eliminate reflections:

python

```
# Absorbing potential for artificial boundaries
sigma[x_mask] += amplitude * (np.abs(X[x_mask]) - rcut)**2
sigma[y_mask] += amplitude * (np.abs(Y[y_mask]) - rcut)**2
```

### 2. Hamiltonian Matrix Assembly

Builds Laplacian and potential terms for 2D Schrödinger:

```
# 2D Laplacian operator
D = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(Nx, Nx))
Z_mat = kron(D, I) + kron(I, D)

# Full Hamiltonian with kinetic, potential, and ABC terms
H0 = Z_mat / (2 * dx**2) - diags(V, 0)
H1 = diags(sigma, 0)  # Artificial boundary potential
A = H0 * 1j - H1
```

### 3. Quantum Schrödingerization Solver

Executes non-unitary quantum simulation:

python

```
# Core quantum simulation for Schrödinger equation with ABC
u = schro(A, u0, T=T, na=na, R=R, order=order, point=point)
u = u.reshape(Nx, Nx)
```

## Execution Pipeline

1. **Parameter Parsing**: Domain size, ABC parameters `rcut` / `amplitude`, qubits
2. **Grid & Potential Setup**: Generate 2D grid, internal potential, and absorbing boundary
3. **Hamiltonian Construction**: Laplacian + potential + artificial boundary terms
4. **Quantum Simulation**: Run Schrödingerization solver
5. **Visualization**: 3D wavefunction plot + quantum circuit diagram

## Output Format

```
{
  "status": "ok",
  "message": "Schrödinger ABC equation solved",
  "grid": { "n_points": 16, "dx": 0.5 },
  "x": [...],
  "y": [...],
  "u": [...complex wavefunction...],
  "circuit": [{ "filename": "xxx_circuit_full.svg" }],
  "plot": { "filename": "xxx_solution.svg" }
}
```

## Supported Features

- 2D Schrödinger equation with **artificial/absorbing boundary conditions**
- Non-unitary dynamics converted via **Schrödingerization**
- Customizable absorbing potential strength and boundary range
- Classical matrix exponentiation quantum solver
- 3D wavefunction visualization
- Automatic quantum circuit diagram generation

## Typical Trigger Phrases

- Simulate 2D quantum dynamics with artificial boundary conditions
- Solve Schrödinger equation with absorbing boundaries using quantum method
- Run ABC Schrödinger simulation and visualize wavefunction
- Quantum simulation for open quantum systems with finite domain

------

### Summary

This skill enables **quantum simulation of open quantum systems** by handling artificial boundary conditions:

- Uses Schrödingerization to restore unitary evolution for non-unitary ABC
- Implements smooth absorbing potentials to mimic infinite domains
- Provides full 2D simulation, visualization, and quantum circuit output
- Matches the paper’s complexity analysis and numerical framework
