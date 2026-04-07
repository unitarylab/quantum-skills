---
name: black_scholes_schrodingerization
description: A quantum-compatible solver for the Black–Scholes equation in log-price formulation using Schrödingerization. Supports European option pricing with far-field boundary treatment, enabling both classical and quantum Hamiltonian-based simulation approaches.
---
# Skill: Quantum Solver for Black–Scholes Equation via Schrödingerization

## Skill Identity

- **Name:** Quantum_BlackScholes_Schrodingerization_Solver
- **Version:** 1.0
- **Domain:** Computational Finance · Quantum Computing · PDEs
- **Equation Type:** Parabolic PDE (Black–Scholes)
- **Task Type:**
  - Financial PDE Modeling
  - PDE Discretization
  - Schrödingerization
  - Quantum/Classic Hybrid Simulation

------

## Skill Objective

Enable the agent to **solve the 1D Black–Scholes equation (European option pricing)** using a Schrödingerization-based framework, supporting:

- Log-price transformation
- Far-field boundary treatment
- Classical and quantum (Trotter) solvers
- Mapping to real stock price space

------

## Mathematical Model

### 1. Log-Price Black–Scholes Equation

$$
\frac{\partial u}{\partial t}
=
\left(r - \frac{\sigma^2}{2}\right)\frac{\partial u}{\partial x}
+
\frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2}
-
r u
$$

- $x = \log S$: log stock price
- $r$: risk-free rate
- $\sigma$: volatility
- $K$: strike price

------

### 2. Initial Condition (European Put)

$$
u(x,0) = \max(K - e^x, 0)
$$

------

### 3. Boundary Conditions

$$
u \sim K - e^x \quad (x \to -\infty), \qquad
u \sim 0 \quad (x \to +\infty)
$$

Handled numerically via **augmented state construction**.

------

## Core Capabilities

### 1. Parameter Parsing

Extract:

- Financial parameters: $r, \sigma, K$
- Spatial domain $[L_1, L_2]$ (log-price)
- Grid size $N_x = 2^{n_x}$
- Time horizon $T$
- Auxiliary dimension $n_a$
- Solver type

------

### 2. Log-Price Discretization

$$
x_i \in [L_1, L_2], \quad \Delta x = \frac{L_2 - L_1}{N_x + 1}
$$

```
L1 = np.log(1e-4)
L2 = np.log(10 * K)
Nx = 2**nx
dx = (L2 - L1) / (Nx + 1)
x = np.linspace(L1 + dx, L2 - dx, Nx)
```

------

### 3. Initial Condition & Augmented State

```
u0 = f0(x)
```

Augmented state for boundary handling:

```
scale_b = K * (0.5 * sigma**2 / dx**2 + 0.25 * (-sigma**2 + 2*r) / dx)
u0 = np.concatenate((u0, np.ones(Nx) * scale_b))
```

Purpose:

- Enforces far-field boundary behavior
- Stabilizes numerical evolution

------

### 4. Operator Construction (Finite Difference)

```
A1, b1 = first_order_derivative(N=Nx, dx=dx)
A2, b2 = second_order_derivative(N=Nx, dx=dx)
```

$$
A =
\left(r - \frac{\sigma^2}{2}\right) A_1
+
\frac{\sigma^2}{2} A_2
-
r I
$$

```
A = (r - 0.5*sigma**2) * A1 + 0.5*sigma**2 * A2
A = A - r * eye(2 * Nx)
```

------

### 5. Schrödingerization Mapping

$$
\frac{du}{dt} = Au
\quad \Rightarrow \quad
\frac{d\psi}{dt} = -iH\psi
$$

Procedure:

1. Introduce auxiliary variable $p$
2. Construct lifted state
3. Apply Fourier transform in $p$
4. Build Hamiltonian

$$
H = D \otimes H_1 + I \otimes H_2
$$

------

### 6. Classical Solver

```
u = schro(
    A,
    u0,
    T=T,
    na=na,
    R=R,
    order=order
)
```

Recover option price:

```
u = u[:Nx] + np.exp(x)
```

------

### 7. Quantum (Trotter) Solver

Hamiltonian splitting:
$$
H = H_1 + H_2
$$

```
H1 = GateSequence(nx+1)
func1 = (sigma**2/2 * TDiff(nx, dx, order=2)).data()[0]
H1.append(func1(dt/R), target=range(nx), control=nx)

H2 = GateSequence(nx+1)
func2 = ((r - sigma**2/2) * TDiff(nx, dx, order=1)).data()[1]
H2.append(func2(dt), target=range(nx), control=nx)
```

Time evolution:

```
u, qc = schro(u0=u0, H1=H1, H2=H2, Nt=Nt, na=na)
```

------

### 8. Solution Reconstruction

$$
S = e^x
$$

```
S = np.exp(x[:end])
option_price = u[:end]
```

------

### 9. Visualization

```
ax.plot(np.exp(x[:end]), u[:end])
```

Outputs:

- Option price $V(S,T)$ vs stock price
- Financially interpretable curve

------

## Execution Workflow (Agent Pipeline)

1. Parse financial and numerical parameters
2. Construct log-price grid
3. Build initial payoff
4. Apply augmented state extension
5. Construct finite difference operators
6. Perform Schrödingerization
7. Select solver:
   - Classical
   - Trotter
8. Perform time evolution
9. Map solution to real price space
10. Generate visualization and circuits

------

## Numerical Considerations

- Black–Scholes is **well-posed**
- Schrödingerization provides:
  - unified Hamiltonian formulation
  - compatibility with quantum simulation

Key factors:

- grid resolution $N_x$
- volatility $\sigma$
- time discretization
- boundary truncation

------

## Skill Trigger Conditions

Activate when:

- PDE contains drift + diffusion + discount structure
- Keywords:
  - “Black–Scholes equation”
  - “option pricing PDE”
  - “quantum finance”
  - “Schrödingerization for finance”

------

## Use Cases

- European option pricing
- Quantum finance simulation
- PDE benchmarking
- Risk and volatility analysis

------

## Summary

This skill provides a **quantum-compatible solver for the Black–Scholes equation**:
$$
\text{Black–Scholes PDE}
\;\rightarrow\;
\text{Log Transformation}
\;\rightarrow\;
\text{Discretization}
\;\rightarrow\;
\text{Schrödingerization}
\;\rightarrow\;
\text{Quantum Evolution}
$$
It unifies:

- financial modeling
- numerical PDE methods
- quantum simulation frameworks