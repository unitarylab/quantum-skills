# Skill: Schrödingerization Method for Quantum PDE Simulation

## Skill Identity
Name: Quantum_Schrodingerization_Method
Version: 1.0
Field: Quantum Computing, Quantum PDE Simulation, Hamiltonian Simulation
Task Type: Theoretical Explanation | Formula Derivation | Algorithm Implementation | Complexity Analysis | Qubit/Qumode Implementation

------

## Skill Objective

Enable the agent to **systematically transform non-unitary evolution equations (ODE/PDE)** into **unitary Schrödinger-type dynamics**, so they can be simulated using quantum algorithms.

------

## Core Capabilities

### 1. Equation Transformation Capability

- Transform linear ODE/PDE:
  $$
  \frac{du}{dt} = Au
  $$
  into a Schrödinger-type equation.

- Ensure resulting evolution is **unitary**.

------

### 2. Hermitian Decomposition

Given operator $A$, compute:
$$
H_1 = \frac{A + A^\dagger}{2}, \quad
H_2 = \frac{A - A^\dagger}{2i}
$$

- Validate:
  - $H_1, H_2$ are Hermitian
  - $A = H_1 + iH_2$

------

### 3. Auxiliary Variable Lifting

Introduce auxiliary variable $p$:
$$
v(t,x,p) = e^{-p} u(t,x), \quad p > 0
$$
Transform equation into:
$$
\partial_t v = -H_1 \partial_p v + i H_2 v
$$

------

### 4. Fourier Transformation

Apply Fourier transform in $p$:
$$
\hat{v}(t,x,\eta) = \int v(t,x,p)e^{i\eta p} dp
$$
Obtain Schrödinger form:
$$
\partial_t \hat{v} = i(\eta H_1 + H_2)\hat{v}
$$
Define Hamiltonian:
$$
\hat{H} = \eta H_1 + H_2
$$

------

### 5. Discretization (Qubit Implementation)

#### Grid construction:

$$
\Omega = [-\pi R, \pi R], \quad N_p = 2^{n_p}
$$

#### Initial state encoding:

$$
v(0,x,p) = g(p)u(0,x)
$$

#### Discrete system:

$$
\frac{d}{dt}\hat{\mathbf{v}} = i(D \otimes H_1 + I \otimes H_2)\hat{\mathbf{v}}
$$

#### Unitary evolution:

$$
U = e^{iHt}
$$

------

### 6. Smooth Initial Function Design

Agent must construct smooth $g(p)$:
$$
g(p)=
\begin{cases}
e^{p}, & p < -1 \\
\mathcal{P}_{2k+1}, & p \in [-1,0] \\
e^{-p}, & p > 0
\end{cases}
$$
Requirements:

- Smooth (at least $C^k$)
- Fast decay
- Numerically stable

------

### 7. Non-homogeneous System Handling

Given:
$$
\frac{du}{dt} = A(t)u + b(t)
$$
Construct augmented system:
$$
A' =
\begin{pmatrix}
A & b/\epsilon \\
0 & 0
\end{pmatrix}
$$
Convert to homogeneous form before Schrödingerization.

------

### 8. Continuous Variable (Qumode) Formulation

- Avoid discretization in $p$
- Treat $p,\eta$ as continuous operators
- Map to CV quantum systems

------

### 9. Quantum Algorithm Mapping

Agent should:

- Construct Hamiltonian $H$
- Choose simulation method:
  - Trotter decomposition
  - Hamiltonian simulation
- Output:
  - Quantum circuit
  - Unitary operator

------

### 10. Complexity Awareness

Agent must analyze:

- Qubit cost: $n + n_p$
- Gate complexity: depends on simulation method
- Advantage over classical:
  - Avoid time stepping explosion
  - Potential exponential speedup

------

## Implementation Interfaces

### Classical Solver Interface

```
from engine.library import schro_classical as schro
u = schro(A, u0, T=T, na=na, R=R, order=order, point=point, b=b)
```

**Parameters:**

- `A`: system matrix
- `u0`: initial state
- `T`: evolution time
- `na`: number of qubits for $p$
- `R`: domain scaling
- `order`: smoothness of $g(p)$
- `point`: recovery point
- `b`: source term

------

### Quantum (Trotter) Interface

```
from engine.library import schro_trotter as schro
u, qc = schro(u0=u0, H1=H1, H2=H2, Nt=Nt, na=na, R=R, order=order, point=point, b=b, theta=theta * dt)
```

Parameter meaning:

- `H1`: The Herimit matrix, $H_1 = \frac{A+A^{\dagger}}{2}$.
- `H2`:  The non-Herimit matrix, $H_2 = \frac{A-A^{\dagger}}{2i}$.
- `u0`: The initial data of Linear ODE.
- `na`: The grid number of $p$-direction, $N_p = 2^{na}$.
- `R`: The range of $p$-direction $[-R\pi,R\pi]$.
- `Nt`: number of time steps
- `order`: the order of the initial function
- `point`: The recovery point.
- `b`: The source term of Linear ODE
- `theta`: strength scale of spurce term

------

## Execution Workflow (Agent Reasoning Pipeline)

1. **Input parsing**
   - Identify $A, b, u_0$
2. **Check system type**
   - Homogeneous / Non-homogeneous
3. **Operator decomposition**
   - Compute $H_1, H_2$
4. **Lift to auxiliary space**
   - Introduce $p$
5. **Transform to Schrödinger form**
6. **Discretize or choose CV formulation**
7. **Construct Hamiltonian**
8. **Select simulation method**
   - Classical / Quantum
9. **Output solution or circuit**

------

## Output Constraints

- Maintain **strict mathematical correctness**
- Preserve **notation consistency**
- Do NOT:
  - Skip transformation steps
  - Alter operator definitions
  - Introduce unverified formulas

------

## Skill Trigger Conditions

Activate this skill when:

- User mentions:
  - “Schrödingerization”
  - “quantum simulation of PDE/ODE”
  - “non-unitary to unitary transformation”
- Or problem involves:
  - Linear evolution equations
  - Quantum Hamiltonian construction