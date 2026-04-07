# Skill: 1D Backward Heat Equation Solver via Schrödingerization

## Skill Identity

- **Name:** Quantum_BackwardHeat_Schrodingerization_Solver
- **Version:** 1.0
- **Domain:** Ill-posed PDEs · Parabolic Equations · Quantum Simulation
- **Equation Type:** Backward Heat Equation (Ill-posed)
- **Task Type:**
  - Ill-posed Problem Regularization
  - PDE Discretization
  - Schrödingerization
  - Quantum/Classic Hybrid Simulation

------

## Skill Objective

Enable the agent to **stably solve ill-posed PDEs**, especially the **1D backward heat equation**, by:

- Transforming exponential growth into controlled evolution
- Applying Schrödingerization
- Supporting both classical and quantum-inspired solvers

------

## Mathematical Model

### Backward Heat Equation

$$
\partial_t u = H u, \quad u(0) = u_0
$$

- Typically $H = -\Delta$（or discrete Laplacian）
- **Ill-posedness:**
  - Exponential increase in the frequency mode index
  - Extremely sensitive to initial values

------

## Core Capabilities

### 1. Ill-posedness Handling

Agent must:

- Detect exponential growth instability
- Apply **warped phase transformation**

Introduce:
$$
w(t,p) = e^{-p} u(t)
$$
Transform equation:
$$
\frac{d}{dt} w = -H \partial_p w
$$

------

### 2. Spatial Discretization

Construct grid:
$$
x_i = i\Delta x, \quad \Delta x = \frac{L}{N}
$$
Construct Laplacian operator (via finite difference):

```
A = -a * CDiff(
    N=Nx,
    dx=dx,
    order=2,
    scheme=scheme,
    boundary=bd
).get_matrix()
```

------

### 3. Schrödingerization Mapping

Transform:
$$
\frac{du}{dt} = A u
\quad \Rightarrow \quad
\frac{d\psi}{dt} = -iH\psi
$$
Steps:

1. Auxiliary lifting (introduce $p$)
2. Fourier transform in $p$
3. Construct Hamiltonian

$$
H = D \otimes H_1 + I \otimes H_2
$$

------

### 4. Classical Solver (Matrix Exponentiation)

Use Schrödingerization-based classical solver:

```
u = schro_classical(
    A,
    u0,
    T=T,
    na=na,
    R=R,
    order=order,
    point=point
)
```

Characteristics:

- Stable compared to direct backward evolution
- Avoids catastrophic amplification

------

### 5. Quantum-Inspired Solver (Trotter Method)

#### Hamiltonian splitting:

$$
H = H_1 + H_2
$$

#### Lie-Trotter decomposition:

$$
e^{-iHt}
\approx
\left(e^{-iH_1 \Delta t} e^{-iH_2 \Delta t}\right)^{N_t}
$$

```
func1, func2 = (a * TDiff(nx, dx, 2, scheme=scheme, boundary=bd)).data()

H1 = func1(dt / R)
H2 = func2(dt)

u, qc = schro_trotter(
    u0=u0,
    H1=H1,
    H2=H2,
    Nt=Nt,
    na=na,
    R=R,
    order=order,
    point=point
)
```

------

### 6. Block Encoding (Extension Capability)

- Reserved for quantum linear algebra methods
- Currently fallback to classical solver

```
u = backHeatEquationAlgorithm._solve_block(eq)
```

------

### 7. Boundary Condition Handling

Agent supports:

- **Dirichlet:**
  $$
  u(0,t) = u(L,t) = 0
  $$

- **Periodic:**
  $$
  u(0,t) = u(L,t)
  $$

- **Neumann:**
  $$
  \partial_x u = 0
  $$

Implementation:

```
if bd == "periodic":
    x = np.arange(0, L, dx)
elif bd == "neumann":
    x = np.arange(0, L+dx, dx)
```

------

### 8. Solution Recovery

After evolution in transformed space:
$$
u(t) = e^{p} w(t,p)
$$

```
w0 = np.exp(-p) * u0
u_t = np.exp(p) * w_t
```

------

### 9. Visualization Capability

#### Solution Plot

```
fig, ax = plt.subplots()
ax.plot(x, u)
ax.set_title("Backward Heat Solution")
fig.savefig("solution.svg")
```

#### Quantum Circuit Visualization

```
qc.draw(filename="circuit_full.svg")
H1.decompose().draw(filename="circuit_H1.svg")
H2.decompose().draw(filename="circuit_H2.svg")
```

------

## Execution Workflow (Agent Pipeline)

1. Parse input parameters
2. Identify ill-posed structure
3. Apply warped transformation
4. Construct spatial operator $A$
5. Perform Schrödingerization
6. Choose solver:
   - Classical (default)
   - Trotter (quantum-inspired)
7. Run time evolution
8. Recover physical solution
9. Generate visualization
10. Output results

------

## Output Specification

Agent must return:

- Solution $u(x,t)$
- Grid data $(x, \Delta x)$
- Stability-related parameters
- (Optional) quantum circuit $qc$
- Visualization files

------

## Stability & Numerical Considerations

- Backward heat equation is **severely ill-posed**
- Schrödingerization acts as:
  - implicit regularization
  - structure-preserving transformation

Accuracy depends on:

- grid size $N$
- auxiliary dimension $n_a$
- Trotter step $N_t$
- smoothness of lifting function

------

## Skill Trigger Conditions

Activate when:

- PDE contains:
  - diffusion operator with reversed time
- Keywords:
  - “backward heat equation”
  - “ill-posed PDE”
  - “unstable evolution”
  - “regularization via Schrödingerization”

------

## Summary

This skill provides a **robust computational framework for ill-posed PDEs**:
$$
\text{Backward Heat}
\;\rightarrow\;
\text{Warped Transformation}
\;\rightarrow\;
\text{Schrödingerization}
\;\rightarrow\;
\text{Unitary Evolution}
$$
It enables:

- Stable numerical simulation
- Quantum-compatible formulation
- Integration of classical and quantum methods