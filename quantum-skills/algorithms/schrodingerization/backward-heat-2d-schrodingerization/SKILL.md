---
name: backward-heat-2d-schrodingerization
description: A Schrödingerization-based solver for the 2D backward heat equation using Kronecker-structured discretization. Handles ill-posed diffusion dynamics and supports classical solvers and quantum Trotter evolution with circuit generation.
---

## One Step to Run 2D Backward Heat Example
```bash
python ./scripts/algorithm.py
```

# Skill: 2D Backward Heat Equation Solver via Schrödingerization

## Skill Objective

Enable the agent to **solve the 2D backward heat equation** in a stable and structured way by:

- Transforming ill-posed dynamics into controlled evolution
- Applying Schrödingerization
- Supporting classical and quantum-inspired solvers
- Providing visualization and circuit outputs

------

## Mathematical Model

### 2D Backward Heat Equation

$$
\frac{\partial u}{\partial t}
=
- a_1 \frac{\partial^2 u}{\partial x^2}
- a_2 \frac{\partial^2 u}{\partial y^2}
+ f(x,y)
$$

- $a_1, a_2$: diffusion coefficients
- $f(x,y)$: source term
- **Ill-posedness:** 高频模态指数增长

------

## Core Capabilities

### 1. Parameter Parsing

Extract from input:

- Domain size $L$
- Final time $T$
- Grid parameter $n_x$
- Auxiliary dimension $n_a$
- Time step $\Delta t$
- Boundary condition
- Initial condition $f_0(x,y)$
- Source function $f(x,y)$
- Solver type

------

### 2. Ill-posedness Treatment

Agent must:

- Detect backward diffusion structure
- Apply Schrödingerization transformation

Introduce auxiliary variable:
$$
w(t,p) = e^{-p} u(t)
$$

------

### 3. Spatial Discretization (2D)

#### Grid construction:

$$
N_x = 2^{n_x}, \quad \Delta x =
\begin{cases}
\frac{L}{N_x}, & \text{periodic} \\
\frac{L}{N_x+1}, & \text{Dirichlet/Neumann}
\end{cases}
$$

Construct mesh:
$$
(x_i, y_j)
$$
Initial condition:
$$
u_0 = f_0(x_i, y_j)
$$
Flatten:
$$
u_0 \rightarrow \mathbf{u}_0 \in \mathbb{R}^{N_x^2}
$$

------

### 4. Operator Construction

#### (A) Classical Matrix Form

Construct 1D Laplacian:

```python
A0 = CDiff(
    N=Nx,
    dx=dx,
    order=2,
    scheme=scheme,
    boundary=bd
).get_matrix()
```

Construct 2D operator:
$$
A =
a_1 (A_0 \otimes I)
+
a_2 (I \otimes A_0)
$$
Source term:
$$
b =
a_1 (f(x)\otimes \mathbf{1})
+
a_2 (\mathbf{1} \otimes f(y))
$$

------

#### (B) Quantum/Trotter Operator

Construct derivative operators:

```python
func1, func2 = TDiff(nx, dx, 2, scheme=scheme, boundary=bd).data()
```

Build Hamiltonian blocks:

- $H_1$: Diffusion term in the x/y direction
- $H_2 = 0$ (No additional term)

------

### 5. Schrödingerization Mapping

Transform:
$$
\frac{du}{dt} = A u + b
\quad \Rightarrow \quad
\frac{d\psi}{dt} = -iH\psi
$$
Steps:

1. Auxiliary lifting
2. Fourier transform in $p$
3. Construct Hamiltonian

$$
H = D \otimes H_1 + I \otimes H_2
$$

------

### 6. Solver Execution

#### (A) Classical Solver

```python
from engine.library import schro_classical
u = schro_classical(
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

#### (B) Trotter Solver

$$
e^{-iHt}
\approx
\left(e^{-iH_1 \Delta t} e^{-iH_2 \Delta t}\right)^{N_t}
$$

```python
from engine.library import schro_trotter
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

#### (C) Block Encoding (Extension)

```python
u = backHeatEquationAlgorithm._solve_block(eq)
```

------

### 7. Solution Reconstruction

Reshape:
$$
\mathbf{u} \rightarrow u(x,y)
$$

```python
u = u.reshape((Nx, Nx))
```

------

### 8. Visualization

#### 3D Surface Plot

```python
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u)
fig.savefig("solution.svg")
```

------

### 9. Quantum Circuit Output

```python
qc.draw(filename="circuit_full.svg")
H1.decompose().draw(filename="circuit_H1.svg")
H2.decompose().draw(filename="circuit_H2.svg")
```

------

## Execution Workflow (Agent Pipeline)

1. Parse input parameters
2. Detect ill-posed structure
3. Construct 2D grid
4. Build spatial operators
5. Apply Schrödingerization

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

6. Select solver:
   - Classical
   - Trotter
   - Block (fallback)
7. Perform time evolution
8. Recover 2D solution
9. Generate visualization & circuits
10. Output structured results

------

## Stability & Numerical Considerations

- 2D backward heat equation is **strongly ill-posed**
- Schrödingerization provides:
  - structure-preserving reformulation
  - improved numerical stability

Key factors:

- spatial resolution $N_x$
- auxiliary dimension $n_a$
- Trotter steps $N_t$
- boundary condition choice

------

## Skill Trigger Conditions

Activate when:

- PDE contains:
  - Laplacian in 2D
  - backward time evolution
- Keywords:
  - “2D backward heat equation”
  - “ill-posed PDE 2D”
  - “Schrödingerization diffusion”

------

## Summary

This skill generalizes the framework:
$$
\text{2D Backward Heat}
\;\rightarrow\;
\text{Discretization (Kronecker)}
\;\rightarrow\;
\text{Schrödingerization}
\;\rightarrow\;
\text{Unitary Evolution}
$$
It enables:

- Stable simulation of ill-posed 2D PDEs
- Integration of classical and quantum algorithms
- Full pipeline from PDE to quantum circuit