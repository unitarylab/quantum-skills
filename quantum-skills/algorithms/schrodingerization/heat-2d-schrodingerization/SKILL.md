---
name: heat-2d-schrodingerization
description: A quantum-compatible solver for the 2D Heat Equation using Schrödingerization to transform the non-unitary diffusion equation into a unitary evolution problem. Supports anisotropic diffusion, Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution with automatic 2D circuit generation and 3D temperature field visualization.
---

# Skill: 2D Heat Equation (Schrödingerization-based Solver)

## Basic Information

- **Skill ID**: `heat_2d_schrodingerization`
- **Skill Name**: Quantum Solver for 2D Heat Equation
- **Domain**: Quantum Computing, Heat Conduction, PDEs, Thermal Simulation
- **Core Purpose**: Solve the **2D heat conduction PDE** on a planar domain using Schrödingerization. Supports anisotropic diffusion, Dirichlet and periodic boundary conditions, source terms, and both classical and Trotter-based quantum evolution. Automatically generates quantum circuits and 3D visualizations of the temperature field.

------

## Mathematical Model

### 2D Heat Equation

$$
\frac{\partial u}{\partial t} = a_1 \frac{\partial^2 u}{\partial x^2} + a_2 \frac{\partial^2 u}{\partial y^2} + f(x,y)
$$

- $a_1, a_2$: diffusion coefficients in x/y directions
- $u(x,y,t)$: temperature field
- $f(x,y)$: source term

### Schrödingerized Hamiltonian (Periodic Form)

$$
H = - \hat{\eta} \otimes \left( a_1 \hat{p}_x^2 + a_2 \hat{p}_y^2 \right)
$$

Valid if:

1. Discrete derivative operator is Hermitian
2. Periodic BC in both x/y directions
3. Source term $f(x,y) = 0$

Otherwise, full Schrödingerization procedure is required.

------

## Supported Features

- 2D anisotropic heat conduction
- Boundary conditions: **Dirichlet**, **Periodic**
- Initial conditions: 2D sine wave, custom
- Quantum solvers: Classical matrix exponentiation, Trotter splitting, Block encoding (fallback)
- Tensor-product finite-difference discretization
- 3D surface / contour visualization
- Automatic quantum circuit generation

------

## Minimal Executable Example

This repository now includes a runnable example at `./example_heat_2d_schrodingerization.py`.

It demonstrates a minimal periodic 2D heat problem:

$$
\frac{\partial u}{\partial t} = a_1 u_{xx} + a_2 u_{yy}, \quad
u(x,y,0) = \sin(2\pi x)\sin(2\pi y)
$$

with:

- $a_1 = 0.8$
- $a_2 = 0.3$
- $L = 1$
- $N_x = N_y = 2^3 = 8$
- periodic boundary conditions

Run it with the UnitaryLab environment:

```bash
python ./.agents/skills/quantum-skills/algorithms/schrodingerization/heat-2d-schrodingerization/example_heat_2d_schrodingerization.py
```

Core code:

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from engine import state_preparation
from engine.library import CDiff
from engine.library.schrodingerization import schro_classical

length = 1.0
final_time = 0.02
diffusion_x = 0.8
diffusion_y = 0.3
num_spatial_qubits = 3
grid_size = 2 ** num_spatial_qubits
dx = length / grid_size

x = np.arange(0.0, length, dx)
y = np.arange(0.0, length, dx)
x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
initial_field = np.sin(2 * np.pi * x_grid / length) * np.sin(2 * np.pi * y_grid / length)

laplacian_1d = CDiff(N=grid_size, dx=dx, order=2, boundary='periodic').get_matrix()
identity = sp.eye(grid_size, format='csc')
operator = diffusion_x * sp.kron(laplacian_1d, identity, format='csc')
operator += diffusion_y * sp.kron(identity, laplacian_1d, format='csc')

solution = np.asarray(
	schro_classical(
		operator,
		initial_field.reshape(-1),
		T=final_time,
		na=7,
		R=4,
		order=3,
	)
)

reference = expm_multiply(final_time * operator, initial_field.reshape(-1))
relative_l2_error = np.linalg.norm(solution - reference) / np.linalg.norm(reference)

normalized_solution = solution / np.linalg.norm(solution)
preparation_circuit = state_preparation(normalized_solution)
zero_state = np.zeros_like(normalized_solution, dtype=complex)
zero_state[0] = 1.0
prepared_state = np.asarray(preparation_circuit.execute(zero_state.copy()))
state_preparation_error = np.linalg.norm(prepared_state - normalized_solution)

print(relative_l2_error)
print(state_preparation_error)
```

Validation signals from the verified run in this workspace:

- `solution_rel_l2_error_vs_matrix_exp ≈ 3.0e-05`
- `state_preparation_error ≈ 1e-16`

This gives two concrete checks:

1. The Schrödingerization output matches the semidiscrete matrix-exponential reference very closely.
2. The normalized final field can be encoded and executed on the simulator with negligible statevector error.

------

## Full Algorithm Pipeline (Step-by-Step)

### Step 1: Parse 2D Domain & Parameters

Extract coefficients, domain, grid, qubits, boundary type:

```python
a1 = eq.get_parameter('a1')
a2 = eq.get_parameter('a2')
L, T, nx, na, R, order, f0 = eq.get_common_coefficients()
bd = eq.boundary.type

Nx = 2**nx
dx = L / (Nx + 1)
x = np.arange(dx, L, dx)
y = np.arange(dx, L, dx)
```

------

### Step 2: Initialize 2D Temperature Field

```python
u0 = f0(x[:, None], y[None, :])  # 2D initial condition
u0 = u0.flatten()                # flatten for solver
```

------

### Step 3: Assemble 2D Laplacian Matrix (Kronecker Product)

```python
A0 = CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()
A = a1 * sp.kron(A0, sp.eye(Nx)) + a2 * sp.kron(sp.eye(Nx), A0)
```

- `kron(A0, I)`: x-direction Laplacian
- `kron(I, A0)`: y-direction Laplacian

------

### Step 4: Build Source Term

```python
b0 = source(x)
b = a1 * np.kron(b0, np.ones(Nx)) + a2 * np.kron(np.ones(Nx), b0)
```

------

### Step 5: Schrödingerization Solver

```python
u = schro(A, u0, T=T, na=na, R=R, order=order, b=b)
u = u.reshape((Nx, Nx))  # reshape to 2D
qc = circuit_classical(nx, na, dim=2)
```

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.
------

### Step 6: Trotter Quantum Circuit (Optional)

```python
func1 = TDiff(nx, dx, 2, boundary=bd).data()[0]
D1 = lambda a: func1(a * dt/R)

H1 = GateSequence(2*nx)
H1.append(D1(a1), range(nx))      # x-direction
H1.append(D1(a2), range(nx, 2*nx))# y-direction
```

------

### Step 7: Run Trotter Evolution

```python
u, qc = schro(u0=u0, H1=H1, H2=None, Nt=Nt, na=na)
u = u.reshape((Nx, Nx))
```

------

### Step 8: Visualization

```python
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='viridis')
```

------

## Boundary & Initial Conditions

- **Dirichlet**: $u=0$ on domain boundary
- **Periodic**: $u(0,y)=u(L,y), u(x,0)=u(x,L)$
- **Initial Condition**: 2D sine wave:

$$
u(x,y,0) = \sin\left(\frac{\pi x}{L_x}\right) \sin\left(\frac{\pi y}{L_y}\right)
$$

------

## Finite-Difference Scheme

2D central difference for Laplacian:
$$
\Delta u_{i,j} = \Delta t \left[ a_1 \frac{u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}}{\Delta x^2} + a_2 \frac{u_{i,j+1} - 2 u_{i,j} + u_{i,j-1}}{\Delta y^2} \right]
$$

------

## Outputs

- 2D temperature field $u(x,y,T)$
- 3D surface plot
- Full quantum circuit diagram
- H1 decomposed Trotter circuit

------

## Trigger Phrases

- Solve 2D heat equation using quantum Schrödingerization
- Quantum simulation for planar heat conduction
- 2D thermal analysis
- Quantum PDE solver for heat conduction

------

## Use Cases

- Chip / PCB thermal simulation
- 2D material heat conduction
- Thin-plate temperature analysis
- Quantum PDE benchmarking

------

### Summary

This skill provides a **complete quantum solution for the 2D heat equation**:

- Uses Kronecker product for 2D Laplacian
- Supports anisotropic diffusion in x/y directions
- Works with classical, Trotter, and block solvers
- Automatically reshapes to 2D field
- Generates professional 3D visualizations
- Fully aligned with your implementation and mathematical framework