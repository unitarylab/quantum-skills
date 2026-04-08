----
name: quantum_multiscale_transport_1d
description: Quantum simulation for 1D multiscale particle transport equations using Schrödingerization and damped dynamical systems. Supports inflow boundary conditions, small ε scaling, velocity-space quadrature, finite-difference discretization, and outputs particle density, gradients, and quantum circuit diagrams.
----

# Skill: Quantum Simulation for 1D Multiscale Transport Equation

## Basic Information

- **Skill ID**: `quantum_multiscale_transport_1d`
- **Skill Name**: Quantum Solver for 1D Multiscale Kinetic Transport Equation
- **Domain**: Quantum Computing, Radiative Transfer, Neutron Transport, Multiscale PDEs
- **Core Capability**: Solve **1D multiscale particle transport equations** using Schrödingerization and damped dynamical systems.

------

## 1. Mathematical Background

### 1.1 1D Multiscale Transport Equation

$$
\epsilon \frac{\partial f(v)}{\partial t} + v \cdot \nabla_x f(v) = \epsilon \big( \sigma_S \int_\Omega f(v')\, dv' - \sigma f(v) \big) + \epsilon Q
$$

- $\epsilon$: ratio of mean free path to system scale
- $\sigma_S$: scattering cross-section
- $\sigma$: total cross-section
- $f(x,v,t)$: particle distribution function

As $\epsilon \to 0$, the equation reduces to a **diffusion equation**.

### 1.2 Schrödingerization Hamiltonian

1. Discretize PDE into linear ODE:

$$
\frac{d u}{d t} = - A u + F
$$

1. Rewrite as damped dynamical system:

$$
\frac{d \tilde{u}}{d t} = - \tilde{A} \tilde{u}, \quad \tilde{u} = \begin{bmatrix} u \\ v \end{bmatrix}
$$

1. Hermitian/anti-Hermitian splitting:

$$
H_1 = 2 \tilde{A} + \tilde{A}^\top, \quad H_2 = 2i \tilde{A} - \tilde{A}^\top
$$

1. Fourier discretization + warped phase transformation:

$$
i \frac{d w}{d t} = (H_1 \otimes D + H_2 \otimes I) w
$$

1. Final quantum Hamiltonian:

$$
H = H_1 \otimes D + H_2 \otimes I
$$

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

------

## 2. Supported Features

- 1D multiscale particle/radiation transport
- Small $\epsilon$ for stiff kinetic regimes
- 4-point Gaussian velocity quadrature
- **Inflow characteristic boundary conditions**
- Zero initial particle density
- Central finite-difference spatial discretization
- Quantum solver via classical matrix exponentiation
- Density & gradient visualization
- Quantum circuit output

------

## 3. Algorithm Pipeline

- ### Step 1: Parse Multiscale & Physical Parameters

  Extract ϵ, cross-sections, domain, qubits, and velocity quadrature.

  ```python
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

  ```python
  F_L = 0.5 * np.ones((Nv,1))
  F_R = np.zeros((Nv,1))
  r0 = 0.25*(f0+f0)
  j0 = 1/(2*eps1)*(f0-f0)
  R0 = np.kron(np.ones((Nv,1)), r0)
  J0 = np.kron(np.ones((Nv,1)), j0)
  ```

  ### Step 3: Build Spatial & Velocity Discrete Operators

  Construct Laplacian, advection, and boundary terms.

  ```python
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

  ```python
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

  ```python
  Lambda_ = np.diag(gauss_weights**0.5)
  Lambda_inv = np.diag(gauss_weights**-0.5)
  
  hR0 = sp.kron(Lambda_, I_Nx) @ R0
  hJ0 = J0 / Nx**2
  hA1 = sp.kron(Lambda_, I_Nx) @ A1 @ sp.kron(Lambda_inv, I_Nx)
  ...
  ```

  ### Step 6: Build Block Dynamical System Matrix

  ```python
  C_top = sp.hstack([B2, hA2, hb_j/Nx**0.5, 0])
  C_middle = sp.hstack([hB1, hA1, 0, hb_r/Nx**0.5])
  C_bottom1 = ...
  C_bottom2 = ...
  C = sp.vstack([C_top, C_middle, C_bottom1, C_bottom2])
  ```

  ### Step 7: Classical Time Integration

  ```python
  hy0 = np.vstack([hJ0, hR0, Nx**0.5, Nx**0.5])
  for k in range(...):
      I_mat = np.eye(C.shape[0])
      mat = I_mat - dt_a*(C-I_mat)
      hy = np.linalg.solve(mat, hy)
  ```

  ### Step 8: Schrödingerization & Quantum Evolution

  #### Step 8.1: Fourier & Warped Transformation

  ```python
  Np = 2**na
  p = np.linspace(L, R, Np+1)
  Fp = Sp @ QFTmtx(nqp)
  iFp = iQFTmtx(nqp) @ Sp
  hc0 = np.kron(np.exp(-np.abs(ep)), hy0).flatten()
  hc0 = iFp @ hc0
  ```

  #### Step 8.2: Hamiltonian Construction

  ```python
  H = C - sp.eye(C.shape[0])
  H1 = (H + H.conj().T)/2
  H2 = (H - H.conj().T)/(2j)
  Dmup = sp.diags(mup.flatten())
  HD = sp.kron(Dmup, H1) - sp.kron(sp.eye(Np), H2)
  ```

  #### Step 8.3: Quantum Time Stepping

  ```python
  hc = hc0.copy()
  for kk in range(...):
      I_HD = sp.eye(HD.shape[0]) + 1j*dt_a*HD
      hc = sp.linalg.spsolve(I_HD, hc)
  ```

  ### Step 9: Solution Recovery

  ```python
  C_matrix = hc.reshape(len(hy0), Np, order='F')
  W = Fp @ C_matrix
  # Optimal pp selection & density reconstruction
  Urr1_full = ...  # particle density
  Ujj1_full = ...  # density gradient
  ```

  ### Step 10: Visualization & Circuit Output

  ```python
  solution_plot_path = self._generate_solution_plot(...)
  circuit_files = self._generate_circuit_plots(...)
  ```


------

## 4. Core Code Snippets

### 4.1 Velocity-Space Gaussian Quadrature

```python
Nv = 4
gauss_points = np.array([0.1834..., 0.5255..., 0.7966..., 0.9602...])
gauss_weights = np.array([0.3626..., 0.3137..., 0.2223..., 0.1012...])
```

Discretizes continuous velocity integral into 4-point quadrature.

### 4.2 Spatial Difference Operators with BC

```python
Delta_xk[0,0] = (-eps1_vk[i])/eps1_vk_dx[i]
Delta_xk[Nx-1,Nx-1] = eps1_vk[i]/eps1_vk_dx[i]
Delta_x[idx,idx] = 1/(2*dx) * Delta_xk
```

Enforces **inflow characteristic boundary conditions**.

### 4.3 Block Dynamical System Matrix

```python
C = sp.vstack([C_top, C_middle, C_bottom1, C_bottom2])
```

Converts transport PDE into a first-order ODE for Schrödingerization.

### 4.4 Quantum Hamiltonian Assembly

```python
HD = sp.kron(Dmup, H1) - sp.kron(sp.eye(Np), H2)
```

Final Hamiltonian for quantum simulation of multiscale transport.

### 4.5 Quantum Time Evolution

```python
I_HD = sp.eye(HD.shape[0]) + 1j*dt_a*HD
hc = sp.linalg.spsolve(I_HD, hc)
```

Implicit quantum evolution step for the stiff transport system.

## 5. Boundary & Initial Conditions

$$
(r + \epsilon j)|_L = F_L(v), \quad (r - \epsilon j)|_R = F_R(v)
$$

------

## 6. Finite-Difference Scheme

$$
\epsilon \frac{d f_{m,i}^n}{dt} + v_m \frac{f_{m,i+1}^n - f_{m,i-1}^n}{2 \Delta x} = \epsilon (\sigma_S \sum_{k=1}^M w_k f_{k,i}^n - \sigma f_{m,i}^n) + \epsilon Q_{m,i}^n
$$

------

## 7. Outputs

- Particle density $\rho(x)$
- Density gradient $\partial_x \rho(x)$
- Dual 1D plots
- Quantum circuit diagram
- Hamiltonian decomposition diagrams

------

## 8. Trigger Phrases

- Solve 1D multiscale transport equation using quantum method
- Quantum simulation for radiative/neutron transport
- Schrödingerization for stiff kinetic PDEs
- Quantum solver for multiscale particle transport

------

## 9. Use Cases

- Radiation shielding simulation
- Nuclear reactor neutron transport
- Optically thick medium heat transfer
- Multiscale particle diffusion

------

### Summary

This skill provides a **complete quantum solution for 1D multiscale transport equations**:

- Full velocity-space quadrature + spatial discretization
- Inflow characteristic BC
- Stiff $\epsilon$ scaling for kinetic/diffusion regimes
- Damped dynamical system + Schrödingerization
- Hermitian Hamiltonian splitting + quantum evolution
- Automatic density/gradient reconstruction
- Professional visualization & quantum circuit output