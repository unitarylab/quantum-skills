---
name: quantum-ou-process-1d
description: Quantum simulation for 1D Ornstein-Uhlenbeck stochastic process using Schrödingerization, warped phase transformation, and Crank–Nicolson quantum evolution. Supports Euler–Maruyama discretization, exact solution validation, trajectory reconstruction, and quantum circuit visualization.
---

## One Step to Run 1D OU Process Example
```bash
python ./scripts/algorithm.py
```

# Skill: Quantum Simulation for 1D Ornstein-Uhlenbeck (OU) Process

## 1. Mathematical Background

### 1.1 1D Ornstein-Uhlenbeck Process

$$
dX(t) = a X(t)\, dt + r\, dW_t
$$

- $a$: drift (mean-reversion) coefficient
- $r$: volatility (diffusion) coefficient
- $dW_t$: standard Brownian motion increment

Models mean-reverting dynamics in finance and physics.

### 1.2 Schrödingerization Hamiltonian

1. Euler–Maruyama discretization → augmented ODE:

$$
\frac{d \tilde{X}(t)}{dt} = a \tilde{X}(t) + r \Delta W_k
$$

1. Augmented state:

$$
\tilde{Y} = [\tilde{X}(t); 1/\Delta t], \quad \tilde{a}_k = [a, r \Delta W_k/\Delta t]
$$

1. Hermitian/anti-Hermitian decomposition:

$$
H_{1,k} = 2 \tilde{a}_k + \tilde{a}_k^\top, \quad H_{2,k} = 2i \tilde{a}_k - \tilde{a}_k^\top
$$

1. Warped phase + Fourier transform → Hamiltonian system:

$$
\frac{d w}{dt} = -i \eta H_{1,k} w + i H_{2,k} w
$$

1. Final Hamiltonian for quantum evolution:

$$
H = \eta H_{1,k} - H_{2,k}
$$

------

## 2. Supported Features

- 1D Ornstein-Uhlenbeck stochastic process
- Euler–Maruyama discretization for SDEs
- Brownian motion generation
- Augmented dynamical system for Schrödingerization
- Warped phase transformation + FFT
- Quantum Crank–Nicolson time evolution
- Exact analytical solution comparison
- Quantum circuit visualization
- Trajectory & comparison plots

------

## 3. Algorithm Pipeline

- ### Step 1: Parse SDE & Quantum Parameters

  Extract drift a, volatility r, time step, initial state, qubits.

  ```python
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

  ```python
  p = np.arange(L, R, dp)
  exp_p = np.exp(-np.abs(p))
  
  # Initial augmented state
  v0 = np.ones(2 * M)
  v0[0::2] = exp_p * x0
  v0[1::2] = exp_p / np.sqrt(dt)
  ```

  ### Step 3: Generate Brownian Motion

  ```python
  xi_BM = np.random.randn(NT)
  xi = r * xi_BM / np.sqrt(dt)
  xi_scale = r * xi_BM
  ```

  ### Step 4: FFT Preprocessing

  ```python
  c0 = A * v0
  c0[0::2] = np.fft.fft(c0[0::2])
  c0[1::2] = np.fft.fft(c0[1::2])
  c0 /= M
  c[:, 0] = c0
  ```

  ### Step 5: Time Evolution with Quantum Hamiltonian

  ```python
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

  ```python
  v_odd = M * np.fft.ifft(c[0::2], axis=0)
  v_even = M * np.fft.ifft(c[1::2], axis=0)
  v = A[:,None] * np.vstack([v_odd, v_even])
  
  # Integrate to recover X(t)
  y_int = np.real(np.sum(v_wave[n_start:], axis=0) * dp / zz)
  ```

  ### Step 7: Exact Solution for Validation

  

  ```python
  # Exact OU solution
  y_expli[0] = x0
  for k in range(NT):
      y_expli[k+1] = np.exp(a*dt)*y_expli[k] + (np.exp(a*dt)-1)/a * xi[k]
  ```

  ### Step 8: Visualization & Quantum Circuit

  ```python
  solution_plot_path = self._generate_solution_plot(...)
  circuit_files = self._generate_circuit_plots(...)
  ```


------

## 4. Core Code Snippets

### 4.1 Hermitian Hamiltonian Decomposition

```python
H_1k = np.array([[a, xi_scale[i]/2], [xi_scale[i]/2, 0]])
H_2k = np.array([[0, -1j*xi_scale[i]/2], [1j*xi_scale[i]/2, 0]])
```

Splits the SDE operator into Hermitian/anti‑Hermitian parts for quantum simulation.

### 4.2 Quantum Hamiltonian Construction

```python
H_k = -1j * (np.kron(Du.toarray(), H_1k) - np.kron(Id.toarray(), H_2k))
```

Final Hamiltonian for Schrödingerization of the OU process.

The Schrödingerization framework can be referred to in './Schr_skills.markdown'.

### 4.3 Crank–Nicolson Quantum Time Step

```python
LHS = I - (dt/2)*H_k
RHS = (I + (dt/2)*H_k) @ c[:,i]
c[:,i+1] = np.linalg.solve(LHS, RHS)
```

Stable implicit time evolution for the quantum system.

### 4.4 Solution Recovery by Integration

```python
y_int = np.real(np.sum(v_wave[n_start:n_end], axis=0) * dp / zz)
```

Recovers the stochastic process X(t) from the quantum wavefunction.5. Boundary & Initial Conditions

- **Boundary**: Not applicable (SDE)
- **Initial**: $X(0) = x_0$

## 5. Boundary & Initial Conditions

### Boundary Condition

Not applicable (time-dependent SDE)

### Initial Condition

$X(0)=x0$

## 6. Discretization Scheme

**Euler–Maruyama:**
$$
X_{n+1} = X_n + a X_n \Delta t + r \Delta W_n, \quad \Delta W_n \sim \mathcal{N}(0, \Delta t)
$$

------

## 7. Outputs

- Quantum stochastic trajectory $X(t)$
- Exact analytical trajectory
- Comparison plot (quantum vs exact)
- Quantum circuit diagram
- Hamiltonian decomposition diagrams

------

## 8. Trigger Phrases

- Solve 1D Ornstein-Uhlenbeck process using quantum method
- Quantum simulation for SDEs
- Schrödingerization for stochastic differential equations
- Quantum mean-reverting process solver

------

## 9. Use Cases

- Financial stochastic modeling (e.g., interest rates, asset prices)
- Physical Brownian particle dynamics
- Mean-reverting process simulation
- Quantum acceleration for SDEs

------

### Summary

This skill provides a **complete quantum solution for the 1D Ornstein-Uhlenbeck process**:

- SDE → quantum Hamiltonian pipeline
- Warped phase + FFT
- Crank–Nicolson quantum evolution
- Exact solution validation
- Trajectory reconstruction & plotting
- Quantum circuit visualization
- Fully aligned with your code and mathematical framework