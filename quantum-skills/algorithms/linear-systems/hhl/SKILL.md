---
name: hhl
description: A quantum algorithm for solving linear systems of equations, providing exponential speedup over classical methods for certain types of problems. This skill includes efficient implementations and educational resources for understanding and utilizing quantum linear systems algorithms in various applications.
---

# One Step to Run HHL Example
```bash
python ./scripts/algorithm.py
```

# HHL Algorithm (Harrow-Hassidim-Lloyd) Skill Guide

## Overview

**HHL Algorithm** is one of the most important quantum algorithms, solving the fundamental problem of **linear equations**: given a Hermitian matrix $A$ and vector $\mathbf{b}$, solve:

$$A\mathbf{x} = \mathbf{b}$$

Unlike classical solvers, **HHL doesn't directly output the solution vector** but prepares a quantum state:

$$|\mathbf{x}\rangle \propto A^{-1}|\mathbf{b}\rangle$$

### Why HHL Matters:

**Exponential Speedup Potential**: For sparse, well-conditioned matrices:
- **Classical** (Conjugate Gradient): $O(N s \kappa \log(1/\epsilon))$
- **Quantum (HHL)**: $O(\log(N) s^2 \kappa^2 / \epsilon)$

When $\kappa$ and $1/\epsilon$ are poly-logarithmic in $N$, quantum gives **exponential advantage**!

### Real-World Applications:

1. **Computational Physics**: Solving differential equations (discretized into linear systems)
2. **Machine Learning**: Fast solving in kernel methods, linear regression
3. **Chemistry**: Simulating molecular properties
4. **Optimization**: Gradient descent acceleration
5. **Finance**: Option pricing, risk modeling

---


## Prerequisites

- **Essential knowledge**:
  - Quantum Phase Estimation (QPE) - required!
  - Eigenvalues and eigenvectors
  - Hermitian matrices and spectral decomposition
  - Controlled rotations and ancilla qubits
  - Post-selection and conditional measurement
- **Mathematical background**: Linear algebra, complex numbers, eigenvalue decomposition
---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms.linear_algebra import HHLAlgorithm
from engine import GateSequence
import numpy as np

# Step 1: Define a linear system Ax = b
# Create a simple 2x2 Hermitian matrix
A = np.array([
    [1.0, 0.5],
    [0.5, 2.0]
], dtype=complex)

# Create a normalized vector b
b = np.array([1.0, 1.0], dtype=complex)
b = b / np.linalg.norm(b)  # Normalize

# Step 2: Run HHL algorithm
algo = HHLAlgorithm()
result = algo.run(
    A=A,
    b=b,
    d=5,  # 5 bits phase precision
    backend='torch'
)

# Step 3: Inspect results
print(result['plot'])
print(f"Quantum solution: {result['solution_quantum']}")
print(f"Classical solution: {result['solution_classical']}")
print(f"Error: {np.linalg.norm(result['solution_quantum'] - result['solution_classical']):.6e}")
print(f"Post-selection success probability: {result['post_selection_prob']:.6f}")
```

### Core Parameters Explained

```python
algo.run(
    A: np.ndarray,           # Hermitian matrix (NxN, N must be power of 2)
    b: np.ndarray,           # Right-hand side vector (will be normalized)
    d: int,                  # Phase register precision bits
    t: Optional[float] = None,  # Evolution time for U = exp(iAt)
    backend: str = 'torch',  # Simulation backend
    algo_dir: str = './hhl_results'  # Output directory
)
```

**Return dictionary contains:**
- `solution_quantum`: Quantum-computed solution vector
- `solution_classical`: Exact classical solution (for comparison)
- `state_system`: The quantum state |x⟩ before post-selection
- `post_selection_prob`: Probability of ancilla=1 & phase=|0⟩
- `full_circuit`: Complete HHL quantum circuit
- `circuit_path`: Location of circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

---

## Understanding the Core Components

### 1. Problem Setup and Spectral Decomposition

Linear system: $A\mathbf{x} = \mathbf{b}$

Since $A$ is Hermitian, decompose:
$$A =  \sum_{j=0}^{N-1} \lambda_j |u_j\rangle\langle u_j|$$

Solution in eigenbasis:
$$A^{-1}|\mathbf{b}\rangle = A^{-1}\sum_j \beta_j |u_j\rangle = \sum_j \frac{\beta_j}{\lambda_j}|u_j\rangle$$

**HHL's Goal**: Implement this in quantum state form while using the eigenbasis change for efficiency.

### 2. HHL Five-Step Pipeline

```python
def run(self, A, b, d, t=None):
    """
    Complete HHL pipeline:
    
    Step 1: Prepare |b⟩ in system register
    Step 2: Run QPE on U = e^(iAt) to extract eigenvalues as phases
    Step 3: Apply controlled reciprocal rotation: λ_j → 1/λ_j
    Step 4: Unmeasure (inverse QPE) to disentangle registers
    Step 5: Post-select on ancilla=1, phase=|0⟩ to get solution
    """
    
    # STEP 1: Initialization and preprocessing
    A_hermitian = validate_and_normalize(A)
    b_normalized = b / norm(b)
    
    # Extract eigenvalues to auto-tune evolution time t
    eigenvalues = np.linalg.eigvalsh(A)
    lambda_min = min(abs(eigenvalues))
    lambda_max = max(abs(eigenvalues))
    
    if t is None:
        # Automatically choose t to avoid phase wraparound
        t = (1 - 1/2^d) / lambda_max
    
    # STEP 2: Prepare quantum circuit
    total_qubits = 1 + d + log2(N)  # ancilla + phase + system
    circuit = GateSequence(total_qubits)
    
    # Initialize system register with |b⟩
    circuit.initialize(b_normalized, system_qubits)
    
    # Create unitary U = e^(iAt)
    U_matrix = exp(1j * A * t)
    U_circuit = unitary_from_matrix(U_matrix)
    
    # STEP 3: Apply QPE on U
    qpe_circuit = build_qpe(U_circuit, d)
    circuit.append(qpe_circuit, phase_qubits + system_qubits)
    
    # STEP 4: Controlled reciprocal rotation (SECRET SAUCE!)
    # For each eigenvalue λ_j (encoded as phase), apply:
    # rotation angle θ_j such that sin(θ_j/2) ∝ 1/λ_j
    reciprocal_rotation = build_controlled_reciprocal(d, t, lambda_min)
    circuit.append(reciprocal_rotation, [ancilla] + phase_qubits)
    
    # STEP 5: Inverse QPE (uncompute phase register)
    circuit.append(qpe_circuit.dagger(), phase_qubits + system_qubits)
    
    # STEP 6: Measure and post-select
    return extract_solution_via_postselection(circuit)
```

### 3. Quantum Phase Estimation Component

HHL uses QPE to extract eigenvalues of $A$ by working with $U = e^{iAt}$:

```python
def setup_phase_estimation(A, t, d):
    """
    Create U = exp(iAt) for phase estimation.
    
    Why this works:
    - If |u_j⟩ is eigenvector of A with eigenvalue λ_j
    - Then |u_j⟩ is also eigenvector of U with eigenvalue e^(iλ_j t)
    - QPE extracts the phase: φ_j = λ_j t / (2π)
    
    The phase φ_j encodes the eigenvalue λ_j!
    """
    # Diagonalize A
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Create diagonal matrix with exp(i λ_j t) phases
    phases = np.exp(1j * eigenvalues * t)
    U = eigenvectors @ np.diag(phases) @ eigenvectors.conj().T
    
    return U  # Use this in QPE!
```

### 4. Controlled Reciprocal Rotation (The Key Non-Linear Step)

This is where the "magic" happens - encoding $1/\lambda$ onto the ancilla:

```python
def _controlled_reciprocal_rotation(self, d, t, k_start, 
                                   signed_phase=False, backend='torch'):
    """
    The ONLY non-linear step in HHL!
    
    For each phase qubit k (representing eigenvalue λ_k):
    - Apply a controlled rotation on ancilla
    - Rotation angle θ_k such that:
      sin(θ_k/2) = C / λ_k, where C is normalization constant
    
    Result: Ancilla amplitude encodes 1/λ_k
    
    This is what enables solving linear systems!
    """
    gs = GateSequence(d + 1)
    controls = list(range(1, d + 1))  # Phase qubits
    target = 0                          # Ancilla qubit
    
    grid = 2 ** d
    C = k_start  # Normalization constant (minimum eigenvalue)
    
    # For each possible phase value k
    for k in range(k_start, grid):
        # Map phase k to eigenvalue (decoded using iQFT earlier)
        eigenvalue_approx = k * (lambda_max / grid)
        
        # Compute rotation angle for ancilla
        # We want: sin(θ/2) ≈ C / eigenvalue
        sine_value = C / eigenvalue_approx  # Must be ≤ 1
        theta = 2.0 * np.arcsin(min(sine_value, 1.0))
        
        # Apply controlled rotation when phase qubits = k
        # (Flip qubits where k has 0 bits, apply MCRy, flip back)
        flip_qubits_for_k(gs, k)
        gs.mcry(theta, controls, target)  # Multi-controlled RY
        unflip_qubits_for_k(gs, k)
    
    return gs
```

### 5. Post-Selection: Extracting the Solution

After inverse QPE, measure ancilla and post-select:

```python
def _postselect_solution_state(self, state, scale_factor, d, n_sys):
    """
    Extract the solution from measured state.
    
    Post-selection strategy:
    - Measure ancilla (index 0): we want outcome |1⟩
    - Measure phase register: we want outcome |0⟩...⟩
    
    When both conditions are met:
    - System register contains the solution!
    - Ancilla=1 means the rotation "worked"
    - Phase=0 means phase register disentangled successfully
    """
    # Extract slice where ancilla = 1
    vec_anc1 = state[1::2]  # Odd indices (ancilla=1)
    
    # Extract slice where phase = |0...0⟩ (first 2^d elements)
    stride = 2 ** d
    vec_sys = vec_anc1[0::stride]  # Every stride-th element
    
    # Normalize and scale
    p_success = np.vdot(vec_sys, vec_sys).real
    solution = np.array(vec_sys) * scale_factor / np.sqrt(p_success)
    
    return solution, p_success
```

---

## Hands-On Example: Small Tridiagonal System

Let's solve a 4x4 tridiagonal system:

```python
from engine.algorithms.linear_algebra import HHLAlgorithm
import numpy as np

# Create a 4x4 tridiagonal matrix (simple, well-conditioned)
N = 4
A = np.zeros((N, N), dtype=complex)
np.fill_diagonal(A, 2.0)           # Diagonal: 2
np.fill_diagonal(A[1:], -1.0)      # Upper: -1
np.fill_diagonal(A[:, 1:], -1.0)   # Lower: -1

# Right-hand side
b = np.ones(N, dtype=complex)

print("Matrix A:")
print(A)
print("\nVector b:", b)

# Classical solution for comparison
x_classical = np.linalg.solve(A, b)
print(f"\nClassical solution: {x_classical}")

# Run HHL
algo = HHLAlgorithm()
result = algo.run(A=A, b=b, d=6)

print(f"\nQuantum solution: {result['solution_quantum']}")
print(f"Error: {np.linalg.norm(result['solution_quantum'] - x_classical):.6e}")
print(f"Success probability: {result['post_selection_prob']:.6f}")
```

**Expected Output**: Quantum and classical solutions should match closely, with error < $10^{-3}$

---

## Implementing Your Own HHL

### Complete Implementation Template

```python
import numpy as np
from typing import Tuple, Dict, Any, Optional
from engine import GateSequence, State

class MyHHL:
    """User-implemented HHL Algorithm."""
    
    def run(self, A: np.ndarray, b: np.ndarray, d: int,
            t: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve Ax = b using quantum HHL.
        
        Args:
            A: Hermitian NxN matrix (N = power of 2)
            b: Right-hand side vector
            d: Phase precision bits
            t: Evolution time (auto-computed if None)
        
        Returns: Solution vector and metadata
        """
        # STEP 1: Validate and prepare
        self._validate_input(A, b)
        A = np.asarray(A, dtype=complex)
        b = np.asarray(b, dtype=complex).reshape(-1)
        N = A.shape[0]
        n_sys = int(np.log2(N))
        
        # Normalize b
        b_norm = np.linalg.norm(b)
        b_state = b / b_norm
        
        # Extract eigenvalues for t selection
        eigenvals = np.linalg.eigvalsh(A)
        lambda_min, lambda_max = float(np.min(np.abs(eigenvals))), \
                                 float(np.max(np.abs(eigenvals)))
        
        if t is None:
            # Auto-tune: avoid phase wraparound
            t = (1 - 1/(2**d)) / lambda_max
        
        # STEP 2: Build quantum circuit
        d_total = 1 + d + n_sys  # ancilla + phase + system
        circuit = GateSequence(d_total)
        
        anc = 0
        phase_qubits = list(range(1, d + 1))
        sys_qubits = list(range(d + 1, d_total))
        
        # Initialize system register with |b⟩
        circuit.initialize(b_state, sys_qubits)
        
        # STEP 3: Create U = exp(iAt)
        U_mat = self._expi_hamiltonian(A, t)
        U_circ = self._make_unitary_circuit(U_mat)
        
        # STEP 4: Apply QPE
        qpe_circ = self._build_qpe(U_circ, d)
        circuit.append(qpe_circ, phase_qubits + sys_qubits)
        
        # STEP 5: Controlled reciprocal rotation
        k_start = int(np.floor(lambda_min * t * (2**d)))
        rot_circ = self._build_reciprocal_rotation(d, t, k_start)
        circuit.append(rot_circ, [anc] + phase_qubits)
        
        # STEP 6: Inverse QPE (uncompute)
        circuit.append(qpe_circ.dagger(), phase_qubits + sys_qubits)
        
        # STEP 7: Execute and extract solution
        state = circuit.execute()
        state_array = np.asarray(state, dtype=complex).reshape(-1)
        
        solution, p_success = self._extract_solution(
            state_array, b_norm, t, d, n_sys, k_start
        )
        
        return {
            "solution": solution,
            "classical_solution": np.linalg.solve(A, b),
            "success_prob": p_success,
            "circuit": circuit
        }
    
    def _validate_input(self, A, b):
        """Check Hermitian, power of 2, etc."""
        A = np.asarray(A, dtype=complex)
        N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError("A must be square")
        if not np.allclose(A, A.conj().T):
            raise ValueError("A must be Hermitian")
        if (N & (N - 1)) != 0:
            raise ValueError("N must be power of 2")
        return A
    
    def _expi_hamiltonian(self, A: np.ndarray, t: float) -> np.ndarray:
        """Compute U = exp(iAt)"""
        eigenvals, eigenvecs = np.linalg.eigh(A)
        phases = np.exp(1j * eigenvals * t)
        U = eigenvecs @ np.diag(phases) @ eigenvecs.conj().T
        return U
    
    def _make_unitary_circuit(self, U: np.ndarray) -> GateSequence:
        """Encode unitary matrix as quantum circuit"""
        n = int(np.log2(U.shape[0]))
        circuit = GateSequence(n)
        circuit.unitary(U, list(range(n)))
        return circuit
    
    def _build_qpe(self, U: GateSequence, d: int) -> GateSequence:
        """Build QPE circuit - VERY IMPORTANT!"""
        # This is identical to standalone QPE
        # Use the QPE algorithm you learned earlier
        pass
    
    def _build_reciprocal_rotation(self, d: int, t: float, 
                                  k_start: int) -> GateSequence:
        """Build the controlled reciprocal rotation circuit"""
        # For each phase value k, apply rotation angle
        # proportional to 1/λ_k
        pass
    
    def _extract_solution(self, state, b_norm, t, d, n_sys, k_start):
        """Extract solution via post-selection"""
        # Post-select on ancilla=1 & phase=|0⟩
        # Scale result back to original problem
        pass
```

---

## Mathematical Deep Dive

### Hermitian Encoding

For non-Hermitian $A$, use block embedding:
$$\tilde{A} = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}$$

Then solve $\tilde{A}\begin{pmatrix} 0 \\ \mathbf{x} \end{pmatrix} = \begin{pmatrix} \mathbf{b} \\ 0 \end{pmatrix}$

### Eigenvalue-to-Phase Encoding

Key insight: For eigenstate $|u_j\rangle$ of $A$ with eigenvalue $\lambda_j$:
$$U|u_j\rangle = e^{i\lambda_j t}|u_j\rangle$$

QPE extracts: $\phi_j = \frac{\lambda_j t}{2\pi} \in [0,1)$

This **encodes $\lambda_j$ as a measurable phase**!

### Reciprocal Transformation

For ancilla state contribution:
$$\sin(\theta_j/2) = \frac{C}{\lambda_j}$$

where $C \leq \lambda_{\min}$ is normalization. The controlled rotation implements:
$$|u_j\rangle|0\rangle \to |u_j\rangle\left(\sqrt{1-C^2/\lambda_j^2}|0\rangle + \frac{C}{\lambda_j}|1\rangle\right)$$

### Post-Selection Success Probability

$$P_{\text{success}} = \frac{C^2}{\lambda_{\max}^2} = O(1/\kappa^2)$$

where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number.

Can be amplified to $O(1/\kappa)$ using amplitude amplification!

### Complexity Analysis

| Resource | Classical | Quantum |
|----------|-----------|---------|
| **Time** | $O(Ns\kappa\log(1/\epsilon))$ | $O(s^2\kappa^2\log N/\epsilon)$ |
| **Space** | $O(N)$ | $O(\log N)$ |
| **Success Prob** | 100% | $O(1/\kappa^2)$ |

When $\kappa, 1/\epsilon = \text{poly}\log(N)$: **Exponential speedup!**

---

## Debugging Tips

1. **Non-Hermitian matrix error?**
   - Use block embedding: $\tilde{A} = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}$
   - Solve for $\begin{pmatrix} 0 \\ \mathbf{x} \end{pmatrix}$

2. **Success probability very low?**
   - Check condition number $\kappa = \lambda_{\max}/\lambda_{\min}$
   - Use amplitude amplification to improve probability
   - Ensure normalization constant $C < \lambda_{\min}$

3. **Phase precision insufficient?**
   - Increase $d$ (number of phase qubits)
   - Remember: precision ~ $1/2^d$
   - For condition number $\kappa$, need $d \approx \log(\kappa)$ bits minimum

4. **Eigenvalue misidentification?**
   - Check that evolution time $t$ doesn't cause wraparound
   - Verify: $\lambda_{\max} \cdot t < 1 - 1/2^d$
   - Consider signed phase mode for negative eigenvalues

5. **Circuit too large?**
   - Qubits scale as $O(\log N)$ - this is already exponential savings!
   - For $N=2^{20}$ (million equations), need only ~20 system qubits
   - Plus ~10 phase qubits for decent precision

---