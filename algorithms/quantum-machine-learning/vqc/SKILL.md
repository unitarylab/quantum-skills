---
name: vqc
description: Skill for understanding, using, and implementing the Variational Quantum Classifier (VQC) for Iris dataset classification with data re-uploading and Parameter Shift Rule via the VQCAlgorithm class.
---

# Variational Quantum Classifier (VQC)

## Purpose

VQC applies a parameterized quantum circuit to supervised classification. This implementation classifies the Iris dataset (4 features, 3 classes) using data re-uploading: each layer re-encodes the input features before the trainable rotation. Gradients are computed via the Parameter Shift Rule.

Use this skill when you need to:
- Classify tabular data with a hybrid quantum-classical neural network.
- Demonstrate quantum machine learning with the Parameter Shift Rule.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Load and standardize the Iris dataset (4 features, 3 classes; 80/20 split).
2. For each batch, evaluate the circuit at $\theta$, $\theta+\pi/2$, $\theta-\pi/2$ per parameter (Parameter Shift).
3. Compute gradients of the cross-entropy loss w.r.t. $\theta$.
4. Update $\theta$ with Adam optimizer.
5. Evaluate accuracy on the test set each epoch.

## Prerequisites

- Pauli-Z expectation values as classifier logits.
- Parameter Shift Rule for quantum gradients.
- Adam optimizer; cross-entropy loss.
- `torch`, `numpy`, `sklearn`, `Circuit`.

## Using the Provided Implementation

```python
from unitarylab_algorithms import VQCAlgorithm

algo = VQCAlgorithm(seed=42)
result = algo.run(
    n_layers=3,
    epochs=20,
    lr=0.05,
    batch_size=16,
    backend='torch'
)

print(f"Final test accuracy: {result['accuracy']:.2%}")
print(result['plot'])
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `42` | Random seed for `torch` and `numpy`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_layers` | `int` | `3` | Variational depth (number of data re-upload + rotation layers). |
| `epochs` | `int` | `20` | Training epochs over the full dataset. |
| `lr` | `float` | `0.05` | Adam learning rate. |
| `batch_size` | `int` | `16` | Mini-batch size. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str\|None` | `None` | Output directory for plots and circuit diagrams. |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `accuracy` | `float` | Final test set accuracy (fraction in [0, 1]). |
| `loss_history` | `List[float]` | Training loss per epoch. |
| `acc_history` | `List[float]` | Test accuracy per epoch. |
| `circuit` | `Circuit` | Example circuit (first training sample). |
| `circuit_path` | `str` | Path to circuit SVG. |
| `plot_path` | `str` | Path to training curve PNG. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`VQCAlgorithm` in `algorithm.py` implements a supervised quantum classifier in five stages using data re-uploading, parameter shift gradients, and Adam optimization.

**`run(n_layers, epochs, lr, batch_size, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | `_load_iris_data()` loads sklearn Iris; normalizes to $[-\pi/2, \pi/2]$; `theta = nn.Parameter(torch.rand(...))` | Data preparation and parameter init |
| 2 — Circuit Mapping | Pre-computes `observables = [_get_observable(i, 4) for i in 1..3]`; builds draw-only example circuit | Architecture preview and observable construction |
| 3 — Training Loop | Per epoch, per batch: manual parameter shift gradient via `_get_batch_logits(xb, th_p, ...)` and `_get_batch_logits(xb, th_m, ...)`; `optimizer.zero_grad()` + `theta.grad = grad_theta` + `optimizer.step()` | Quantum gradient-based training |
| 4 — Evaluation | `_evaluate(x_test, y_test, ...)` computes final accuracy | Test accuracy computation |
| 5 — Export | `path_circ = qc_draw.draw(...)`; `_generate_all_plots(loss_history, acc_history, ...)` | Saves circuit + training curve + accuracy curve |

**Helper Methods:**

- **`_load_iris_data()`** — Loads `sklearn.datasets.load_iris()`, applies `StandardScaler`, normalizes to $[-\pi/2, \pi/2]$, and stratified 80/20 splits. Returns PyTorch tensors.
- **`_get_observable(target, total)`** — Builds the $Z_\text{target} \otimes I_\text{rest}$ matrix via `torch.kron`.
- **`_build_circuit(x, theta, backend)`** — Creates `Circuit(4, backend=backend)`. Per layer $l$: `ry(x[q], q)` for each qubit (data encoding), `ry(theta[q,l], q)` (variational), then `cx(q, (q+1)%4)` entanglement (all but last layer).
- **`_get_batch_logits(x_batch, theta, observables, backend)`** — Loops over samples: builds circuit, executes, converts to PyTorch, computes `(psi† obs psi).real` for each observable. Returns `(batch_size, 3)` logit tensor.
- **`_evaluate(x_test, y_test, theta, observables, backend)`** — Same as logits computation but returns argmax accuracy.

**Key training detail:** Parameter shift is implemented manually as two forward passes (`th_p` and `th_m` differing by `±π/2`) for **every single parameter** in every batch. Total circuit evaluations per epoch: `2 * n_qubits * n_layers * batch_count`.

**Data flow:** Iris data → `_build_circuit` × batches → `_get_batch_logits` → CrossEntropyLoss → manual parameter shift → `theta.grad` → Adam step → final params → `_evaluate` → result dict.

## Understanding the Key Quantum Components
Each circuit layer encodes the 4 input features as $R_y(\pi \cdot x_i)$ rotations before the trainable rotations $R_y(\theta_{i,l})$. Re-uploading repeats this encoding in every layer, which is necessary for the circuit to retain input information throughout all layers (unlike classical networks, quantum circuits cannot copy states).

### 2. Parameterized Circuit (Ansatz)
Each layer $l$:
```
Ry(π·x[0])  Ry(π·x[1])  Ry(π·x[2])  Ry(π·x[3])   ← data encoding
Ry(θ[0,l])  Ry(θ[1,l])  Ry(θ[2,l])  Ry(θ[3,l])   ← trainable params
CX(0→1)  CX(1→2)  CX(2→3)                          ← entanglement
```

### 3. Pauli-Z Measurement as Logits
Three qubits (1, 2, 3) are measured in the $Z$ basis. Their expectation values $\langle Z_1\rangle, \langle Z_2\rangle, \langle Z_3\rangle$ form a 3-component logit vector for 3-class cross-entropy softmax classification.

### 4. Parameter Shift Rule
For a parameter $\theta_k$ entering as $R_y(\theta_k)$:
$$\frac{\partial \langle O\rangle}{\partial \theta_k} = \frac{1}{2}\left[\langle O\rangle_{\theta_k + \pi/2} - \langle O\rangle_{\theta_k - \pi/2}\right]$$
This gives exact quantum gradients without finite differences. Requires $2 \times (\text{total params})$ circuit evaluations per backward pass.

### 5. Adam Optimization
The Parameter Shift gradients are used with Adam, combining first and second moment estimates for adaptive learning rates per parameter.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Data re-uploading $R_y(\pi x_i)$ per layer | `qc.ry(float(x[q]), q)` before variational rotations in each layer |
| Trainable rotations $R_y(\theta_{q,l})$ | `qc.ry(float(theta[q,l]), q)` after data encoding per layer |
| Entanglement ring $CX(q, (q+1)\%4)$ | `qc.cx(q, (q+1)%4)` for all qubits except last layer |
| Logit $\langle Z_k\rangle = \langle\psi|\hat{Z}_k|\psi\rangle$ | `(psi_t.conj() @ obs @ psi_t).real.item()` for observable $k$ |
| Observable $Z_k \otimes I_\text{rest}$ | `_get_observable(target, total)` via `torch.kron` |
| Parameter shift: $\partial\langle O\rangle/\partial\theta_k$ | `(loss_p - loss_m) * 0.5` where `th_p/m = theta ± π/2` |
| Adam optimizer update | `torch.optim.Adam([theta], lr=lr)` |
| Cross-entropy loss $\mathcal{L}$ | `torch.nn.CrossEntropyLoss()(10 * logits, yb)` (scale 10×) |
| Iris dataset, 4 features → 4 qubits | `_load_iris_data()` → `torch.tensor(xt)` of shape `(N, 4)` |

**Notes on implementation:** The observable scale factor `10 *` in the loss computation boosts logit magnitudes before softmax — without it, the near-zero $\langle Z\rangle$ values lead to nearly uniform class probabilities and slow convergence. This is an engineering choice not reflected in the theory.

## Mathematical Deep Dive
$$\mathcal{L} = -\frac{1}{|B|}\sum_{i\in B} \sum_{c=1}^3 y_{ic}\log[\text{softmax}(\mathbf{z}_i)_c]$$

**Parameter Shift gradient:**
$$\nabla_\theta E = \frac{1}{2}[E(\theta+\pi/2) - E(\theta-\pi/2)]$$

**Data re-uploading expressibility:** The $n\_layers$-deep re-uploading circuit for $d$ features can represent polynomials of degree up to $n\_layers$ in the Fourier frequencies of the input.

## Hands-On Example

```python
from unitarylab_algorithms import VQCAlgorithm

# Deeper model for better accuracy
algo = VQCAlgorithm(seed=7)
result = algo.run(n_layers=5, epochs=30, lr=0.03, batch_size=8)

print(f"Best accuracy: {max(result['acc_history']):.2%}")
loss_end = result['loss_history'][-1]
print(f"Final loss: {loss_end:.4f}")
```

## Implementing Your Own Version

The following skeleton reconstructs the VQC data-reuploading circuit and Parameter Shift training loop from `VQCAlgorithm`.

```python
# Simplified reconstruction — mirrors VQCAlgorithm._build_circuit(), _get_batch_logits()
import numpy as np
import torch
from unitarylab.core import Circuit

def build_circuit(x: np.ndarray, theta: torch.Tensor,
                  backend: str = 'torch') -> Circuit:
    """
    Data re-uploading VQC for n_qubits=4 (Iris features).
    theta shape: (n_qubits, n_layers).
    Each layer: Rz(x)*Ry(theta[q,l])*Rz(x) on each qubit + CNOT ring.
    """
    n_qubits = theta.shape[0]
    n_layers = theta.shape[1]
    gs = Circuit(n_qubits, backend=backend)
    for l in range(n_layers):
        for q in range(n_qubits):
            gs.rz(float(x[q % len(x)]), q)       # encode feature
            gs.ry(float(theta[q, l]), q)           # trainable rotation
            gs.rz(float(x[q % len(x)]), q)       # encode feature again
        for q in range(n_qubits - 1):
            gs.cx(q, q + 1)                        # entanglement
    return gs

def get_pauli_z(qubit_idx: int, n_qubits: int) -> np.ndarray:
    """Build the n_qubit-system Z observable for qubit qubit_idx."""
    ops = [np.eye(2) for _ in range(n_qubits)]
    ops[qubit_idx] = np.array([[1., 0.], [0., -1.]])
    result = ops[0]
    for op in ops[1:]: result = np.kron(result, op)
    return result

def vqc_logits(x: np.ndarray, theta: torch.Tensor,
               n_classes: int = 3, backend: str = 'torch') -> torch.Tensor:
    """Compute Pauli-Z expectation values as classification logits."""
    n_qubits = theta.shape[0]
    gs = build_circuit(x, theta, backend)
    sv = np.asarray(gs.execute()).flatten()
    logits = []
    for c in range(n_classes):
        Z_c = get_pauli_z(c, n_qubits)
        exp_val = float(np.real(sv.conj() @ Z_c @ sv))
        logits.append(exp_val)
    return torch.tensor(logits, dtype=torch.float64)
```

**Component roles**:
- `build_circuit` — implements data re-uploading: each layer encodes `x` via `Rz`, applies trainable `Ry`, re-encodes `x` via `Rz`, then adds a CNOT entanglement ladder.
- `get_pauli_z`/`vqc_logits` — computes Pauli-Z expectation values $\langle\psi|Z_c|\psi\rangle$ as classification logits for each of `n_classes` output neurons; mirrors `_get_observable` + `_get_batch_logits`.
- A full training loop applies Parameter Shift (`theta[q,l] ± π/2`) to compute exact gradients, then uses `CrossEntropyLoss(logits, labels)` via Adam.

## Debugging Tips

1. **Low accuracy (<80%)**: Increase `n_layers` (5+) and `epochs` (40+). Iris is nearly linearly separable; 3-4 layers at 20 epochs should reach ~90%.
2. **Gradient computation is slow**: Each epoch requires $2 \times n\_layers \times n\_qubits$ circuit evaluations per batch. Parameter Shift is exact but expensive. Reduce `n_layers` or `batch_size` for speed.
3. **`lr` too large**: Causes loss oscillation. Use `lr=0.01–0.05` for stable training.
4. **Fixed dataset**: The Iris dataset is loaded internally; `x_train` / `y_train` are not user-supplied. To use custom data, subclass `VQCAlgorithm`.
5. **`n_qubits` is fixed at 4**: Matches Iris feature dimension. Changing circuit depth is done through `n_layers`, not `n_qubits`.
