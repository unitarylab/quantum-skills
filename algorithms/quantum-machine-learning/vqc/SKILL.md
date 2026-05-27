---
name: vqc
description: Skill for understanding, using, and implementing the Variational Quantum Classifier (VQC) for Iris dataset classification with data re-uploading and Parameter Shift Rule via the VQCAlgorithm class.
---

# Variational Quantum Classifier (VQC)

## Purpose

VQC applies a parameterized quantum circuit to supervised classification. This implementation classifies the Iris dataset (4 features, 3 classes) using a single encoding layer followed by multiple variational layers; gradients are computed via the Parameter Shift Rule.

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
from algorithms.quantum_machine_learning.vqc.algorithm import VQCAlgorithm

algo = VQCAlgorithm()
result = algo.run(
    layers=3,
    epochs=20,
    lr=0.05,
    batch_size=16,
    backend='torch'
)

print(f"Final test accuracy: {result['Final Accuracy']:.2%}")
print(result['plot'])
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output text formatting mode. |
| `algo_dir` | `str\|None` | `None` | Output directory for results; auto-derived from cwd when `None`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `layers` | `int` | `3` | Variational depth (number of rotation + entanglement layers). |
| `epochs` | `int` | `20` | Training epochs over the full dataset. |
| `lr` | `float` | `0.05` | Adam learning rate. |
| `batch_size` | `int` | `16` | Mini-batch size. |
| `backend` | `str` | `'torch'` | Simulation backend passed to `Circuit.execute()`. |
| `device` | `str` | `'cpu'` | Compute device passed to `Circuit.execute()`. |
| `dtype` | `type` | `np.complex128` | Complex dtype for statevector simulation. |

## Return Fields

The return value is built by `_build_return_dict(success, circuit_path, filepath, circuit)` merged with `self.output`.

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` on error. |
| `circuit_path` | `str` | Absolute path to the saved circuit SVG diagram. |
| `plot` | `List[Dict]` | List of output files: each entry is `{"format": ext, "filename": abs_path}`. |
| `circuit` | `Circuit` | Example `Circuit` object built from the first training sample. |
| `Final Loss` | `float` | Cross-entropy loss at the last epoch. |
| `Final Accuracy` | `float` | Test set accuracy at the last epoch (fraction in [0, 1]). |
| `Quantal Computation Time (s)` | `float` | Wall-clock time (seconds) for the core quantum training loop. |

## Implementation Architecture

`VQCAlgorithm` in `algorithm.py` implements a supervised quantum classifier in five stages using a single-encoding PQC, parameter shift gradients, and Adam optimization.

**`run(layers, epochs, lr, batch_size, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | `_load_iris_data()` loads sklearn Iris; normalizes to $[-\pi/2, \pi/2]$; `theta = nn.Parameter(torch.rand(...))` | Data preparation and parameter init |
| 2 — Circuit Mapping | Pre-computes `observables = [_get_observable(i, 4) for i in 1..3]`; builds draw-only example circuit | Architecture preview and observable construction |
| 3 — Training Loop | Per epoch, per batch: manual parameter shift gradient via `_get_batch_logits(xb, th_p, ...)` and `_get_batch_logits(xb, th_m, ...)`; `optimizer.zero_grad()` + `theta.grad = grad_theta` + `optimizer.step()` | Quantum gradient-based training |
| 4 — Evaluation | `_evaluate(x_test, y_test, ...)` computes final accuracy | Test accuracy computation |
| 5 — Export | `save_circuit(qc_draw)`; `_generate_all_plots(loss_history, acc_history, algo_dir)` returns SVG path | Saves circuit diagram and training metrics plot |

**Helper Methods:**

- **`_load_iris_data()`** — Loads `sklearn.datasets.load_iris()`, applies `StandardScaler`, normalizes to $[-\pi/2, \pi/2]$, and stratified 80/20 splits. Returns PyTorch tensors.
- **`_get_observable(target, total)`** — Builds the $Z_\text{target} \otimes I_\text{rest}$ matrix via `torch.kron`.
- **`_build_circuit(x, theta)`** — Creates `Circuit(4)`. First encodes all 4 features once via `ry(x[q], q)`. Then for each layer $l$: `ry(theta[q,l], q)` (variational) for each qubit, followed by the full entanglement ring `cx(q, (q+1)%4)` for all 4 qubits — except no entanglement is added after the last layer.
- **`_get_batch_logits(x_batch, theta, observables)`** — Loops over samples: calls `_build_circuit`, executes with `backend`, `device`, `dtype`, converts to PyTorch, computes `(bra @ obs @ psi).real` for each observable. Returns `(batch_size, 3)` logit tensor.
- **`_evaluate(x_test, y_test, theta, obs)`** — Decorated with `@torch.no_grad()`; calls `_get_batch_logits` and returns argmax accuracy as a float.

**Key training detail:** Parameter shift is implemented manually as two forward passes (`th_p` and `th_m` differing by `±π/2`) for **every single parameter** in every batch. Total circuit evaluations per epoch: `2 * n_qubits * layers * batch_count`.

**Data flow:** Iris data → `_build_circuit` × batches → `_get_batch_logits` → CrossEntropyLoss → manual parameter shift → `theta.grad` → Adam step → final params → `_evaluate` → result dict.

## Understanding the Key Quantum Components
The 4 input features are encoded once at the beginning of the circuit as $R_y(x_i)$ rotations. The subsequent `layers` variational layers each apply trainable $R_y(\theta_{q,l})$ rotations followed by a ring of CNOT gates; no entanglement is added after the final layer.

### 2. Parameterized Circuit (Ansatz)
```
─── Encoding (once) ───
Ry(x[0])  Ry(x[1])  Ry(x[2])  Ry(x[3])            ← data encoding

─── Layer l = 0 .. layers-2 ───
Ry(θ[0,l])  Ry(θ[1,l])  Ry(θ[2,l])  Ry(θ[3,l])   ← trainable params
CX(0→1)  CX(1→2)  CX(2→3)  CX(3→0)               ← full ring entanglement

─── Last layer (no entanglement) ───
Ry(θ[0,L-1])  Ry(θ[1,L-1])  Ry(θ[2,L-1])  Ry(θ[3,L-1])
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
| Single encoding $R_y(x_i)$ at circuit start | `qc.ry(float(x[q]), q)` applied once for all qubits before the variational layers |
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

**Circuit expressibility:** The `layers`-deep variational circuit captures increasing entanglement structure as depth grows; deeper circuits can represent more complex decision boundaries.

## Hands-On Example

```python
from algorithms.quantum_machine_learning.vqc.algorithm import VQCAlgorithm

# Deeper model for better accuracy
algo = VQCAlgorithm()
result = algo.run(layers=5, epochs=30, lr=0.03, batch_size=8)

print(f"Final accuracy: {result['Final Accuracy']:.2%}")
print(f"Final loss: {result['Final Loss']:.4f}")
print(f"Quantum compute time: {result['Quantal Computation Time (s)']:.4f}s")
print(result['plot'])  # list of {"format": ..., "filename": ...}
```

## Implementing Your Own Version

The following skeleton reconstructs the VQC data-reuploading circuit and Parameter Shift training loop from `VQCAlgorithm`.

```python
# Simplified reconstruction — mirrors VQCAlgorithm._build_circuit(), _get_batch_logits()
import numpy as np
import torch
from unitarylab import Circuit

def build_circuit(x: np.ndarray, theta: torch.Tensor) -> Circuit:
    """
    VQC for n_qubits=4 (Iris features).
    theta shape: (n_qubits, layers).
    Encodes features once, then alternates variational rotations + CNOT ring.
    No entanglement after the last layer.
    """
    n_qubits = theta.shape[0]
    n_layers = theta.shape[1]
    qc = Circuit(n_qubits)
    for q in range(n_qubits):
        qc.ry(float(x[q]), q)                      # encode feature (once)
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(theta[q, l]), q)            # trainable rotation
        if l < n_layers - 1:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)       # full ring entanglement
    return qc

def get_pauli_z(qubit_idx: int, n_qubits: int) -> np.ndarray:
    """Build the n_qubit-system Z observable for qubit qubit_idx."""
    ops = [np.eye(2) for _ in range(n_qubits)]
    ops[qubit_idx] = np.array([[1., 0.], [0., -1.]])
    result = ops[0]
    for op in ops[1:]: result = np.kron(result, op)
    return result

def vqc_logits(x: np.ndarray, theta: torch.Tensor,
               n_classes: int = 3, backend: str = 'torch',
               device: str = 'cpu', dtype=np.complex128) -> torch.Tensor:
    """Compute Pauli-Z expectation values as classification logits."""
    n_qubits = theta.shape[0]
    qc = build_circuit(x, theta)
    state0 = np.zeros((2 ** n_qubits, 1), dtype=np.complex128); state0[0, 0] = 1.0
    psi = np.asarray(
        qc.execute(initial_state=state0, backend=backend, device=device, dtype=dtype).state
    ).flatten()
    psi_t = torch.as_tensor(psi).to(torch.complex128)
    logits = []
    for c in range(n_classes):
        Z_c = torch.as_tensor(get_pauli_z(c, n_qubits)).to(torch.complex128)
        exp_val = (psi_t.conj() @ Z_c @ psi_t).real.item()
        logits.append(exp_val)
    return torch.tensor(logits, dtype=torch.float64)
```

**Component roles**:
- `build_circuit` — encodes `x` once via `Ry` then applies `layers` variational `Ry` blocks with a full CNOT ring (skipped after the last layer); mirrors `VQCAlgorithm._build_circuit`.
- `get_pauli_z`/`vqc_logits` — computes Pauli-Z expectation values $\langle\psi|Z_c|\psi\rangle$ as classification logits for each of `n_classes` output neurons; mirrors `_get_observable` + `_get_batch_logits`. Requires explicit `initial_state` and backend/device/dtype args matching `Circuit.execute()`.
- A full training loop applies Parameter Shift (`theta[q,l] ± π/2`) to compute exact gradients, then uses `CrossEntropyLoss(10 * logits, labels)` via Adam (the `10 ×` scale factor is essential for convergence).

## Debugging Tips

1. **Low accuracy (<80%)**: Increase `layers` (5+) and `epochs` (40+). Iris is nearly linearly separable; 3-4 layers at 20 epochs should reach ~90%.
2. **Gradient computation is slow**: Each epoch requires $2 \times layers \times n\_qubits$ circuit evaluations per batch. Parameter Shift is exact but expensive. Reduce `layers` or `batch_size` for speed.
3. **`lr` too large**: Causes loss oscillation. Use `lr=0.01–0.05` for stable training.
4. **Fixed dataset**: The Iris dataset is loaded internally; `x_train` / `y_train` are not user-supplied. Falls back to the hardcoded dataset arrays when `sklearn` is unavailable.
5. **`n_qubits` is fixed at 4**: Matches Iris feature dimension. Changing circuit depth is done through `layers`, not `n_qubits`.
