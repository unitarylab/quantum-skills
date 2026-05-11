---
name: qnn
description: Skill for understanding, using, and implementing the Quantum Neural Network (QNN) with parameterized quantum circuits for supervised learning via the QNNAlgorithm class.
---

# Quantum Neural Network (QNN)

## Purpose

QNN implements a supervised learning model using a Parameterized Quantum Circuit (PQC). It encodes classical features into quantum states via angle encoding, applies variational Rx/Ry/Rz layers, and learns from labeled training data.

Use this skill when you need to:
- Apply a PQC-based neural network to arbitrary supervised learning tasks.
- Demonstrate the component structure of quantum machine learning models.

**Note:** The current implementation uses a simplified training loop for demonstration. The quantum state preparation and circuit structure are real; the forward-pass loss update is a simplified approximation of the full quantum gradient computation.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Encode each training sample's features as $R_x$ rotations (angle encoding).
2. Apply $L$ variational layers, each with per-qubit $R_x/R_y/R_z$ rotations and CNOT entanglement.
3. Simulate training via simplified loss computation over `epochs`.
4. Return the trained parameter state and training history.

## Prerequisites

- Parameterized quantum circuits (PQC); angle encoding.
- Supervised learning concepts; cross-entropy or MSE loss.
- `numpy`, `Circuit`, `Register`.

## Using the Provided Implementation

```python
import numpy as np
from unitarylab-algorithms import QNNAlgorithm

# Synthetic binary classification data
np.random.seed(0)
X_train = np.random.randn(40, 2)           # 40 samples, 2 features
y_train = (X_train[:, 0] > 0).astype(int) # Binary labels

algo = QNNAlgorithm(seed=42)
result = algo.run(
    x_train=X_train,
    y_train=y_train,
    n_qubits=2,
    layers=2,
    epochs=20,
    learning_rate=0.1,
    backend='torch'
)

print(f"Final accuracy: {result['accuracy']:.2%}")
print(result['plot'])
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `42` | Random seed for parameter initialization. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x_train` | `np.ndarray` | required | Training features, shape `(N_samples, N_features)`. |
| `y_train` | `np.ndarray` | required | Integer labels, shape `(N_samples,)`. |
| `n_qubits` | `int` | required | Number of qubits; must satisfy `n_classes ≤ 2^n_qubits`. |
| `layers` | `int` | `2` | Number of variational layers. |
| `epochs` | `int` | `20` | Training iterations over all samples. |
| `learning_rate` | `float` | `0.1` | Gradient step size. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str\|None` | `None` | Output directory for results. |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `accuracy` | `float` | Final training accuracy. |
| `loss_history` | `List[float]` | Loss per epoch. |
| `circuit` | `Circuit` | Example PQC circuit (first training sample). |
| `circuit_path` | `str` | Path to circuit SVG. |
| `plot_path` | `str` | Path to training curve PNG. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`QNNAlgorithm` in `algorithm.py` implements a parameterized quantum circuit classifier in five stages with one circuit builder. **Important:** the training loop in the current implementation uses a **simulated loss** (Gaussian-perturbed label MSE) rather than actual quantum gradient computation.

**`run(x_train, y_train, n_qubits, layers, epochs, learning_rate, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | Validates `num_classes <= 2^n_qubits`; initializes `self.params = np.random.uniform(0, 2π, (layers, n_qubits, 3))` | Parameter array shape `(L, N, 3)` for Rx/Rz/Ry per qubit per layer |
| 2 — Circuit Mapping | Creates `Circuit(Register('q', n_qubits), backend=backend)`; calls `_build_vqc_layer(gs, x_train[0], ...)` for visualization | Architecture preview only |
| 3 — Training Loop | For each epoch/sample: computes `prediction = label + Gaussian_noise * exp(-epoch/10)`; MSE loss simulation; no actual quantum gradient | **Simplified training placeholder** |
| 4 — Evaluation | `final_acc = 1/num_classes + (1 - 1/num_classes) * (1 - min(1, final_loss))` | Derived accuracy estimate (not actual measurement) |
| 5 — Export | `gs.draw(filename=circuit_path)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_build_vqc_layer(gs, x, params, n_qubits, layers, n_features)`** — For each layer `l`, per qubit `q`: applies `rx(x[q % n_features], q)` (data encoding), `rz(params[l,q,0], q)`, `ry(params[l,q,1], q)` (variational), then `cnot(q, q+1)` entanglement chain.

**Critical limitation note:** The `run()` method's training loop (Stage 3) does **not** call `_build_vqc_layer`, `gs.execute()`, or any quantum circuit during training. The loss is a mathematical placeholder based on Gaussian noise decaying with epoch. The `_build_vqc_layer` method is only invoked once in Stage 2 for visualization purposes.

**Data flow (actual):** `x_train[0]` → `_build_vqc_layer()` (visualization only) → `gs` drawn → simulated loss loop → approximated `final_acc` → result dict.

## Understanding the Key Quantum Components
Input features are mapped to qubit rotations:
$$|\psi_x\rangle = \bigotimes_{i=0}^{n-1} R_x(x_i) |0\rangle$$
Each feature angle is modulated per qubit.

### 2. Variational Layer Structure
Each of the $L$ layers applies:
```
Per-qubit: Rx(params[l,q,0]) → Ry(params[l,q,1]) → Rz(params[l,q,2])
Entanglement: CNOT(q, q+1) for q in range(n_qubits-1)
```
Parameters are stored as shape `(layers, n_qubits, 3)`.

### 3. Feature Encoding Constraint
The number of features is encoded cyclically: `x_feature_index = (qubit * layers + layer_idx) % n_features`. This allows re-encoding when $n\_features < n\_qubits \times n\_layers$.

### 4. Output Measurement
The computational basis probabilities $|c_0|^2, \ldots, |c_{2^n-1}|^2$ of the final state determine the predicted class. The class with the highest amplitude is the prediction.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Angle encoding $R_x(x_i)$ | `gs.rx(x[q % n_features], q)` per qubit in `_build_vqc_layer()` |
| Variational rotations $R_z(\theta_{l,q,0})$, $R_y(\theta_{l,q,1})$ | `gs.rz(params[l,q,0], q)`, `gs.ry(params[l,q,1], q)` |
| Entanglement CNOT chain | `gs.cnot(q, q+1)` for `q in range(n_qubits-1)` |
| Parameter array $\theta \in \mathbb{R}^{L \times N \times 3}$ | `self.params = np.random.uniform(0, 2π, (layers, n_qubits, 3))` |
| Training via gradient descent | **Not implemented**: Stage 3 uses Gaussian-noise MSE proxy |
| Loss convergence | `loss_history[epoch] = mean(epoch_errors)` based on label + noise |
| Final accuracy estimate | `1/K + (1-1/K)*(1 - min(1, final_loss))` — analytical proxy |

**Notes on implementation:** The PQC is fully constructed in `_build_vqc_layer()` and could theoretically be used for quantum gradient computation, but the `run()` training loop bypasses this entirely. If actual QNN training with parameter shift is needed, `_build_vqc_layer` must be called per sample with `gs.execute()` and the Parameter Shift Rule applied over `self.params`.

## Mathematical Deep Dive
$$U(\theta, x) = \prod_{l=1}^{L}\left[\prod_{q} R_x(\theta_{l,q,0})R_y(\theta_{l,q,1})R_z(\theta_{l,q,2}) \cdot \prod_{\text{edges}} \text{CNOT}\right] \cdot U_{\text{encode}}(x)$$

**Complexity:** The circuit has $3 \cdot l \cdot n$ single-qubit gates and $(n-1) \cdot l$ CNOT gates total.

**Training (simplified simulation):** The current implementation uses an approximated loss for demonstration purposes. For production use, full quantum backpropagation via the Parameter Shift Rule is required.

## Hands-On Example

```python
import numpy as np
from sklearn.datasets import load_iris
from unitarylab-algorithms import QNNAlgorithm

iris = load_iris()
X = iris.data[:, :2]    # first 2 features
y = (iris.target > 0).astype(int)  # binary: setosa vs. others

algo = QNNAlgorithm(seed=123)
result = algo.run(
    x_train=X, y_train=y,
    n_qubits=2, layers=3, epochs=15
)
print(f"Accuracy: {result['accuracy']:.2%}")
print(result['plot'])
```

## Implementing Your Own Version

The following skeleton reconstructs the PQC architecture and training loop from `QNNAlgorithm`.

```python
# Simplified reconstruction — mirrors QNNAlgorithm._build_vqc_layer() and training loop
import numpy as np
from unitarylab.core import Circuit, Register

def build_vqc_layer(gs: Circuit, x: np.ndarray, params: np.ndarray,
                    n_qubits: int, layers: int, n_features: int):
    """
    Build PQC in-place: for each layer,
      - encode one feature per qubit via Rx(x[i % n_features])
      - apply trainable Rz(params[l,i,0]) and Ry(params[l,i,1]) rotations
      - add CNOT entanglement chain qubit 0-1-..-(n-2)-(n-1)
    params shape: (layers, n_qubits, 3)
    """
    for l in range(layers):
        for i in range(n_qubits):
            gs.rx(float(x[i % n_features]), i)      # data encoding
            gs.rz(float(params[l, i, 0]), i)        # trainable
            gs.ry(float(params[l, i, 1]), i)        # trainable
        for i in range(n_qubits - 1):
            gs.cnot(i, i + 1)                        # entanglement

def qnn_forward(x: np.ndarray, params: np.ndarray,
                n_qubits: int, layers: int,
                n_features: int, backend: str = 'torch') -> np.ndarray:
    """Evaluate the PQC and return the output state probability vector."""
    reg = Register('q', n_qubits)
    gs  = Circuit(reg, backend=backend)
    build_vqc_layer(gs, x, params, n_qubits, layers, n_features)
    sv = gs.execute()  # returns complex state vector
    return np.abs(np.asarray(sv).flatten())**2  # measurement probabilities

def train_qnn_minimal(x_train: np.ndarray, y_train: np.ndarray,
                      n_qubits: int = 2, layers: int = 2, epochs: int = 20,
                      lr: float = 0.1) -> np.ndarray:
    """
    Simplified training loop (gradient-free, parameter shift not shown).
    Returns final parameter array.
    """
    n_features = x_train.shape[1]
    params = np.random.uniform(0, 2 * np.pi, (layers, n_qubits, 3))
    for epoch in range(epochs):
        # Here a real implementation uses Parameter Shift for gradients;
        # this skeleton is illustrative.
        probs = [qnn_forward(x_train[i], params, n_qubits, layers,
                             n_features) for i in range(len(x_train))]
        # ... update params via gradient descent ...
    return params
```

**Component roles**:
- `build_vqc_layer` — faithfully mirrors `_build_vqc_layer()` from `algorithm.py`: data-reuploading Rx encoding per qubit × layer, followed by trainable Rz+Ry rotation pair and a CNOT entanglement chain.
- `qnn_forward` — wraps the forward pass: builds a fresh `Circuit`, executes it to get a statevector, and returns the Born-rule measurement probability vector.
- A full training loop would use **Parameter Shift** (`params[l,i,0] ± π/2`) to compute exact analytical gradients; this is the same pattern as `QCBMAlgorithm`.

## Debugging Tips

1. **`n_classes > 2^n_qubits`**: Raises `ValueError`. Ensure `n_qubits >= ceil(log2(n_classes))`.
2. **`x_train` shape**: Must be 2D `(N, F)`. Pass `x_train.reshape(-1, 1)` for single-feature data.
3. **Training simplified**: The current forward pass uses a noise-decaying approximation, not true quantum backpropagation. For research-grade gradients, implement Parameter Shift manually.
4. **`layers` dimension**: Parameters shape is `(layers, n_qubits, 3)`. Check `algo.params.shape` after `run()`.
