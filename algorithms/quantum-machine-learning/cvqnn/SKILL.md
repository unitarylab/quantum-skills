---
name: cvqnn
description: Skill for understanding, using, and implementing the Continuous Variable Quantum Neural Network (CVQNN) for binary classification via the CVQNNAlgorithm class, CVSimulator, and CVClassifier.
---

# Continuous Variable Quantum Neural Network (CVQNN)

## Purpose

`CVQNNAlgorithm` implements binary classification using continuous variable (CV) quantum optics. Classical 2D features are encoded as coherent-state displacements in a truncated Fock space. Per-layer unitary evolution (squeezing, displacement, rotation, Kerr nonlinearity, beamsplitter) is parameterized by `torch.nn.Parameter` tensors and trained end-to-end with PyTorch's automatic differentiation. The observable is the $\hat{x}$ quadrature expectation value of mode 0.

Use this skill when you need to:
- Apply a CV quantum model to 2-feature binary classification tasks.
- Understand the structure of optical quantum gates in truncated Fock space.
- Run gradient-based training over quantum circuit parameters using PyTorch.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Normalize input features and encode each 2D sample as coherent-state displacements $D(x_0)$, $D(x_1)$ on two optical modes initialized in the vacuum state.
2. Apply $L$ variational layers. Each layer: beamsplitter entanglement across modes, then per-mode sequence of $K(\kappa)$, $D(\alpha)$, $S(r)$, $R(\theta)$.
3. Measure the $\hat{x}$ quadrature expectation value of mode 0 as the model output; train with MSE loss via PyTorch `autograd`.
4. Evaluate accuracy and generate a decision-boundary plot.
5. Return a result dictionary matching `_build_return_dict()`.

## Prerequisites

- Continuous variable (CV) quantum optics: coherent states, Fock space truncation.
- PyTorch `nn.Module` and `autograd`; Adam optimizer.
- `numpy`, `torch`, `matplotlib`, `unitarylab.Circuit`.

## Using the Provided Implementation

```python
import numpy as np
from sklearn.datasets import make_moons
from unitarylab_algorithms.quantum_machine_learning.cvqnn.algorithm import CVQNNAlgorithm

X, y = make_moons(n_samples=40, noise=0.1, random_state=42)

algo = CVQNNAlgorithm(text_mode="legacy")
result = algo.run(
    x_train=X,
    y_train=y,
    n_layers=2,
    cutoff=6,
    epochs=40,
    lr=0.05
)

print(f"Status: {result['status']}")
print(f"Final Accuracy: {result['Final Accuracy']:.2%}")
print(f"Final Loss:     {result['Final Loss']:.6f}")
print(f"Circuit saved:  {result['circuit_path']}")
for f in result['plot']:
    print(f"Plot saved:     {f['filename']}  (format: {f['format']})")
```

## Core Parameters Explained

### Constructor — `CVQNNAlgorithm`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output formatting mode passed to `BaseAlgorithm`. Use `"legacy"` for ASCII logging. |
| `algo_dir` | `str\|None` | `None` | Output directory. Auto-derived from `os.getcwd()/results/quantum-machine-learning/cvqnn` if `None`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x_train` | `np.ndarray` | required | Training features, shape `(N_samples, 2)`. Must be 2-feature input. |
| `y_train` | `np.ndarray` | required | Binary integer labels `{0, 1}`, shape `(N_samples,)`. |
| `n_layers` | `int` | `2` | Number of variational CV layers (`L`). |
| `cutoff` | `int` | `6` | Fock space truncation dimension for `CVSimulator`. |
| `epochs` | `int` | `40` | Adam optimizer training iterations. |
| `lr` | `float` | `0.05` | Adam learning rate. |

## Return Fields

The return value is built by `_build_return_dict(success, circuit_path, filepath, circuit)`, which merges base keys with `self.output`.

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` otherwise. |
| `circuit_path` | `str` | Absolute path to the saved topology circuit SVG. |
| `plot` | `List[Dict]` | List of `{"format": "svg", "filename": "<abs_path>/CVQNN_Metrics.svg"}` — training loss curve and decision boundary plot. |
| `circuit` | `Circuit` | The `Circuit` object returned by `_build_circuit()`. |
| `Final Loss` | `float` | MSE loss value at the last epoch. |
| `Final Accuracy` | `float` | Fraction of training samples correctly classified. |
| `Total Computation Time (s)` | `float` | Wall-clock time for the full `run()` call. |

## Implementation Architecture

`CVQNNAlgorithm` in `algorithm.py` is a `BaseAlgorithm` subclass. The actual quantum computation is performed by two inner classes:

- **`CVSimulator`** — Constructs Fock-space matrix representations of CV gates (annihilation `a`, creation `adag`, number `n_op`, position `x_op`) as `torch.complex128` tensors. Gate methods: `displacement(alpha)`, `squeezing(z)`, `rotation(theta)`, `kerr(kappa)` — all return unitary matrices via `torch.matrix_exp`.
- **`CVClassifier(nn.Module)`** — Holds all trainable parameters as `nn.Parameter` tensors: `sq_r (L×2)`, `disp_r (L×2)`, `rot_theta (L×2)`, `kerr_k (L×2)`, `bs_theta (L)`. Its `forward(x_batch)` iterates over the batch, encodes each sample, applies layers, and returns the $\hat{x}$ expectation on mode 0.

**`run(x_train, y_train, n_layers, cutoff, epochs, lr)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Preprocessing | z-score normalize `x_train`; convert to `torch.float64`; map labels to `{+1, -1}` via `torch.where` | Prepares tensors for PyTorch autograd |
| 2 — Circuit Topology | Calls `_build_circuit(n_layers)` → `Circuit(n_modes)` with `ry`/`cx`/`rx`/`ry`/`rz` placeholders | Topology diagram only; not used for computation |
| 3 — Training Loop | `CVClassifier.forward` + `nn.MSELoss` + `torch.optim.Adam`; real PyTorch `loss.backward()` and `optimizer.step()` each epoch | True gradient-based optimization over all `nn.Parameter` weights |
| 4 — Evaluation | `torch.where(final_preds > 0, 1.0, 0.0)` compared to original `y_tensor` | Computes training accuracy from thresholded quadrature output |
| 5 — Export | `_generate_metrics_plots(...)` saves `CVQNN_Metrics.svg`; `save_circuit(qc)` saves topology SVG | Produces all output files |

**Helper Methods:**

- **`_build_circuit(n_layers, n_modes=2)`** — Builds a `Circuit(n_modes)` topology: `ry` encoding layer per mode, then per layer `cx(0,1)` (beamsplitter) + `rx/ry/rz` variational gates per mode. Returns `Circuit` for diagram export only.
- **`_generate_metrics_plots(X_norm, y_train, model, loss_history, acc, algo_dir)`** — Saves a 2-panel SVG: training loss curve (`ax1`) and decision boundary contour (`ax2`) via `model` forward pass on a grid.

**Data flow:** `x_train` → normalize → `CVClassifier.forward` (repeated `epochs` times with Adam) → threshold predictions → accuracy → `_generate_metrics_plots` → `_build_return_dict`.

## Understanding the Key Quantum Components

### 1. Fock Space and CVSimulator
`CVSimulator(cutoff_dim, device)` truncates infinite-dimensional Fock space to dimension `cutoff`. The annihilation operator matrix is:
$$a_{n-1,n} = \sqrt{n}, \quad n = 1, \ldots, \text{cutoff}-1$$
All gate unitaries are computed via `torch.matrix_exp` over `torch.complex128` matrices.

### 2. Feature Encoding
Each input feature $x_i$ is encoded by displacing mode $i$ from vacuum:
$$|\psi_i\rangle = D(x_i)|0\rangle, \quad D(\alpha) = e^{\alpha a^\dagger - \alpha^* a}$$
The two-mode state is initialized as $|\psi_0\rangle \otimes |\psi_1\rangle$ via `torch.kron`.

### 3. Variational Layer Structure
Each of the $L$ layers (`CVClassifier.forward`, loop over `L`):
```
Beamsplitter:  U_bs = exp(bs_theta[L] * (adag0 @ a1 - a0 @ adag1))
Per-mode unit: K(kerr_k[L,m]) @ D(disp_r[L,m]) @ S(sq_r[L,m]) @ R(rot_theta[L,m])
```
All parameters (`sq_r`, `disp_r`, `rot_theta`, `kerr_k`, `bs_theta`) are `nn.Parameter` tensors trained by Adam.

### 4. Output Observable
After all layers, the $\hat{x}$ quadrature expectation of mode 0 is the model output:
$$\hat{x} = \langle\psi| (\hat{x}_{\text{op}} \otimes I) |\psi\rangle, \quad \hat{x}_{\text{op}} = \frac{a + a^\dagger}{\sqrt{2}}$$
A positive value predicts class 1; negative predicts class 0.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Vacuum state $|0\rangle$ | `CVSimulator.vacuum` — `torch.zeros((cutoff,1))[0]=1` |
| Displacement $D(\alpha)$ | `CVSimulator.displacement(alpha)` → `torch.matrix_exp(alpha*adag - conj(alpha)*a)` |
| Squeezing $S(r)$ | `CVSimulator.squeezing(z)` → `torch.matrix_exp(0.5*(conj(z)*a²  - z*adag²))` |
| Phase rotation $R(\theta)$ | `CVSimulator.rotation(theta)` → `torch.matrix_exp(-1j*theta*n_op)` |
| Kerr nonlinearity $K(\kappa)$ | `CVSimulator.kerr(kappa)` → `torch.matrix_exp(1j*kappa*n_op²)` |
| Beamsplitter $BS(\theta)$ | `torch.matrix_exp(bs_theta[L]*(adag0@a1 - a0@adag1))` inside `CVClassifier.forward` |
| Trainable parameters | `CVClassifier.sq_r`, `.disp_r`, `.rot_theta`, `.kerr_k`, `.bs_theta` as `nn.Parameter` |
| $\hat{x}$ observable | `CVSimulator.x_op = (a + adag) / sqrt(2)`; used in `curr_state.conj().T @ kron(x_op,I) @ curr_state` |
| MSE loss + Adam | `nn.MSELoss()` + `torch.optim.Adam(model.parameters(), lr=lr)` |
| Architecture topology | `_build_circuit(n_layers)` → `Circuit(n_modes)` with `ry/cx/rx/ry/rz` placeholders |

## Mathematical Deep Dive

The two-mode state after encoding is $|\psi^{(0)}\rangle = D(x_0)|0\rangle \otimes D(x_1)|0\rangle$.

Each variational layer $l$ applies:
$$|\psi^{(l)}\rangle = (U_0^{(l)} \otimes U_1^{(l)}) \cdot U_{\text{BS}}^{(l)} \cdot |\psi^{(l-1)}\rangle$$

where
$$U_m^{(l)} = K(\kappa_m^{(l)}) \cdot D(\alpha_m^{(l)}) \cdot S(r_m^{(l)}) \cdot R(\theta_m^{(l)})$$
$$U_{\text{BS}}^{(l)} = e^{\theta_{\text{bs}}^{(l)}(a_0^\dagger a_1 - a_0 a_1^\dagger)}$$

The scalar output is the $\hat{x}$ expectation on mode 0:
$$\hat{y} = \langle\psi^{(L)}| (\hat{x}_{\text{op}} \otimes I) |\psi^{(L)}\rangle \in \mathbb{R}$$

Training minimizes $\mathcal{L} = \frac{1}{N}\sum_i (\hat{y}_i - t_i)^2$ with $t_i \in \{+1,-1\}$ using Adam. All derivatives flow through `torch.matrix_exp` automatically.

**Truncation:** Fock space is capped at `cutoff` dimensions. Larger `cutoff` improves accuracy at quadratic memory cost ($\text{cutoff}^2 \times \text{cutoff}^2$ unitary matrices for two-mode operations).

## Hands-On Example

```python
import numpy as np
from sklearn.datasets import make_moons
from unitarylab_algorithms.quantum_machine_learning.cvqnn.algorithm import CVQNNAlgorithm

# Moon-shaped binary classification dataset
X, y = make_moons(n_samples=60, noise=0.15, random_state=0)

algo = CVQNNAlgorithm(text_mode="legacy")
result = algo.run(
    x_train=X,
    y_train=y,
    n_layers=3,
    cutoff=6,
    epochs=50,
    lr=0.03
)

print(f"Status:   {result['status']}")
print(f"Accuracy: {result['Final Accuracy']:.2%}")
print(f"Loss:     {result['Final Loss']:.6f}")
print(f"Time:     {result['Total Computation Time (s)']:.2f}s")
print(f"Circuit:  {result['circuit_path']}")
for p in result['plot']:
    print(f"Plot:     {p['filename']}")
```

## Implementing Your Own Version

The following skeleton reconstructs `CVSimulator` and the `CVClassifier` forward pass.

```python
import numpy as np
import torch
import torch.nn as nn

# --- Rebuild CVSimulator ---
cutoff = 6
a_data = np.zeros((cutoff, cutoff))
for n in range(1, cutoff):
    a_data[n-1, n] = np.sqrt(n)
a     = torch.tensor(a_data, dtype=torch.complex128)
adag  = a.T.conj()
x_op  = (a + adag) / np.sqrt(2)
n_op  = adag @ a
vacuum = torch.zeros((cutoff, 1), dtype=torch.complex128)
vacuum[0, 0] = 1.0 + 0j

def displacement(alpha):
    return torch.matrix_exp(alpha * adag - torch.conj(torch.as_tensor(alpha)) * a)

def squeezing(z):
    return torch.matrix_exp(0.5 * (torch.conj(torch.as_tensor(z)) * (a @ a)
                                   - torch.as_tensor(z) * (adag @ adag)))

def rotation(theta):
    return torch.matrix_exp(-1j * torch.as_tensor(theta) * n_op)

def kerr(kappa):
    return torch.matrix_exp(1j * torch.as_tensor(kappa) * (n_op @ n_op))

# --- Minimal CVClassifier-style forward pass for one sample ---
def cv_forward(x: torch.Tensor, sq_r, disp_r, rot_theta, kerr_k, bs_theta,
               n_layers: int) -> torch.Tensor:
    I = torch.eye(cutoff, dtype=torch.complex128)
    st0 = displacement(x[0]) @ vacuum
    st1 = displacement(x[1]) @ vacuum
    state = torch.kron(st0, st1)
    for L in range(n_layers):
        a0 = torch.kron(a, I); adag0 = a0.conj().T
        a1 = torch.kron(I, a); adag1 = a1.conj().T
        U_bs = torch.matrix_exp(bs_theta[L] * (adag0 @ a1 - a0 @ adag1))
        U0 = kerr(kerr_k[L,0]) @ displacement(disp_r[L,0]) @ squeezing(sq_r[L,0]) @ rotation(rot_theta[L,0])
        U1 = kerr(kerr_k[L,1]) @ displacement(disp_r[L,1]) @ squeezing(sq_r[L,1]) @ rotation(rot_theta[L,1])
        state = torch.kron(U0, U1) @ U_bs @ state
    x_exp = state.conj().T @ torch.kron(x_op, I) @ state
    return x_exp.real.squeeze()
```

**Component roles**:
- `displacement / squeezing / rotation / kerr` — directly mirror `CVSimulator` methods; each returns a `(cutoff × cutoff)` unitary via `torch.matrix_exp`.
- `cv_forward` — mirrors `CVClassifier.forward` for a single sample: encodes features as displacements, applies `n_layers` of beamsplitter + per-mode unitaries, returns the $\hat{x}$ expectation.
- To train, wrap `cv_forward` in `nn.Parameter` tensors and call `loss.backward()` with an Adam optimizer, exactly as in `CVClassifier`.

## Debugging Tips

1. **`x_train` must be 2-column**: `CVClassifier.forward` encodes exactly two features as `displacement(features[0])` and `displacement(features[1])`. Pass `x_train[:, :2]` if the dataset has more columns.
2. **Memory / speed with large `cutoff`**: Two-mode Kronecker products produce `(cutoff² × cutoff²)` matrices. Default `cutoff=6` gives `36×36`. Raising to `cutoff=10` gives `100×100` and is ~7× slower per forward pass.
3. **`complex128` gradient issues**: All `CVSimulator` tensors are `torch.complex128`. Ensure input features are cast with `torch.tensor(X, dtype=torch.float64)` before passing to `CVClassifier`.
4. **Loss not decreasing**: Try reducing `lr` (e.g. `0.01`) or increasing `n_layers`. Kerr nonlinearity (`kerr_k`) initialized near zero may limit expressivity in early epochs.
5. **Checking returned file paths**: `result['circuit_path']` is the topology SVG; `result['plot'][0]['filename']` is the metrics SVG (`CVQNN_Metrics.svg`). Both are absolute paths under `algo_dir`.
