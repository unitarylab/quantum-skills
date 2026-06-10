---
name: qcbm
description: Skill for understanding, using, and implementing the Quantum Circuit Born Machine (QCBM) for learning discrete probability distributions (Bars-and-Stripes) via the QCBMAlgorithm class.
---

# Quantum Circuit Born Machine (QCBM)

## Purpose

QCBM is an unsupervised generative model that uses the Born rule to map a parameterized quantum circuit's measurement outcomes to a probability distribution. It learns to match an arbitrary target distribution by minimizing KL divergence.

Use this skill when you need to:
- Learn and generate samples from a discrete probability distribution using a quantum circuit.
- Demonstrate generative quantum machine learning with Born-rule probabilities.

## Overview

1. Target: 2×2 Bars-and-Stripes (BAS) distribution over 4 bits (6 valid patterns out of 16).
2. Initialize variational parameters $\theta \in \mathbb{R}^{layers \times n}$.
3. For each epoch: compute Born-rule probabilities $p_\theta(x)$, compute KL divergence, compute Parameter Shift gradients.
4. Update $\theta$ with Adam optimizer.

## Prerequisites

- Born rule: measurement probabilities $p_\theta(x) = |\langle x|\psi(\theta)\rangle|^2$.
- KL divergence; Parameter Shift Rule.
- Adam optimizer.
- `torch`, `numpy`, `Circuit`.

## Using the Provided Implementation

```python
from unitarylab_algorithms import QCBMAlgorithm

algo = QCBMAlgorithm(text_mode="plain")
result = algo.run(
    n=4,
    layers=4,
    epochs=40,
    lr=0.1,
    backend='torch'
)

print(f"Final KL Loss: {result['Final KL Loss']:.4f}")
print(f"Circuit path: {result['circuit_path']}")
for f in result['plot']:
    print(f"Output file: {f['filename']} ({f['format']})")
```

## Core Parameters Explained

### Constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Output text rendering mode. |
| `algo_dir` | `str\|None` | `None` | Output directory for plots and circuit. Auto-derived from CWD if `None`. |

### `run()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `4` | Number of qubits. BAS target is fixed at 4 qubits. |
| `layers` | `int` | `4` | Number of variational layers. |
| `epochs` | `int` | `40` | Training epochs. |
| `lr` | `float` | `0.1` | Adam optimizer learning rate. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device. |
| `dtype` | `type` | `np.complex128` | State vector dtype. |

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` otherwise. |
| `circuit_path` | `str` | Path to saved circuit diagram (SVG). |
| `circuit` | `Circuit` | Ansatz `Circuit` object (visualization only). |
| `plot` | `List[Dict]` | List of output file dicts, each with `'format'` (file extension) and `'filename'` (absolute path). Contains 3 files: KL loss curve SVG, distribution comparison SVG, BAS samples SVG. |
| `Final KL Loss` | `float` | KL divergence value at the last training epoch. |
| `Quantum Computation Time` | `float` | Wall-clock seconds for the training loop. |

## Implementation Architecture

`QCBMAlgorithm` in `algorithm.py` trains a variational quantum circuit to learn a discrete probability distribution (2×2 Bars and Stripes) using KL divergence minimization and the Parameter Shift Rule.

**`run(n, layers, epochs, lr, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Initialization | `_get_bas_dist(n)` → `target_probs`; `theta = nn.Parameter(torch.rand((layers, n))*2π)`; Adam optimizer | BAS target distribution and random parameter init |
| 2 — Circuit Preview | `_build_circuit(theta.detach(), n)` | Visualization only — not used for training |
| 3 — Training Loop | Per epoch: `curr_probs = _get_probs(theta, ...)`; manual KL loss; parameter shift per `(l,q)`: `_get_probs(th_p/th_m, ...)`; `grad = 0.5*(p_p - p_m)`; `optimizer.step()` | Full quantum gradient-based KL minimization |
| 4 — Evaluation | `_get_probs(theta, n)` with final params → `final_probs` | Final distribution capture |
| 5 — Export | `_generate_all_outputs(n, target, final, history, algo_dir)` | Saves 3 SVG plots: KL loss curve, distribution comparison bar chart, BAS sample grid |

**Helper Methods:**
- **`_get_bas_dist(n)`** — Hardcoded BAS distribution: `valid = [0, 3, 5, 10, 12, 15]`; returns `(tensor, valid)` with uniform probability `1/6` over these 6 states and `0` elsewhere.
- **`_build_circuit(theta, n)`** — `Circuit(n)`; per layer $l$: `ry(theta[l,q], q)` for all qubits (RY gates); then ring `cx(q, (q+1)%n)` for all qubits (skipped in last layer). Returns a `Circuit` object.
- **`_get_probs(theta, n)`** — Calls `_build_circuit`, executes with `initial_state=|0⟩` using `self.backend`, `self.device`, `self.dtype`, reads `.state` from the result, converts `|ψ|²` to a torch tensor of length `2^n`.

**Data flow:** `_get_bas_dist()` → `target_probs` → training loop → `_get_probs()` (current + shift±) → KL gradient per `(l,q)` → Adam step → `final_probs` → `_generate_all_outputs()` → result dict.

## Understanding the Key Quantum Components

### 1. Born Rule Probability
The probability of measuring basis state $|x\rangle$:
$$p_\theta(x) = |\langle x|U(\theta)|0^{\otimes n}\rangle|^2$$
This is the squared amplitude — the uniquely quantum mechanism. All $2^n$ probabilities sum to 1 by unitarity.

### 2. Variational Circuit Ansatz
Each layer applies:
```
Per-qubit: Rz(θ[l,q])
Entanglement: CNOT(q, q+1 mod n_qubits)  [ring topology]
```
The ring entanglement generates long-range correlations needed to represent BAS patterns.

### 3. 2×2 Bars-and-Stripes Target Distribution
The BAS dataset encodes 2×2 binary images where columns or rows are uniformly ON or OFF:
- All-rows: `0000`, `1111` (all off / all on)
- Row patterns: `0011`, `1100`
- Column patterns: `0101`, `1010`

The target is uniform over these 6 patterns: $p_{\text{target}}(x) = 1/6$ for valid patterns, 0 otherwise.

### 4. KL Divergence Loss
$$\mathcal{L} = D_{\text{KL}}(p_{\text{target}} \| p_\theta) = \sum_x p_{\text{target}}(x) \log\frac{p_{\text{target}}(x)}{p_\theta(x) + \epsilon}$$

### 5. Parameter Shift Gradient
$$\frac{\partial \mathcal{L}}{\partial \theta_{l,q}} = \frac{1}{2}\left[\mathcal{L}(\theta_{l,q}+\pi/2) - \mathcal{L}(\theta_{l,q}-\pi/2)\right]$$

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Target distribution $\pi$ (BAS) | `_get_bas_dist()` — states `[0,3,5,10,12,15]`, each prob `1/6` |
| Variational circuit $U(\theta)$ | `_build_circuit(theta, n)` — RY layers + ring CNOT entanglement |
| Born rule probabilities $p_k = |\langle k|\psi\rangle|^2$ | `_get_probs()` → `torch.abs(psi)**2` |
| KL divergence $D_{\text{KL}}(\pi \| p_\theta)$ | `sum(target * log((target+ε)/(curr+ε)))` in training loop |
| Parameter shift rule $\partial_\theta \mathcal{L}$ | `grad = 0.5*(p_plus - p_minus)` for each `(l,q)` index |
| KL gradient wrt $\theta_{l,q}$ | `grad_theta[l,q] = sum(-(target/(curr+ε)) * grad_p)` |
| Adam optimizer update | `torch.optim.Adam([theta], lr=lr)` |
| BAS valid states | `[0(0000), 3(0011), 5(0101), 10(1010), 12(1100), 15(1111)]` |
| Note — gate type | Circuit uses `ry(theta[l,q], q)` (RY gates); the "Rz" reference in Key Quantum Components is a discrepancy — actual gate is RY |

## Mathematical Deep Dive

**State:** $|\psi(\theta)\rangle = U(\theta)|0^n\rangle = \prod_{l=1}^{L}[\text{CX-ring} \cdot \bigotimes_q R_z(\theta_{l,q})]|0^n\rangle$.

**Probabilities:** $\mathbf{p}_\theta = (p_\theta(0), \ldots, p_\theta(2^n-1))$ where $\sum_x p_\theta(x) = 1$.

**Information-theoretic convergence:** As $\mathcal{L} \to 0$, $p_\theta \to p_{\text{target}}$ in total variation distance.

## Hands-On Example

```python
from unitarylab_algorithms import QCBMAlgorithm

for text_mode in ["plain", "legacy"]:
    algo = QCBMAlgorithm(text_mode=text_mode)
    result = algo.run(n=4, layers=6, epochs=60, lr=0.08)
    final_loss = result['Final KL Loss']
    print(f"text_mode={text_mode}: final KL = {final_loss:.4f}")
    print(f"Outputs: {[f['filename'] for f in result['plot']]}")
```

## Implementing Your Own Version

The following skeleton reconstructs the QCBM circuit builder, Born-rule probability extraction, and Parameter Shift gradient loop.

```python
# Simplified reconstruction — mirrors QCBMAlgorithm._build_circuit(), _get_probs(), training loop
import numpy as np
import torch
from unitarylab import Circuit

def build_circuit(theta: torch.Tensor, n: int) -> Circuit:
    """Ry-layer + CNOT-ring architecture, layers = theta.shape[0]."""
    qc = Circuit(n)
    for l in range(theta.shape[0]):
        for q in range(n):
            qc.ry(float(theta[l, q]), q)
        if l < theta.shape[0] - 1:
            for q in range(n):
                qc.cx(q, (q + 1) % n)  # ring entanglement
    return qc

def get_probs(theta: torch.Tensor, n: int,
             backend: str = 'torch', device: str = 'cpu',
             dtype=np.complex128) -> torch.Tensor:
    """Execute circuit and return Born-rule probability vector (length 2^n)."""
    qc = build_circuit(theta, n)
    psi0 = np.zeros((2**n, 1), dtype=np.complex128)
    psi0[0, 0] = 1.0
    result = qc.execute(initial_state=psi0, backend=backend, device=device, dtype=dtype)
    amplitudes = np.asarray(result.state).flatten()
    return torch.as_tensor(np.abs(amplitudes)**2)

def train_qcbm(target_probs: torch.Tensor, n: int = 4,
               layers: int = 4, epochs: int = 40, lr: float = 0.1,
               backend: str = 'torch') -> torch.Tensor:
    """Full KL-divergence training loop with Parameter Shift gradients."""
    theta = torch.nn.Parameter(torch.rand((layers, n)) * 2 * np.pi)
    optimizer = torch.optim.Adam([theta], lr=lr)
    shift = np.pi / 2
    eps = 1e-12

    for ep in range(1, epochs + 1):
        curr_probs = get_probs(theta.detach(), n, backend)
        kl_loss = torch.sum(target_probs * torch.log((target_probs + eps) / (curr_probs + eps)))

        # Parameter Shift gradients
        grad = torch.zeros_like(theta)
        for l in range(layers):
            for q in range(n):
                th_p = theta.detach().clone(); th_p[l, q] += shift
                th_m = theta.detach().clone(); th_m[l, q] -= shift
                p_p = get_probs(th_p, n, backend)
                p_m = get_probs(th_m, n, backend)
                dp  = 0.5 * (p_p - p_m)  # gradient of prob w.r.t. theta[l,q]
                grad[l, q] = torch.sum(-(target_probs / (curr_probs + eps)) * dp)

        optimizer.zero_grad(); theta.grad = grad; optimizer.step()
    return theta.detach()
```

**Component roles**:
- `build_circuit` — faithfully mirrors `_build_circuit(theta, n)`: per-layer RY rotations on all qubits, followed by a CNOT ring (except on the last layer).
- `get_probs` — mirrors `_get_probs(theta, n)`: executes from $|0\rangle^{\otimes n}$, reads `.state` from the execution result, and returns $|\langle x|\psi(\theta)\rangle|^2$ for all $x$.
- `train_qcbm` — mirrors the training loop: KL divergence as the loss, Parameter Shift Rule for exact analytical gradients, Adam optimizer.

## Debugging Tips

1. **KL divergence not decreasing**: Increase `layers` (6+) or `epochs` (80+). For BAS with 4 qubits, at least 4 layers are needed.
2. **`n != 4`**: The BAS target distribution is hard-coded for 4 qubits. Changing `n` changes the circuit but the target distribution remains 4-qubit BAS.
3. **Slow training**: Parameter Shift requires $2 \times layers \times n$ circuit evaluations per epoch. For `layers=4, n=4`, this is 32 circuits per epoch.
4. **`lr` too large**: May cause KL to oscillate or diverge. Use `lr=0.05–0.15`.
5. **Numerical stability**: The KL divergence adds `eps=1e-12` to avoid `log(0)`. If probabilities collapse to 0 for some states, consider entropy regularization.
