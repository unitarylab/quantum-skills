"""Manual implementation of Continuous Variable Quantum Neural Network (CVQNN).

CVQNN implements binary classification using continuous variable quantum optics.
Classical 2D features are encoded as coherent-state displacements in a truncated
Fock space. Per-layer unitary evolution (squeezing, displacement, rotation, Kerr
nonlinearity, beamsplitter) is parameterized by trainable tensors and optimized
end-to-end with PyTorch's automatic differentiation.

Architecture:
    1. Encode each 2D sample as coherent-state displacements D(x0), D(x1)
       on two optical modes initialized in vacuum.
    2. Apply L variational layers: beamsplitter → per-mode K(κ)·D(α)·S(r)·R(θ).
    3. Measure the x̂ quadrature expectation of mode 0 as the model output.
    4. Train with MSE loss via Adam optimizer.
    5. Evaluate accuracy; generate decision-boundary + loss plot.

Fock space is truncated to `cutoff` dimensions. All gate unitaries are
computed via torch.matrix_exp over torch.complex128 matrices.

Components:
    - CVSimulator: Fock-space matrices and CV gate unitaries
    - CVClassifier: nn.Module with trainable CV parameters
    - cvqnn_train: End-to-end training pipeline
    - CVQNNAlgorithm: Class-based interface
    - plot_metrics: Decision boundary + loss curve visualization

Reference:
    SKILL.md — CVQNN
"""

from __future__ import annotations

import time
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from unitarylab.core import Circuit
except ImportError:
    Circuit = None

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ===================================================================
# 1. CVSimulator — Fock-space CV gate representations
# ===================================================================


class CVSimulator:
    """Continuous variable optical quantum simulator in truncated Fock space.

    Builds matrix representations of ladder operators (a, a†) and common
    CV gates via torch.matrix_exp. All matrices are torch.complex128.

    Args:
        cutoff_dim: Fock space truncation dimension. Larger = more accurate
            but quadratically more expensive (two-mode ops are cutoff²×cutoff²).
        device: PyTorch device ('cpu' or 'cuda').
    """

    def __init__(self, cutoff_dim: int = 6, device: str = "cpu"):
        self.cutoff = cutoff_dim
        self.device = device

        # Annihilation operator a: a[n-1, n] = sqrt(n)
        a_data = np.zeros((cutoff_dim, cutoff_dim))
        for n in range(1, cutoff_dim):
            a_data[n - 1, n] = np.sqrt(n)

        self.a = torch.tensor(a_data, dtype=torch.complex128).to(device)
        self.adag = self.a.T.conj()
        self.x_op = (self.a + self.adag) / np.sqrt(2)  # x̂ = (a+a†)/√2
        self.n_op = self.adag @ self.a  # n̂ = a†a

        # Vacuum state |0⟩
        self.vacuum = torch.zeros((cutoff_dim, 1), dtype=torch.complex128).to(device)
        self.vacuum[0, 0] = 1.0 + 0j

    # -- Gate unitaries via torch.matrix_exp --------------------------------

    def displacement(self, alpha: complex) -> torch.Tensor:
        """D(α) = exp(α·a† − α*·a).

        Args:
            alpha: Complex displacement amplitude.

        Returns:
            (cutoff × cutoff) unitary matrix.
        """
        alpha_t = torch.as_tensor(alpha, dtype=torch.complex128).to(self.device)
        return torch.matrix_exp(alpha_t * self.adag - torch.conj(alpha_t) * self.a)

    def squeezing(self, z: complex) -> torch.Tensor:
        """S(z) = exp(½(z*·a² − z·a†²)).

        Args:
            z: Complex squeezing parameter z = r·e^{iφ}.

        Returns:
            (cutoff × cutoff) unitary matrix.
        """
        z_t = torch.as_tensor(z, dtype=torch.complex128).to(self.device)
        return torch.matrix_exp(
            0.5 * (torch.conj(z_t) * (self.a @ self.a) - z_t * (self.adag @ self.adag))
        )

    def rotation(self, theta: float) -> torch.Tensor:
        """R(θ) = exp(−i·θ·n̂).

        Args:
            theta: Rotation angle (radians).

        Returns:
            (cutoff × cutoff) unitary matrix.
        """
        theta_t = torch.as_tensor(theta, dtype=torch.complex128).to(self.device)
        return torch.matrix_exp(-1j * theta_t * self.n_op)

    def kerr(self, kappa: float) -> torch.Tensor:
        """K(κ) = exp(i·κ·n̂²).

        Args:
            kappa: Kerr nonlinearity strength.

        Returns:
            (cutoff × cutoff) unitary matrix.
        """
        kappa_t = torch.as_tensor(kappa, dtype=torch.complex128).to(self.device)
        return torch.matrix_exp(1j * kappa_t * (self.n_op @ self.n_op))


# ===================================================================
# 2. CVClassifier — trainable nn.Module
# ===================================================================


class CVClassifier(nn.Module):
    """Continuous variable quantum classifier.

    Two-mode CV circuit with L variational layers. Each layer:
        - Beamsplitter (entangling two modes)
        - Per-mode sequence: K(κ) → D(α) → S(r) → R(θ)

    Trainable parameters (all nn.Parameter):
        sq_r:      (L × 2) squeezing magnitudes
        disp_r:    (L × 2) displacement amplitudes (real)
        rot_theta: (L × 2) rotation angles
        kerr_k:    (L × 2) Kerr nonlinearity strengths
        bs_theta:  (L,)    beamsplitter angles

    Args:
        n_layers: Number of variational layers L.
        cutoff: Fock space truncation dimension.
        device: PyTorch device.
    """

    def __init__(self, n_layers: int = 2, cutoff: int = 6, device: str = "cpu"):
        super().__init__()
        self.sim = CVSimulator(cutoff_dim=cutoff, device=device)
        self.n_layers = n_layers
        self.cutoff = cutoff

        # Trainable parameters with sensible initial ranges
        self.sq_r = nn.Parameter(torch.randn(n_layers, 2) * 0.1)
        self.disp_r = nn.Parameter(torch.randn(n_layers, 2) * 0.1)
        self.rot_theta = nn.Parameter(torch.rand(n_layers, 2) * 2 * np.pi)
        self.kerr_k = nn.Parameter(torch.randn(n_layers, 2) * 0.05)
        self.bs_theta = nn.Parameter(torch.rand(n_layers) * np.pi)

    def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode → layers → x̂ measurement.

        Args:
            x_batch: Input tensor of shape (batch_size, 2).

        Returns:
            Predicted outputs of shape (batch_size, 1). Positive → class 1,
            negative → class 0.
        """
        outputs = []
        I = torch.eye(self.cutoff, dtype=torch.complex128).to(self.sim.device)

        for i in range(x_batch.shape[0]):
            features = x_batch[i]

            # Encode: D(x0)|0⟩ ⊗ D(x1)|0⟩
            st0 = self.sim.displacement(features[0]) @ self.sim.vacuum
            st1 = self.sim.displacement(features[1]) @ self.sim.vacuum
            curr_state = torch.kron(st0, st1)

            # Variational layers
            for layer in range(self.n_layers):
                # Beamsplitter: exp(θ·(a0†a1 − a0 a1†))
                a0 = torch.kron(self.sim.a, I)
                adag0 = a0.conj().T
                a1 = torch.kron(I, self.sim.a)
                adag1 = a1.conj().T
                U_bs = torch.matrix_exp(
                    self.bs_theta[layer] * (adag0 @ a1 - a0 @ adag1)
                )

                # Per-mode unitaries: K(κ) → D(α) → S(r) → R(θ)
                U0 = (
                    self.sim.kerr(self.kerr_k[layer, 0])
                    @ self.sim.displacement(self.disp_r[layer, 0])
                    @ self.sim.squeezing(self.sq_r[layer, 0])
                    @ self.sim.rotation(self.rot_theta[layer, 0])
                )
                U1 = (
                    self.sim.kerr(self.kerr_k[layer, 1])
                    @ self.sim.displacement(self.disp_r[layer, 1])
                    @ self.sim.squeezing(self.sq_r[layer, 1])
                    @ self.sim.rotation(self.rot_theta[layer, 1])
                )

                curr_state = torch.kron(U0, U1) @ U_bs @ curr_state

            # Measurement: ⟨x̂ ⊗ I⟩ on mode 0
            x_exp = (
                curr_state.conj().T
                @ torch.kron(self.sim.x_op, I)
                @ curr_state
            )
            outputs.append(x_exp.real.squeeze())

        return torch.stack(outputs).view(-1, 1)


# ===================================================================
# 3. Data preprocessing
# ===================================================================


def preprocess_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Normalize features and convert labels to {+1, −1} target format.

    Args:
        x_train: Features of shape (N, 2).
        y_train: Binary labels {0, 1} of shape (N,).

    Returns:
        Tuple of (x_tensor, y_tensor, y_target, x_mean, x_std).
            - x_tensor: Normalized float64 tensor (N, 2).
            - y_tensor: Original labels as float64 tensor (N, 1).
            - y_target: Target values {+1, −1} for MSE loss (N, 1).
            - x_mean, x_std: Normalization statistics for later use.
    """
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)  # avoid div by zero
    X_norm = (x_train - x_mean) / x_std

    x_tensor = torch.tensor(X_norm, dtype=torch.float64)
    y_tensor = torch.tensor(y_train, dtype=torch.float64).view(-1, 1)
    y_target = torch.where(y_tensor > 0.5, 1.0, -1.0)

    return x_tensor, y_tensor, y_target, x_mean, x_std


# ===================================================================
# 4. Training pipeline
# ===================================================================


def cvqnn_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_layers: int = 2,
    cutoff: int = 6,
    epochs: int = 40,
    lr: float = 0.05,
    device: str = "cpu",
    verbose: bool = True,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a CVQNN for binary classification.

    Pipeline:
        1. Normalize features, convert labels to {+1, −1}.
        2. Initialize CVClassifier with random parameters.
        3. Train L epochs with Adam + MSE loss.
        4. Evaluate training accuracy via thresholded quadrature output.
        5. Generate decision-boundary + loss plot.

    Args:
        x_train: Training features of shape (N, 2).
        y_train: Binary integer labels {0, 1} of shape (N,).
        n_layers: Number of variational CV layers.
        cutoff: Fock space truncation dimension.
        epochs: Training epochs.
        lr: Adam learning rate.
        device: PyTorch device ('cpu' or 'cuda').
        verbose: Print progress.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - Final Loss: MSE loss at last epoch
            - Final Accuracy: Training accuracy
            - Total Computation Time (s): Wall-clock time
            - loss_history: List of per-epoch losses
            - model: Trained CVClassifier
            - X_norm: Normalized training features
            - y_train: Original labels

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If x_train is not 2-column.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for CVQNN. Install with: pip install torch"
        )

    if x_train.shape[1] != 2:
        raise ValueError(
            f"CVQNN requires exactly 2 input features, got {x_train.shape[1]}"
        )

    # Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)

    total_start = time.perf_counter()

    # --- Stage 1: Preprocessing ---
    x_tensor, y_tensor, y_target, x_mean, x_std = preprocess_data(x_train, y_train)

    if verbose:
        print(f"CVQNN Training")
        print(f"  Samples:  {x_train.shape[0]}")
        print(f"  Features: 2 (CV modes)")
        print(f"  Layers:   {n_layers}")
        print(f"  Cutoff:   {cutoff} (Fock space)")
        print(f"  Epochs:   {epochs}, lr={lr}")

    # --- Stage 2: Model initialization ---
    model = CVClassifier(n_layers=n_layers, cutoff=cutoff, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Params:   {n_params} trainable")

    # --- Stage 3: Training loop ---
    if verbose:
        print(f"  Training...")

    loss_history: List[float] = []
    train_start = time.perf_counter()

    for e in range(1, epochs + 1):
        optimizer.zero_grad()
        preds = model(x_tensor)
        loss = criterion(preds, y_target)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))

        if verbose and (e % 10 == 0 or e == 1 or e == epochs):
            print(f"    Epoch {e:>4}/{epochs} | Loss: {loss.item():.6f}")

    train_time = time.perf_counter() - train_start

    # --- Stage 4: Evaluation ---
    with torch.no_grad():
        final_preds = model(x_tensor)
        # threshold: > 0 → class 1, ≤ 0 → class 0
        pred_labels = torch.where(final_preds > 0, 1.0, 0.0)
        accuracy = float((pred_labels == y_tensor).float().mean().item())

    # --- Stage 5: Metrics plot ---
    X_norm = (x_train - x_mean) / x_std
    metrics_path = None
    if HAS_MATPLOTLIB:
        metrics_path = _plot_metrics(
            X_norm, y_train, model, loss_history, accuracy,
            output_path=os.path.join(output_dir or os.getcwd(), "CVQNN_Metrics.svg"),
        )

    total_time = time.perf_counter() - total_start

    if verbose:
        print(f"  Train time:      {train_time:.2f}s")
        print(f"  Final Loss:      {loss_history[-1]:.6f}")
        print(f"  Final Accuracy:  {accuracy:.2%}")
        print(f"  Total time:      {total_time:.2f}s")
        if metrics_path:
            print(f"  Metrics plot:    {metrics_path}")
        print(f"  Status:          ok")

    circuit = _build_cvqnn_circuit(n_layers)
    return {
        "status": "ok",
        "Final Loss": loss_history[-1],
        "Final Accuracy": accuracy,
        "Total Computation Time (s)": round(total_time, 2),
        "Training Time (s)": round(train_time, 2),
        "loss_history": loss_history,
        "model": model,
        "X_norm": X_norm,
        "y_train": y_train,
        "x_mean": x_mean,
        "x_std": x_std,
        "metrics_path": metrics_path,
        "n_params": n_params,
        "circuit": circuit,
    }


# ===================================================================
# 5. Visualization
# ===================================================================


def _plot_metrics(
    X_norm: np.ndarray,
    y_train: np.ndarray,
    model: CVClassifier,
    loss_history: List[float],
    accuracy: float,
    output_path: str = "CVQNN_Metrics.svg",
    grid_resolution: int = 20,
) -> str:
    """Generate a 2-panel plot: training loss + decision boundary.

    Args:
        X_norm: Normalized training features.
        y_train: Training labels.
        model: Trained CVClassifier.
        loss_history: Per-epoch loss values.
        accuracy: Final training accuracy.
        output_path: File path for the saved SVG.
        grid_resolution: Grid points per axis for decision boundary.

    Returns:
        Absolute path to the saved plot.
    """
    if not HAS_MATPLOTLIB:
        return ""

    import os

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(loss_history, color="#e74c3c", lw=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)

    # Decision boundary
    x_min, x_max = X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5
    y_min, y_max = X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    with torch.no_grad():
        pts = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64,
        )
        zz = model(pts).view(xx.shape).numpy()

    ax2.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.8)
    ax2.scatter(
        X_norm[:, 0], X_norm[:, 1], c=y_train,
        cmap="RdBu_r", edgecolors="k", s=40,
    )
    ax2.set_title(f"Decision Boundary (Acc: {accuracy:.2%})")
    ax2.set_xlabel("Feature 0 (normalized)")
    ax2.set_ylabel("Feature 1 (normalized)")

    plt.tight_layout()
    out = os.path.abspath(output_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_decision_boundary(
    model: CVClassifier,
    X_norm: np.ndarray,
    y_train: np.ndarray,
    output_path: str = "decision_boundary.png",
    grid_resolution: int = 30,
) -> str:
    """Standalone decision boundary plot.

    Args:
        model: Trained CVClassifier.
        X_norm: Normalized features.
        y_train: Labels.
        output_path: Output file path.
        grid_resolution: Grid resolution.

    Returns:
        Absolute path to saved plot.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    import os

    plt.figure(figsize=(7, 6))
    x_min, x_max = X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5
    y_min, y_max = X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    with torch.no_grad():
        pts = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64,
        )
        zz = model(pts).view(xx.shape).numpy()

    plt.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.8)
    plt.scatter(
        X_norm[:, 0], X_norm[:, 1], c=y_train,
        cmap="RdBu_r", edgecolors="k", s=40,
    )
    plt.title("CVQNN Decision Boundary")
    plt.xlabel("Feature 0 (normalized)")
    plt.ylabel("Feature 1 (normalized)")
    plt.tight_layout()

    out = os.path.abspath(output_path)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ===================================================================
# 6. Single forward pass (minimal usage)
# ===================================================================


def cvqnn_forward(
    x: np.ndarray,
    sq_r: np.ndarray,
    disp_r: np.ndarray,
    rot_theta: np.ndarray,
    kerr_k: np.ndarray,
    bs_theta: np.ndarray,
    cutoff: int = 6,
) -> float:
    """Minimal single-sample CVQNN forward pass.

    This implements the core quantum computation without PyTorch autograd
    dependency — useful for understanding the algorithm or for inference
    with pre-trained parameters.

    Args:
        x: 2-element feature vector.
        sq_r: Squeezing magnitudes, shape (L, 2).
        disp_r: Displacement amplitudes, shape (L, 2).
        rot_theta: Rotation angles, shape (L, 2).
        kerr_k: Kerr strengths, shape (L, 2).
        bs_theta: Beamsplitter angles, shape (L,).
        cutoff: Fock space truncation.

    Returns:
        Scalar x̂ quadrature expectation on mode 0.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required")

    sim = CVSimulator(cutoff_dim=cutoff)
    I = torch.eye(cutoff, dtype=torch.complex128)
    n_layers = len(bs_theta)

    # Encode
    st0 = sim.displacement(x[0]) @ sim.vacuum
    st1 = sim.displacement(x[1]) @ sim.vacuum
    state = torch.kron(st0, st1)

    for layer in range(n_layers):
        # Beamsplitter
        a0 = torch.kron(sim.a, I)
        adag0 = a0.conj().T
        a1 = torch.kron(I, sim.a)
        adag1 = a1.conj().T
        U_bs = torch.matrix_exp(
            bs_theta[layer] * (adag0 @ a1 - a0 @ adag1)
        )
        # Per-mode unitaries
        U0 = (
            sim.kerr(kerr_k[layer, 0])
            @ sim.displacement(disp_r[layer, 0])
            @ sim.squeezing(sq_r[layer, 0])
            @ sim.rotation(rot_theta[layer, 0])
        )
        U1 = (
            sim.kerr(kerr_k[layer, 1])
            @ sim.displacement(disp_r[layer, 1])
            @ sim.squeezing(sq_r[layer, 1])
            @ sim.rotation(rot_theta[layer, 1])
        )
        state = torch.kron(U0, U1) @ U_bs @ state

    # Measurement
    x_exp = state.conj().T @ torch.kron(sim.x_op, I) @ state
    return float(x_exp.real.squeeze().item())


# ===================================================================
# 7. Class-based interface
# ===================================================================


def _build_cvqnn_circuit(n_layers: int):
    """Build the topology used by the source CVQNN algorithm."""
    if Circuit is None:
        raise ImportError("unitarylab.core.Circuit is required for CVQNN export")
    qc = Circuit(2)
    for mode in (0, 1):
        qc.ry(0.0, mode)
    for _ in range(n_layers):
        qc.cx(0, 1)
        for mode in (0, 1):
            qc.rx(0.0, mode)
            qc.ry(0.0, mode)
            qc.rz(0.0, mode)
    return qc


class CVQNNAlgorithm:
    """Class-based solver for CVQNN binary classification.

    Usage:
        solver = CVQNNAlgorithm()
        result = solver.run(
            x_train=X, y_train=y, n_layers=3, cutoff=6, epochs=50,
        )
        print(result['Final Accuracy'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir or os.path.join(
            os.getcwd(), "results", "quantum-machine-learning", "cvqnn"
        )
        os.makedirs(self.algo_dir, exist_ok=True)

    def run(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        n_layers: int = 2,
        cutoff: int = 6,
        epochs: int = 40,
        lr: float = 0.05,
    ) -> Dict[str, Any]:
        """Train CVQNN. See cvqnn_train() for docs."""
        result = cvqnn_train(
            x_train=x_train, y_train=y_train,
            n_layers=n_layers, cutoff=cutoff,
            epochs=epochs, lr=lr, device="cpu", output_dir=self.algo_dir,
            verbose=(self.text_mode != "plain"),
        )
        circuit = result.get("circuit")
        if circuit is None:
            raise ImportError("CVQNN training did not produce a Circuit")
        circuit_path = os.path.abspath(
            os.path.join(self.algo_dir, "cvqnn_algorithm_circuit.svg")
        )
        circuit.draw(filename=circuit_path, title="CVQNN Algorithm Circuit")
        result["circuit_path"] = circuit_path
        return self._build_return_dict(result)

    def _build_circuit(self, n_layers: int):
        """Build the documented two-mode CVQNN topology when Circuit is available."""
        if Circuit is None:
            return None
        return _build_cvqnn_circuit(n_layers)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "Final Loss": result.get("Final Loss"),
            "Final Accuracy": result.get("Final Accuracy"),
            "Total Computation Time (s)": result.get("Total Computation Time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": (
                [{"format": "svg", "filename": result["metrics_path"]}]
                if result.get("metrics_path")
                else []
            ),
            "circuit": result.get("circuit"),
        }


# ===================================================================
# 8. Synthetic data generators
# ===================================================================


def make_moons_dataset(
    n_samples: int = 60,
    noise: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a two-moons binary classification dataset.

    Uses sklearn if available, otherwise falls back to manual construction.

    Args:
        n_samples: Total number of samples.
        noise: Standard deviation of Gaussian noise.
        seed: Random seed.

    Returns:
        Tuple of (X, y) — features (n_samples, 2) and labels (n_samples,).
    """
    try:
        from sklearn.datasets import make_moons

        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        return X, y
    except ImportError:
        # Manual fallback
        rng = np.random.default_rng(seed)
        n_per_class = n_samples // 2
        # Class 0: crescent around (-0.5, 0.5)
        theta0 = rng.uniform(0, np.pi, n_per_class)
        r0 = 1.0 + rng.normal(0, noise, n_per_class)
        x0 = np.column_stack([r0 * np.cos(theta0) - 0.5, r0 * np.sin(theta0)])
        # Class 1: crescent around (0.5, -0.3)
        theta1 = rng.uniform(np.pi, 2 * np.pi, n_per_class)
        r1 = 1.0 + rng.normal(0, noise, n_per_class)
        x1 = np.column_stack([r1 * np.cos(theta1) + 0.5, r1 * np.sin(theta1) + 0.5])
        X = np.vstack([x0, x1])
        y = np.array([0] * n_per_class + [1] * n_per_class)
        return X, y


def make_circles_dataset(
    n_samples: int = 60,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate concentric circles binary classification dataset.

    Args:
        n_samples: Total number of samples.
        noise: Standard deviation of Gaussian noise on radius.
        seed: Random seed.

    Returns:
        Tuple of (X, y).
    """
    rng = np.random.default_rng(seed)
    n_per = n_samples // 2
    # Inner circle (class 1)
    theta_in = rng.uniform(0, 2 * np.pi, n_per)
    r_in = 0.5 + rng.normal(0, noise, n_per)
    x_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    # Outer ring (class 0)
    theta_out = rng.uniform(0, 2 * np.pi, n_per)
    r_out = 1.5 + rng.normal(0, noise, n_per)
    x_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    X = np.vstack([x_in, x_out])
    y = np.array([1] * n_per + [0] * n_per)
    return X, y


# ===================================================================
# 9. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "moons_n40_L2_c6_e30",
        "dataset": "moons", "n_samples": 40, "n_layers": 2,
        "cutoff": 6, "epochs": 30, "lr": 0.05,
        "min_acc": 0.60,
    },
    {
        "name": "moons_n40_L3_c6_e50",
        "dataset": "moons", "n_samples": 40, "n_layers": 3,
        "cutoff": 6, "epochs": 50, "lr": 0.03,
        "min_acc": 0.65,
    },
    {
        "name": "circles_n40_L2_c6_e40",
        "dataset": "circles", "n_samples": 40, "n_layers": 2,
        "cutoff": 6, "epochs": 40, "lr": 0.05,
        "min_acc": 0.55,
    },
    {
        "name": "moons_n30_L1_c4_e20",
        "dataset": "moons", "n_samples": 30, "n_layers": 1,
        "cutoff": 4, "epochs": 20, "lr": 0.05,
        "min_acc": 0.55,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case."""
    name = case["name"]
    dataset = case["dataset"]
    n_samples = case["n_samples"]

    if dataset == "moons":
        X, y = make_moons_dataset(n_samples=n_samples, seed=42)
    else:
        X, y = make_circles_dataset(n_samples=n_samples, seed=42)

    result = cvqnn_train(
        x_train=X, y_train=y,
        n_layers=case["n_layers"], cutoff=case["cutoff"],
        epochs=case["epochs"], lr=case["lr"],
        verbose=False, seed=42,
    )

    acc = result["Final Accuracy"]
    ok = acc >= case["min_acc"]
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {name}: acc={acc:.2%} (min={case['min_acc']:.0%}), "
          f"loss={result['Final Loss']:.4f}, time={result['Total Computation Time (s)']:.1f}s")
    return ok


# ===================================================================
# 10. Main
# ===================================================================


def main() -> None:
    """Run the complete CVQNN demonstration pipeline."""
    if not HAS_TORCH:
        print("PyTorch is not installed. Install with: pip install torch")
        return

    print("=" * 60)
    print("CVQNN — Continuous Variable Quantum Neural Network")
    print("=" * 60)

    solver = CVQNNAlgorithm()

    # --- Demo 1: Two moons (basic) ---
    print("\n--- Demo 1: Two Moons (n=40, L=2, epochs=40) ---")
    X1, y1 = make_moons_dataset(n_samples=40, noise=0.1, seed=42)
    result1 = solver.run(
        x_train=X1, y_train=y1, n_layers=2, cutoff=6, epochs=40, lr=0.05,
    )
    print(f"  Accuracy: {result1['Final Accuracy']:.2%}")
    print(f"  Loss:     {result1['Final Loss']:.6f}")

    # --- Demo 2: Two moons (deeper) ---
    print("\n--- Demo 2: Two Moons (n=60, L=3, epochs=60) ---")
    X2, y2 = make_moons_dataset(n_samples=60, noise=0.15, seed=0)
    result2 = solver.run(
        x_train=X2, y_train=y2, n_layers=3, cutoff=6, epochs=60, lr=0.03,
    )
    print(f"  Accuracy: {result2['Final Accuracy']:.2%}")
    print(f"  Loss:     {result2['Final Loss']:.6f}")

    # --- Demo 3: Concentric circles ---
    print("\n--- Demo 3: Concentric Circles (n=40, L=2, epochs=50) ---")
    X3, y3 = make_circles_dataset(n_samples=40, noise=0.1, seed=42)
    result3 = solver.run(
        x_train=X3, y_train=y3, n_layers=2, cutoff=6, epochs=50, lr=0.05,
    )
    print(f"  Accuracy: {result3['Final Accuracy']:.2%}")
    print(f"  Loss:     {result3['Final Loss']:.6f}")

    # --- Demo 4: Effect of cutoff ---
    print("\n--- Demo 4: Effect of Fock cutoff dimension ---")
    X4, y4 = make_moons_dataset(n_samples=30, noise=0.1, seed=42)
    for c in [3, 4, 5, 6, 8]:
        result4 = cvqnn_train(
            x_train=X4, y_train=y4, n_layers=2, cutoff=c,
            epochs=30, lr=0.05, verbose=False, seed=42,
        )
        print(f"  cutoff={c}: acc={result4['Final Accuracy']:.2%}, "
              f"loss={result4['Final Loss']:.4f}, "
              f"time={result4['Total Computation Time (s)']:.1f}s")

    # --- Demo 5: Decision boundary visualization ---
    print("\n--- Demo 5: Decision Boundary Plot ---")
    X5, y5 = make_moons_dataset(n_samples=40, noise=0.1, seed=42)
    result5 = cvqnn_train(
        x_train=X5, y_train=y5, n_layers=3, cutoff=6,
        epochs=50, lr=0.03, verbose=False, seed=42,
    )
    if result5["metrics_path"]:
        print(f"  Plot saved: {result5['metrics_path']}")

    # --- Demo 6: Minimal forward pass (no autograd) ---
    print("\n--- Demo 6: Minimal Forward Pass (pre-trained params) ---")
    model = result5["model"]
    with torch.no_grad():
        sq_r_np = model.sq_r.numpy()
        disp_r_np = model.disp_r.numpy()
        rot_np = model.rot_theta.numpy()
        kerr_np = model.kerr_k.numpy()
        bs_np = model.bs_theta.numpy()
    # Run forward on a few samples
    for i in range(3):
        x_val = result5["X_norm"][i]
        y_val = cvqnn_forward(
            x_val, sq_r_np, disp_r_np, rot_np, kerr_np, bs_np, cutoff=6,
        )
        print(f"  Sample {i}: x̂_expectation={y_val:.4f}, "
              f"pred={'1' if y_val > 0 else '0'}, true={y5[i]}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
