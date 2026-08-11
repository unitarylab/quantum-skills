"""Manual implementation of Variational Quantum Classifier (VQC).

VQC applies a parameterized quantum circuit to supervised classification.
This implementation classifies the Iris dataset (4 features, 3 classes)
using data re-uploading and Parameter Shift Rule gradients.

Architecture:
    1. Load Iris dataset, normalize features to [−π/2, π/2].
    2. Encode 4 features as Ry(x_q) on 4 qubits (once at circuit start).
    3. Apply L variational layers: Ry(θ[q,l]) + ring CNOT (skip last layer).
    4. Measure ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_3⟩ as 3-class logits.
    5. CrossEntropyLoss(10×logits, y); Parameter Shift gradients; Adam.

Cost per epoch: 2 × n_qubits × layers × num_batches circuit evaluations.

Components:
    - load_iris_data: Load and preprocess Iris dataset
    - get_pauli_z_observable: Build Z_k ⊗ I_rest observable
    - build_vqc_circuit: Encoding + variational layers
    - get_logits: Compute ⟨Z_k⟩ expectations as class logits
    - vqc_train: Full Parameter Shift training pipeline
    - VQCClassifier: Class-based interface
    - plot_vqc_metrics: Loss + accuracy curves

Reference:
    SKILL.md — VQC (Variational Quantum Classifier)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

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

from unitarylab.core import Circuit


# ===================================================================
# 1. Iris data loading
# ===================================================================


def load_iris_data(
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Load and preprocess the Iris dataset for VQC.

    - StandardScaler normalization
    - Feature mapping to [−π/2, π/2]
    - Stratified 80/20 train/test split

    Uses sklearn if available; falls back to hardcoded dataset arrays.

    Args:
        test_size: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test) as torch tensors.
    """
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        iris = load_iris()
        X = StandardScaler().fit_transform(iris["data"])
        # Map to [−π/2, π/2]
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12) * np.pi
        X = X - np.pi / 2

        xt, xv, yt, yv = train_test_split(
            X, iris["target"], test_size=test_size,
            stratify=iris["target"], random_state=seed,
        )
    except ImportError:
        raise ImportError(
            "scikit-learn is required to load the Iris dataset. "
            "Install with: pip install scikit-learn"
        )

    return (
        torch.tensor(xt, dtype=torch.float64),
        torch.tensor(yt, dtype=torch.long),
        torch.tensor(xv, dtype=torch.float64),
        torch.tensor(yv, dtype=torch.long),
    )


# ===================================================================
# 2. Pauli-Z observables
# ===================================================================


def get_pauli_z_observable(
    qubit_idx: int,
    n_qubits: int = 4,
) -> "torch.Tensor":
    """Build Z_k ⊗ I_rest as a (2^n × 2^n) complex matrix.

    Args:
        qubit_idx: Index of the qubit to measure (0-indexed).
        n_qubits: Total number of qubits.

    Returns:
        Observable matrix as torch.complex128 tensor.
    """
    I = torch.eye(2, dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    op = Z if qubit_idx == 0 else I
    for i in range(1, n_qubits):
        op = torch.kron(op, Z if i == qubit_idx else I)
    return op


def get_observables(n_classes: int = 3, n_qubits: int = 4) -> List["torch.Tensor"]:
    """Build the list of Pauli-Z observables for each output class.

    Measures qubits 1, 2, 3 (indices 1, 2, 3) in Z basis.
    Qubit 0 is not measured (used as reference).

    Args:
        n_classes: Number of output classes (= Iris classes).
        n_qubits: Number of qubits.

    Returns:
        List of n_classes observable matrices.
    """
    return [get_pauli_z_observable(i, n_qubits) for i in range(1, n_classes + 1)]


# ===================================================================
# 3. VQC circuit builder
# ===================================================================


def build_vqc_circuit(
    x: np.ndarray,
    theta: "torch.Tensor",
) -> Circuit:
    """Build the VQC circuit for a single sample.

    Architecture (n_qubits=4, L=theta.shape[1]):
        1. Encoding: ry(x[q], q) for all 4 qubits (once).
        2. For each layer l = 0..L-1:
           a. Variational: ry(theta[q,l], q) for all qubits.
           b. Entanglement: ring CX(q, (q+1)%4) — skip if l == L-1.

    Args:
        x: Feature vector of length n_qubits.
        theta: Parameter tensor of shape (n_qubits, layers).

    Returns:
        Circuit object.
    """
    n_qubits = theta.shape[0]
    n_layers = theta.shape[1]
    qc = Circuit(n_qubits, name=f"VQC_L{n_layers}")

    # Data encoding (once)
    for q in range(n_qubits):
        qc.ry(float(x[q]), q)

    # Variational layers
    for l in range(n_layers):
        # Trainable RY rotations
        for q in range(n_qubits):
            qc.ry(float(theta[q, l]), q)

        # Ring CNOT entanglement (skip last layer)
        if l < n_layers - 1:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)

    return qc


# ===================================================================
# 4. Logit computation
# ===================================================================


def get_logits(
    x: np.ndarray,
    theta: "torch.Tensor",
    observables: List["torch.Tensor"],
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> "torch.Tensor":
    """Compute Pauli-Z expectation values as classification logits.

    For a single sample x:
        logits[k] = ⟨ψ(x,θ)| Z_{k+1} |ψ(x,θ)⟩

    Args:
        x: Single sample feature vector.
        theta: Parameter tensor (n_qubits, layers).
        observables: List of Z_k observables (one per class).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Logit tensor of shape (n_classes,).
    """
    n_qubits = theta.shape[0]
    qc = build_vqc_circuit(x, theta)

    # Execute with |0⟩^n initial state
    state0 = np.zeros(1 << n_qubits, dtype=dtype)
    state0[0] = 1.0
    psi = torch.as_tensor(
        qc.execute(
            initial_state=state0, backend=backend, device=device, dtype=dtype,
        ).state,
    ).to(torch.complex128)

    bra = psi.conj()
    logits = [(bra @ op @ psi).real for op in observables]
    return torch.tensor(logits, dtype=torch.float64)


def get_batch_logits(
    x_batch: "torch.Tensor",
    theta: "torch.Tensor",
    observables: List["torch.Tensor"],
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> "torch.Tensor":
    """Compute logits for a batch of samples.

    Args:
        x_batch: Feature tensor (batch_size, n_qubits).
        theta: Parameter tensor.
        observables: Z_k observables.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Logit tensor (batch_size, n_classes).
    """
    all_logits = []
    for x in x_batch:
        logits = get_logits(
            x.numpy(), theta, observables, backend, device, dtype,
        )
        all_logits.append(logits)
    return torch.stack(all_logits)


# ===================================================================
# 5. Evaluation
# ===================================================================


@torch.no_grad()
def evaluate_accuracy(
    x_test: "torch.Tensor",
    y_test: "torch.Tensor",
    theta: "torch.Tensor",
    observables: List["torch.Tensor"],
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> float:
    """Compute test accuracy.

    Args:
        x_test: Test features.
        y_test: Test labels.
        theta: Current parameters.
        observables: Z_k observables.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Accuracy as float in [0, 1].
    """
    logits = get_batch_logits(x_test, theta, observables, backend, device, dtype)
    preds = torch.argmax(logits, dim=1)
    return float((preds == y_test).float().mean().item())


# ===================================================================
# 6. Training pipeline
# ===================================================================


def vqc_train(
    layers: int = 3,
    epochs: int = 20,
    lr: float = 0.05,
    batch_size: int = 16,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a VQC on the Iris dataset.

    Pipeline:
        1. Load and preprocess Iris data.
        2. Initialize parameters θ and Adam optimizer.
        3. For each epoch, per batch:
           a. Compute logits → loss.
           b. Parameter shift gradient for each θ[q,l].
           c. Adam step.
        4. Evaluate test accuracy each epoch.
        5. Return trained model and metrics.

    Training cost per epoch:
        2 × n_qubits × layers × (n_train / batch_size) circuit evals.

    For default (4 qubits, 3 layers, 120 train samples, batch 16):
        ≈ 2 × 4 × 3 × 8 = 192 circuit evals per epoch.

    Args:
        layers: Number of variational layers (>= 2).
        epochs: Training epochs (20-50 typical).
        lr: Adam learning rate (0.01-0.1).
        batch_size: Mini-batch size.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - Final Loss: Cross-entropy at last epoch
            - Final Accuracy: Test accuracy at last epoch
            - loss_history: Per-epoch loss list
            - acc_history: Per-epoch accuracy list
            - Quantal Computation Time (s): Training time
            - theta: Final trained parameters
            - layers, epochs: Config values

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If layers < 1.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)

    # --- Stage 1: Data loading ---
    x_train, y_train, x_test, y_test = load_iris_data(seed=seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
    )

    n_qubits = 4  # Fixed: 4 Iris features
    n_classes = 3  # Fixed: 3 Iris classes

    # Initialize parameters θ ∈ [0, 2π)
    theta = torch.nn.Parameter(torch.rand((n_qubits, layers)) * 2 * np.pi)
    optimizer = torch.optim.Adam([theta], lr=lr)
    criterion = nn.CrossEntropyLoss()
    shift = np.pi / 2

    if verbose:
        print(f"VQC Training (Iris Classification)")
        print(f"  Dataset:      {len(x_train)} train / {len(x_test)} test")
        print(f"  Features:     {n_qubits} (→ {n_qubits} qubits)")
        print(f"  Classes:      {n_classes}")
        print(f"  Layers:       {layers}")
        print(f"  Epochs:       {epochs}, lr={lr}, batch={batch_size}")
        print(f"  Params:       {n_qubits * layers}")
        print(f"  Shifts/epoch: {2 * n_qubits * layers * (len(x_train) // batch_size)}")

    # --- Stage 2: Observables ---
    observables = get_observables(n_classes, n_qubits)

    # Preview circuit
    qc_preview = build_vqc_circuit(x_train[0].numpy(), theta.detach())

    # --- Stage 3: Training loop ---
    if verbose:
        print(f"  Training...")

    loss_history: List[float] = []
    acc_history: List[float] = []
    train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        epoch_loss = 0.0

        for xb, yb in train_loader:
            theta_base = theta.detach().clone()
            grad = torch.zeros_like(theta_base)

            # Parameter shift for each parameter
            for q in range(n_qubits):
                for l in range(layers):
                    # Shifted parameters
                    th_p = theta_base.clone()
                    th_p[q, l] += shift
                    th_m = theta_base.clone()
                    th_m[q, l] -= shift

                    # Loss at shifted points (10× scale for convergence)
                    logits_p = get_batch_logits(
                        xb, th_p, observables, backend, device, dtype,
                    )
                    logits_m = get_batch_logits(
                        xb, th_m, observables, backend, device, dtype,
                    )
                    loss_p = criterion(10 * logits_p, yb)
                    loss_m = criterion(10 * logits_m, yb)

                    # Gradient: ∂L/∂θ = ½[L(θ+π/2) − L(θ-π/2)]
                    grad[q, l] = (loss_p - loss_m) * 0.5

            optimizer.zero_grad()
            theta.grad = grad
            optimizer.step()

            # Accumulate loss
            with torch.no_grad():
                logits_cur = get_batch_logits(
                    xb, theta, observables, backend, device, dtype,
                )
                epoch_loss += criterion(10 * logits_cur, yb).item() * xb.size(0)

        avg_loss = epoch_loss / len(x_train)
        loss_history.append(avg_loss)
        test_acc = evaluate_accuracy(
            x_test, y_test, theta, observables, backend, device, dtype,
        )
        acc_history.append(test_acc)

        if verbose and (ep % 5 == 0 or ep == 1 or ep == epochs):
            print(f"    Epoch {ep:>3}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Test Acc: {test_acc:.2%}")

    train_time = time.perf_counter() - train_start

    final_loss = loss_history[-1]
    final_acc = acc_history[-1]

    if verbose:
        print(f"  Training time:  {train_time:.2f}s")
        print(f"  Final Loss:     {final_loss:.4f}")
        print(f"  Final Accuracy: {final_acc:.2%}")
        print(f"  Status:         ok")

    return {
        "status": "ok",
        "Final Loss": final_loss,
        "Final Accuracy": final_acc,
        "loss_history": loss_history,
        "acc_history": acc_history,
        "Quantal Computation Time (s)": round(train_time, 2),
        "theta": theta.detach(),
        "layers": layers,
        "epochs": epochs,
        "circuit": qc_preview,
        "circuit_path": "",
        "plot": [],
    }


# ===================================================================
# 7. Visualization
# ===================================================================


def plot_vqc_metrics(
    loss_history: List[float],
    acc_history: List[float],
    output_path: str = "VQC_Metrics.svg",
) -> str:
    """Generate dual-axis loss + accuracy plot.

    Args:
        loss_history: Per-epoch loss values.
        acc_history: Per-epoch accuracy values.
        output_path: Output file path.

    Returns:
        Absolute path to saved plot.
    """
    if not HAS_MATPLOTLIB:
        return ""

    import os

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(loss_history, color="#e74c3c", lw=2, label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss", color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")

    ax2 = ax1.twinx()
    ax2.plot(acc_history, color="#2ecc71", lw=2, label="Accuracy")
    ax2.set_ylabel("Test Accuracy", color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")
    ax2.set_ylim(0, 1.05)

    plt.title("VQC Training Progress")
    fig.tight_layout()

    out = os.path.abspath(output_path)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ===================================================================
# 8. Class-based interface
# ===================================================================


class VQCClassifier:
    """Class-based VQC classifier.

    Usage:
        clf = VQCClassifier()
        result = clf.run(layers=3, epochs=20, lr=0.05)
        print(result['Final Accuracy'])

        # Predict on new data
        preds = clf.predict(x_new)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self._theta: Optional["torch.Tensor"] = None
        self._observables: Optional[List["torch.Tensor"]] = None

    def run(
        self,
        layers: int = 3,
        epochs: int = 20,
        lr: float = 0.05,
        batch_size: int = 16,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Train VQC. See vqc_train() for docs."""
        result = vqc_train(
            layers=layers, epochs=epochs, lr=lr, batch_size=batch_size,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self._theta = result["theta"]
        n_classes = 3
        self._observables = get_observables(n_classes, 4)

        # Generate metrics plot
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            plot_path = plot_vqc_metrics(
                result["loss_history"], result["acc_history"],
                os.path.join(self.algo_dir, "VQC_Metrics.svg"),
            )
            result["plot"] = [{"format": "svg", "filename": plot_path}]

        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Final Loss": result.get("Final Loss"),
            "Final Accuracy": result.get("Final Accuracy"),
            "Quantal Computation Time (s)": result.get(
                "Quantal Computation Time (s)", 0.0,
            ),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }

    def predict(
        self,
        x: np.ndarray,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> np.ndarray:
        """Predict class labels for new samples.

        Args:
            x: Feature array of shape (N, 4).
            backend: Simulation backend.
            device: Compute device.
            dtype: Numerical dtype.

        Returns:
            Predicted class labels (0, 1, or 2).

        Raises:
            RuntimeError: If model not trained.
        """
        if self._theta is None or self._observables is None:
            raise RuntimeError("Model not trained. Call run() first.")

        x_t = torch.tensor(x, dtype=torch.float64)
        logits = get_batch_logits(
            x_t, self._theta, self._observables, backend, device, dtype,
        )
        return torch.argmax(logits, dim=1).numpy()


# ===================================================================
# 9. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "iris_L3_e20",
        "layers": 3, "epochs": 20, "lr": 0.05, "batch_size": 16,
        "min_acc": 0.70,
    },
    {
        "name": "iris_L5_e30",
        "layers": 5, "epochs": 30, "lr": 0.03, "batch_size": 8,
        "min_acc": 0.80,
    },
    {
        "name": "iris_L4_e25",
        "layers": 4, "epochs": 25, "lr": 0.05, "batch_size": 16,
        "min_acc": 0.75,
    },
    {
        "name": "iris_L2_e15",
        "layers": 2, "epochs": 15, "lr": 0.1, "batch_size": 32,
        "min_acc": 0.60,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]

    result = vqc_train(
        layers=case["layers"], epochs=case["epochs"],
        lr=case["lr"], batch_size=case["batch_size"],
        verbose=False, seed=42,
    )

    acc = result["Final Accuracy"]
    ok = acc >= case["min_acc"]
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {name}: acc={acc:.2%} (min={case['min_acc']:.0%}), "
          f"loss={result['Final Loss']:.4f}, "
          f"time={result['Quantal Computation Time (s)']:.0f}s")
    return ok


# ===================================================================
# 10. Main
# ===================================================================

def main() -> None:
    """Run the complete VQC demonstration pipeline."""
    if not HAS_TORCH:
        print("PyTorch is not installed. Install with: pip install torch")
        return

    print("=" * 60)
    print("VQC — Variational Quantum Classifier")
    print("  Iris Dataset (4 features → 4 qubits, 3 classes)")
    print("=" * 60)

    clf = VQCClassifier()

    # --- Demo 1: Basic training ---
    print("\n--- Demo 1: Basic VQC (L=3, epochs=20) ---")
    result1 = vqc_train(
        layers=3, epochs=20, lr=0.05, batch_size=16, verbose=True, seed=42,
    )
    print(f"  Accuracy: {result1['Final Accuracy']:.2%}")

    # --- Demo 2: Deeper model ---
    print("\n--- Demo 2: Deeper VQC (L=5, epochs=30) ---")
    result2 = vqc_train(
        layers=5, epochs=30, lr=0.03, batch_size=8, verbose=True, seed=42,
    )
    print(f"  Accuracy: {result2['Final Accuracy']:.2%}")

    # --- Demo 3: Training dynamics ---
    print("\n--- Demo 3: Training Dynamics ---")
    acc3 = result2["acc_history"]
    print(f"  Epoch 1  acc: {acc3[0]:.2%}")
    print(f"  Epoch 10 acc: {acc3[min(9, len(acc3)-1)]:.2%}")
    print(f"  Epoch 20 acc: {acc3[min(19, len(acc3)-1)]:.2%}")
    print(f"  Final acc:    {acc3[-1]:.2%}")
    print(f"  Improvement:  {acc3[-1] - acc3[0]:.2%}")

    # --- Demo 4: Effect of layers ---
    print("\n--- Demo 4: Effect of circuit depth ---")
    for l_val in [1, 2, 3, 4, 5, 6]:
        result4 = vqc_train(
            layers=l_val, epochs=20, lr=0.05, batch_size=16,
            verbose=False, seed=42,
        )
        print(f"  L={l_val}: acc={result4['Final Accuracy']:.2%}, "
              f"loss={result4['Final Loss']:.4f}, "
              f"time={result4['Quantal Computation Time (s)']:.0f}s")

    # --- Demo 5: Effect of learning rate ---
    print("\n--- Demo 5: Effect of learning rate (L=4, epochs=25) ---")
    for lr_val in [0.01, 0.03, 0.05, 0.10, 0.20]:
        result5 = vqc_train(
            layers=4, epochs=25, lr=lr_val, batch_size=16,
            verbose=False, seed=42,
        )
        marker = " ✓" if result5["Final Accuracy"] > 0.80 else ""
        print(f"  lr={lr_val:.2f}: acc={result5['Final Accuracy']:.2%}, "
              f"loss={result5['Final Loss']:.4f}{marker}")

    # --- Demo 6: Prediction on test samples ---
    print("\n--- Demo 6: Predictions on first 10 test samples ---")
    _, _, x_test, y_test = load_iris_data()
    model = VQCClassifier()
    result6 = model.run(layers=5, epochs=30, lr=0.03, batch_size=8)
    preds = model.predict(x_test[:10].numpy())
    for i in range(10):
        match = "✓" if preds[i] == y_test[i].item() else "✗"
        iris_names = ["setosa", "versicolor", "virginica"]
        print(f"    Sample {i}: true={iris_names[y_test[i].item()]:>10}, "
              f"pred={iris_names[preds[i]]:>10} [{match}]")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
