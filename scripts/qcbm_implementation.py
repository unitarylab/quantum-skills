"""Manual implementation of Quantum Circuit Born Machine (QCBM).

QCBM is an unsupervised generative model that uses the Born rule to map a
parameterized quantum circuit's measurement outcomes to a probability
distribution. It learns the 2×2 Bars-and-Stripes (BAS) distribution over
4 qubits by minimizing KL divergence via the Parameter Shift Rule.

Architecture:
    1. Target: BAS distribution — 6 valid 2×2 patterns with uniform prob 1/6.
    2. Circuit: L layers of RY rotations + ring CNOT entanglement.
    3. Born rule: p_θ(x) = |⟨x|U(θ)|0⟩^n|².
    4. Loss: D_KL(p_target || p_θ) minimized via Parameter Shift gradients.
    5. Optimizer: Adam.

Bars-and-Stripes (2×2, 4 qubits):
    Valid states: 0000(all-off), 0011(row pair), 0101(col pair),
                  1010(col pair), 1100(row pair), 1111(all-on)
    Each has probability 1/6 ≈ 0.1667.

Components:
    - get_bas_dist: Build BAS target distribution
    - build_qcbm_circuit: RY + ring-CNOT ansatz
    - get_probs: Born-rule probability vector
    - qcbm_train: Full Parameter Shift + KL divergence training
    - QCBMGenerator: Class-based interface
    - plot_qcbm_results: Loss curve + distribution comparison + sample grid

Reference:
    SKILL.md — QCBM
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
# 1. Bars-and-Stripes target distribution
# ===================================================================


def _grid_shape(n_qubits: int) -> Tuple[int, int]:
    """Determine the 2D grid shape for n qubits.

    Tries to make the grid as square as possible.

    Args:
        n_qubits: Number of qubits.

    Returns:
        Tuple of (rows, cols).
    """
    rows = int(np.sqrt(n_qubits))
    while rows > 1 and n_qubits % rows != 0:
        rows -= 1
    cols = n_qubits // rows
    return rows, cols


def get_bas_dist(n_qubits: int = 4) -> Tuple[np.ndarray, List[int]]:
    """Build the Bars-and-Stripes target distribution.

    For a rows×cols grid, valid patterns are:
    - Bars (rows uniform): each column in a row has the same value
    - Stripes (cols uniform): each row in a column has the same value

    For 2×2 (n=4):
        Valid = [0(0000), 3(0011), 5(0101), 10(1010), 12(1100), 15(1111)]

    Args:
        n_qubits: Number of qubits. Must correspond to a valid grid.

    Returns:
        Tuple of (probs_array, valid_states_list).
            - probs_array: shape (2^n,) with uniform prob on valid states.
            - valid_states: sorted list of valid state indices.
    """
    rows, cols = _grid_shape(n_qubits)
    valid: set = set()

    # Row-uniform patterns (bars): each row is all-0 or all-1
    for row_bits in range(1 << rows):
        bits = f"{row_bits:0{rows}b}"
        state = "".join(bit * cols for bit in bits)
        valid.add(int(state, 2))

    # Column-uniform patterns (stripes): each column is all-0 or all-1
    for col_bits in range(1 << cols):
        bits = f"{col_bits:0{cols}b}"
        state = bits * rows
        valid.add(int(state, 2))

    valid_sorted = sorted(valid)
    probs = np.zeros(1 << n_qubits)
    probs[valid_sorted] = 1.0 / len(valid_sorted)
    return probs, valid_sorted


def get_bas_dist_torch(n_qubits: int = 4) -> Tuple["torch.Tensor", List[int]]:
    """Torch version of get_bas_dist."""
    probs, valid = get_bas_dist(n_qubits)
    return torch.from_numpy(probs), valid


# ===================================================================
# 2. QCBM circuit builder
# ===================================================================


def build_qcbm_circuit(theta: "torch.Tensor", n_qubits: int) -> Circuit:
    """Build the QCBM variational circuit.

    Architecture (per layer l):
        - RY(θ[l,q]) on each qubit q
        - If l < L-1: ring CNOT entanglement: CX(q, (q+1) mod n) for all q

    The ring CNOT generates long-range correlations needed for BAS patterns.
    The last layer has no CNOT so the RY gates act as the final rotation
    before measurement.

    Args:
        theta: Parameter tensor of shape (layers, n_qubits).
        n_qubits: Number of qubits.

    Returns:
        Circuit object.
    """
    layers = theta.shape[0]
    qc = Circuit(n_qubits, name=f"QCBM_L{layers}")

    for l in range(layers):
        # RY rotations on all qubits
        for q in range(n_qubits):
            qc.ry(float(theta[l, q]), q)

        # Ring CNOT entanglement (skip on last layer)
        if l < layers - 1:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)

    return qc


# ===================================================================
# 3. Born-rule probability extraction
# ===================================================================


def get_probs(
    theta: "torch.Tensor",
    n_qubits: int,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> "torch.Tensor":
    """Execute circuit and return Born-rule probability vector p_θ(x).

    p_θ(x) = |⟨x|ψ(θ)⟩|² = |amplitude_x|²

    Args:
        theta: Parameter tensor (layers, n_qubits).
        n_qubits: Number of qubits.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Torch tensor of shape (2^n,) with probabilities summing to 1.
    """
    qc = build_qcbm_circuit(theta, n_qubits)
    result = qc.execute(backend=backend, device=device, dtype=dtype)
    amplitudes = np.asarray(result.state, dtype=complex).flatten()
    return torch.as_tensor(np.abs(amplitudes) ** 2)


# ===================================================================
# 4. KL divergence loss
# ===================================================================


def kl_divergence(
    target_probs: "torch.Tensor",
    current_probs: "torch.Tensor",
    eps: float = 1e-12,
) -> "torch.Tensor":
    """Compute KL divergence D_KL(p_target || p_current).

    D_KL = Σ_x p_target(x) · log(p_target(x) / (p_current(x) + ε))

    Only states with non-zero target probability contribute to the sum.

    Args:
        target_probs: Target distribution (shape 2^n,).
        current_probs: Current model distribution (shape 2^n,).
        eps: Small constant for numerical stability.

    Returns:
        Scalar KL divergence.
    """
    return torch.sum(
        target_probs * torch.log((target_probs + eps) / (current_probs + eps))
    )


# ===================================================================
# 5. Parameter Shift gradient + training loop
# ===================================================================


def compute_parameter_shift_gradient(
    theta: "torch.Tensor",
    target_probs: "torch.Tensor",
    current_probs: "torch.Tensor",
    n_qubits: int,
    shift: float = np.pi / 2,
    eps: float = 1e-12,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> "torch.Tensor":
    """Compute gradient of KL loss via the Parameter Shift Rule.

    For each parameter θ[l,q]:
        ∂L/∂θ[l,q] = Σ_x -target(x)/(curr(x)+ε) · ∂p(x)/∂θ[l,q]

    where ∂p/∂θ[l,q] = ½[p(θ[l,q]+π/2) − p(θ[l,q]-π/2)]

    Args:
        theta: Current parameters (layers, n_qubits).
        target_probs: Target distribution (2^n,).
        current_probs: Current probabilities (2^n,).
        n_qubits: Number of qubits.
        shift: Parameter shift amount (default π/2).
        eps: KL stability constant.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Gradient tensor of same shape as theta.
    """
    layers, n_q = theta.shape
    grad = torch.zeros_like(theta)

    for l in range(layers):
        for q in range(n_q):
            # Shifted parameters
            th_plus = theta.detach().clone()
            th_plus[l, q] += shift
            th_minus = theta.detach().clone()
            th_minus[l, q] -= shift

            # Shifted probabilities
            p_plus = get_probs(th_plus, n_qubits, backend, device, dtype)
            p_minus = get_probs(th_minus, n_qubits, backend, device, dtype)

            # Gradient of probability wrt θ[l,q]
            grad_p = 0.5 * (p_plus - p_minus)

            # Chain rule: d(KL)/dθ = Σ -target/(curr+ε) · dp/dθ
            grad[l, q] = torch.sum(
                -(target_probs / (current_probs + eps)) * grad_p,
            )

    return grad


def qcbm_train(
    n_qubits: int = 4,
    layers: int = 4,
    epochs: int = 40,
    lr: float = 0.1,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a QCBM to learn the BAS distribution.

    Pipeline:
        1. Build BAS target distribution (uniform over 6 valid states).
        2. Initialize random parameters θ ∈ [0, 2π).
        3. For each epoch: compute probabilities, KL loss, parameter shift
           gradients, Adam update.
        4. Return final parameters, loss history, and final distribution.

    Training cost: 2 × layers × n_qubits circuit evaluations per epoch
    (from parameter shift: ±π/2 for each parameter).

    Args:
        n_qubits: Number of qubits (default 4 for 2×2 BAS).
        layers: Number of variational layers (>= 3 recommended).
        epochs: Training epochs (40-100 typical).
        lr: Adam learning rate (0.05-0.15 typical).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - Final KL Loss: KL divergence at last epoch
            - loss_history: List of per-epoch KL values
            - final_probs: Learned distribution (2^n,)
            - target_probs: BAS target distribution (2^n,)
            - valid_states: List of BAS valid state indices
            - theta: Final trained parameters (layers, n_qubits)
            - Quantum Computation Time (s): Training time
            - n_qubits, layers, epochs: Config values

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If parameters are invalid.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    if layers <= 0:
        raise ValueError(f"layers must be positive, got {layers}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)

    # --- Stage 1: Target distribution ---
    target_probs, valid_states = get_bas_dist_torch(n_qubits)

    # Initialize parameters
    theta = torch.nn.Parameter(torch.rand((layers, n_qubits)) * 2 * np.pi)
    optimizer = torch.optim.Adam([theta], lr=lr)
    shift = np.pi / 2
    eps = 1e-12

    if verbose:
        print(f"QCBM Training")
        print(f"  Qubits:       {n_qubits}")
        print(f"  State space:  2^{n_qubits} = {1 << n_qubits}")
        print(f"  Valid states: {len(valid_states)} ({valid_states})")
        print(f"  Target prob:  1/{len(valid_states)} ≈ {1.0/len(valid_states):.4f}")
        print(f"  Layers:       {layers}")
        print(f"  Epochs:       {epochs}, lr={lr}")
        print(f"  Shifts/epoch: {2 * layers * n_qubits} circuit evals")

    # --- Stage 2: Preview circuit ---
    qc_preview = build_qcbm_circuit(theta.detach(), n_qubits)
    if verbose:
        print(f"  Circuit:      {qc_preview.get_num_qubits()} qubits")

    # --- Stage 3: Training loop ---
    if verbose:
        print(f"  Training...")

    loss_history: List[float] = []
    train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        # Current probabilities
        curr_probs = get_probs(theta.detach(), n_qubits, backend, device, dtype)

        # KL divergence loss
        loss_val = kl_divergence(target_probs, curr_probs, eps)
        loss_history.append(float(loss_val.item()))

        # Parameter shift gradient
        grad = compute_parameter_shift_gradient(
            theta.detach(), target_probs, curr_probs, n_qubits,
            shift=shift, eps=eps, backend=backend, device=device, dtype=dtype,
        )

        # Adam update
        optimizer.zero_grad()
        theta.grad = grad
        optimizer.step()

        if verbose and (ep % 10 == 0 or ep == 1 or ep == epochs):
            print(f"    Epoch {ep:>4}/{epochs} | KL Loss: {loss_val.item():.6f}")

    train_time = time.perf_counter() - train_start

    # --- Stage 4: Final evaluation ---
    with torch.no_grad():
        final_probs = get_probs(theta, n_qubits, backend, device, dtype).numpy()

    # Validation: check that BAS states have high probability
    bas_mass = float(np.sum(final_probs[valid_states]))
    final_kl = loss_history[-1]

    if verbose:
        print(f"  Training time:  {train_time:.2f}s")
        print(f"  Final KL Loss:  {final_kl:.6f}")
        print(f"  BAS prob mass:  {bas_mass:.4f} (ideal: 1.0)")
        idx_max = np.argmax(final_probs)
        print(f"  Mode state:     {idx_max} "
              f"({'VALID' if idx_max in valid_states else 'invalid'})")
        print(f"  Status:         ok")

    return {
        "status": "ok",
        "Final KL Loss": final_kl,
        "loss_history": loss_history,
        "final_probs": final_probs,
        "target_probs": target_probs.numpy(),
        "valid_states": valid_states,
        "theta": theta.detach(),
        "bas_prob_mass": bas_mass,
        "Quantum Computation Time (s)": round(train_time, 2),
        "n_qubits": n_qubits,
        "layers": layers,
        "epochs": epochs,
        "circuit": qc_preview,
        "circuit_path": "",
        "plot": [],
    }


# ===================================================================
# 6. Visualization
# ===================================================================


def plot_qcbm_results(
    target_probs: np.ndarray,
    final_probs: np.ndarray,
    loss_history: List[float],
    n_qubits: int,
    valid_states: List[int],
    output_dir: str = ".",
) -> Dict[str, str]:
    """Generate QCBM result plots: loss, distribution, samples.

    Args:
        target_probs: BAS target probabilities (2^n,).
        final_probs: Learned probabilities (2^n,).
        loss_history: Per-epoch KL loss values.
        n_qubits: Number of qubits.
        valid_states: BAS valid state indices.
        output_dir: Output directory.

    Returns:
        Dict mapping plot type to file path.
    """
    if not HAS_MATPLOTLIB:
        return {}

    import os

    paths: Dict[str, str] = {}
    rows, cols = _grid_shape(n_qubits)
    x = np.arange(len(target_probs))

    # --- Loss curve ---
    loss_path = os.path.abspath(os.path.join(output_dir, "QCBM_Loss.svg"))
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, color="#e67e22", lw=2)
    plt.title("QCBM KL Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    paths["loss"] = loss_path

    # --- Distribution comparison ---
    dist_path = os.path.abspath(os.path.join(output_dir, "QCBM_Distribution.svg"))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, target_probs, 0.4, label="Target (BAS)", alpha=0.6, color="gray")
    plt.bar(
        x + 0.2, final_probs, 0.4, label="QCBM Learned",
        color="#3498db", alpha=0.8,
    )
    # Mark valid states
    for vs in valid_states:
        plt.axvline(x=vs, color="green", alpha=0.2, linewidth=0.5)
    plt.xlabel("Basis State (decimal)")
    plt.ylabel("Probability")
    plt.title("QCBM: Target vs Learned Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150)
    plt.close()
    paths["distribution"] = dist_path

    # --- Sample grid ---
    samples_path = os.path.abspath(os.path.join(output_dir, "QCBM_Samples.svg"))
    # Sample 12 states from the learned distribution
    rng = np.random.default_rng(42)
    samples = rng.choice(len(final_probs), size=min(12, len(final_probs)),
                         p=final_probs / final_probs.sum())
    n_samples = len(samples)
    n_cols_grid = min(4, n_samples)
    n_rows_grid = int(np.ceil(n_samples / n_cols_grid))

    fig = plt.figure(figsize=(3 * n_cols_grid, 3 * n_rows_grid))
    for i, s in enumerate(samples):
        ax = fig.add_subplot(n_rows_grid, n_cols_grid, i + 1)
        bits = f"{int(s):0{n_qubits}b}"
        grid = np.array([int(b) for b in bits]).reshape(rows, cols)
        ax.imshow(grid, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        is_valid = "✓" if int(s) in valid_states else "✗"
        ax.set_title(f"State {s} [{is_valid}]")
    plt.suptitle("QCBM Generated Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(samples_path, dpi=150)
    plt.close()
    paths["samples"] = samples_path

    return paths


# ===================================================================
# 7. Class-based interface
# ===================================================================


class QCBMGenerator:
    """Class-based QCBM generative model.

    Usage:
        model = QCBMGenerator()
        result = model.run(n_qubits=4, layers=4, epochs=40, lr=0.1)
        print(result['Final KL Loss'])

        # Generate samples
        samples = model.sample(n_samples=8)
        print(samples)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self._final_probs: Optional[np.ndarray] = None
        self._n_qubits: int = 4

    def run(
        self,
        n_qubits: int = 4,
        layers: int = 4,
        epochs: int = 40,
        lr: float = 0.1,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Train QCBM. See qcbm_train() for docs."""
        result = qcbm_train(
            n_qubits=n_qubits, layers=layers, epochs=epochs, lr=lr,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self._final_probs = result["final_probs"]
        self._n_qubits = n_qubits

        # Generate plots
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            plot_paths = plot_qcbm_results(
                result["target_probs"], result["final_probs"],
                result["loss_history"], n_qubits,
                result["valid_states"], self.algo_dir,
            )
            result["plot"] = [
                {"format": "svg", "filename": p}
                for p in plot_paths.values()
            ]

        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Final KL Loss": result.get("Final KL Loss"),
            "Quantum Computation Time (s)": result.get("Quantum Computation Time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }

    def sample(self, n_samples: int = 8, seed: int = 42) -> np.ndarray:
        """Generate samples from the learned distribution.

        Args:
            n_samples: Number of samples.
            seed: Random seed.

        Returns:
            Array of sampled state indices (integers).

        Raises:
            RuntimeError: If model has not been trained yet.
        """
        if self._final_probs is None:
            raise RuntimeError("Model not trained. Call run() first.")
        rng = np.random.default_rng(seed)
        return rng.choice(
            len(self._final_probs), size=n_samples,
            p=self._final_probs / self._final_probs.sum(),
        )


# ===================================================================
# 8. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "BAS_4q_L4_e40",
        "n_qubits": 4, "layers": 4, "epochs": 40, "lr": 0.1,
        "max_kl": 2.0,
    },
    {
        "name": "BAS_4q_L6_e60",
        "n_qubits": 4, "layers": 6, "epochs": 60, "lr": 0.08,
        "max_kl": 1.5,
    },
    {
        "name": "BAS_4q_L3_e30",
        "n_qubits": 4, "layers": 3, "epochs": 30, "lr": 0.1,
        "max_kl": 2.5,
    },
    {
        "name": "BAS_4q_L8_e80",
        "n_qubits": 4, "layers": 8, "epochs": 80, "lr": 0.05,
        "max_kl": 1.0,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = qcbm_train(
        n_qubits=case["n_qubits"], layers=case["layers"],
        epochs=case["epochs"], lr=case["lr"],
        verbose=False, seed=42,
    )
    kl = result["Final KL Loss"]
    ok = kl < case["max_kl"]
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {name}: KL={kl:.4f} (max={case['max_kl']}), "
          f"BAS_mass={result['bas_prob_mass']:.3f}, "
          f"time={result['Quantum Computation Time (s)']:.0f}s")
    return ok


# ===================================================================
# 9. Main
# ===================================================================

def main() -> None:
    """Run the complete QCBM demonstration pipeline."""
    if not HAS_TORCH:
        print("PyTorch is not installed. Install with: pip install torch")
        return

    print("=" * 60)
    print("QCBM — Quantum Circuit Born Machine")
    print("  Bars-and-Stripes Generative Model")
    print("=" * 60)

    # --- Demo 1: Basic training ---
    print("\n--- Demo 1: Basic QCBM (L=4, epochs=40) ---")
    result1 = qcbm_train(
        n_qubits=4, layers=4, epochs=40, lr=0.1, verbose=True, seed=42,
    )
    print(f"  Final KL:    {result1['Final KL Loss']:.6f}")
    print(f"  BAS mass:    {result1['bas_prob_mass']:.4f}")

    # --- Demo 2: Deeper model ---
    print("\n--- Demo 2: Deeper QCBM (L=6, epochs=60) ---")
    result2 = qcbm_train(
        n_qubits=4, layers=6, epochs=60, lr=0.08, verbose=True, seed=42,
    )
    print(f"  Final KL:    {result2['Final KL Loss']:.6f}")
    print(f"  BAS mass:    {result2['bas_prob_mass']:.4f}")

    # --- Demo 3: Distribution comparison ---
    print("\n--- Demo 3: Learned vs Target Distribution ---")
    target3, valid3 = get_bas_dist(4)
    final3 = result2["final_probs"]
    print(f"  Valid states: {valid3}")
    print(f"  Target probs (BAS): {[f'{target3[v]:.3f}' for v in valid3]}")
    print(f"  Learned probs:      {[f'{final3[v]:.3f}' for v in valid3]}")
    # Top BAS state recovery
    bas_ranked = sorted(valid3, key=lambda v: final3[v], reverse=True)
    print(f"  Top BAS state: {bas_ranked[0]} (prob={final3[bas_ranked[0]]:.4f})")
    print(f"  Top overall:   {np.argmax(final3)} (prob={final3.max():.4f})")

    # --- Demo 4: Layer count vs KL ---
    print("\n--- Demo 4: Effect of circuit depth ---")
    for l_val in [2, 3, 4, 6, 8]:
        result4 = qcbm_train(
            n_qubits=4, layers=l_val, epochs=50, lr=0.08,
            verbose=False, seed=42,
        )
        print(f"  L={l_val}: KL={result4['Final KL Loss']:.4f}, "
              f"BAS_mass={result4['bas_prob_mass']:.3f}, "
              f"time={result4['Quantum Computation Time (s)']:.0f}s")

    # --- Demo 5: Sample generation ---
    print("\n--- Demo 5: Generated Samples ---")
    model = QCBMGenerator()
    result5 = model.run(n_qubits=4, layers=6, epochs=60, lr=0.08)
    samples = model.sample(n_samples=12, seed=123)
    _, valid5 = get_bas_dist(4)
    valid_count = sum(1 for s in samples if s in valid5)
    print(f"  Samples: {list(samples)}")
    print(f"  Valid BAS: {valid_count}/{len(samples)} "
          f"({valid_count/len(samples):.0%})")

    # --- Demo 6: BAS distribution validation ---
    print("\n--- Demo 6: BAS Distribution (2×2 grid) ---")
    probs6, valid6 = get_bas_dist(4)
    rows6, cols6 = _grid_shape(4)
    print(f"  Grid: {rows6}×{cols6}")
    print(f"  Valid states ({len(valid6)}):")
    for s in valid6:
        bits = f"{s:04b}"
        grid = np.array([int(b) for b in bits]).reshape(rows6, cols6)
        grid_str = str(grid).replace("\n", ", ")
        print(f"    {s} ({bits}): {grid_str} — prob={probs6[s]:.4f}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
