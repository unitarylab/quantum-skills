"""Quantum Fourier Transform (QFT) — Fourier-basis transform and inverse."""

import numpy as np
from unitarylab_algorithms import QFTAlgorithm


def example_forward_qft():
    """Apply QFT to |001> on a 3-qubit register."""
    state = np.zeros(8, dtype=complex)
    state[1] = 1.0

    algo = QFTAlgorithm(text_mode="plain")
    result = algo.run(
        n=3,
        state=state,
        inverse=False,
        backend="torch",
    )

    print("=" * 50)
    print("QFT Example: 3-Qubit Basis State |001>")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file       : {f['filename']} ({f['format']})")
    print(f"  Status            : {result['status']}")
    print(f"  Verification error: {result['Verification error']:.6e}")
    print(f"  Final state       : {np.round(result['Final state'], 4)}")
    print(f"  Expected state    : {np.round(result['Expected state'], 4)}")
    print(f"  Circuit path      : {result.get('circuit_path')}")


def example_inverse_qft():
    """Apply inverse QFT to a normalized 3-qubit test state."""
    state = np.array([1, 1j, -1, -1j, 0.5, -0.5j, 0.25, -0.25j], dtype=complex)
    state = state / np.linalg.norm(state)

    algo = QFTAlgorithm(text_mode="plain")
    result = algo.run(
        n=3,
        state=state,
        inverse=True,
        backend="torch",
    )

    print("=" * 50)
    print("IQFT Example: 3-Qubit Normalized Test State")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file       : {f['filename']} ({f['format']})")
    print(f"  Status            : {result['status']}")
    print(f"  Verification error: {result['Verification error']:.6e}")
    print(f"  Circuit path      : {result.get('circuit_path')}")


if __name__ == "__main__":
    example_forward_qft()
    example_inverse_qft()
