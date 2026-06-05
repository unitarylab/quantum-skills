"""VQLS verification script generated from the leaf skill."""

from pathlib import Path
import sys

import numpy as np


def _add_workspace_root_to_path() -> None:
    """Allow running from the skill folder without installing the sibling package."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        workspace_root = parent.parent
        if (workspace_root / "unitarylab_algorithms").is_dir():
            sys.path.insert(0, str(workspace_root))
            return


_add_workspace_root_to_path()

from unitarylab_algorithms.linear_algebra.vqls.algorithm import VQLSAlgorithm


def example_two_qubit_vqls() -> dict:
    """Run a compact VQLS instance using parameters described in the skill."""
    algo = VQLSAlgorithm(text_mode="plain")
    result = algo.run(
        n_qubits=2,
        coefficients=[1.0, 0.2, 0.2],
        max_iterations=50,
        tolerance=1e-6,
        initial_spread=0.5,
        backend="torch",
        device="cpu",
        dtype=np.complex128,
    )

    print("=" * 60)
    print("VQLS Example: 2-qubit variational linear solver")
    print("=" * 60)
    print(f"  Status              : {result.get('status')}")
    print(f"  Fidelity            : {result.get('Fidelity'):.6f}")
    print(f"  Relative Error      : {result.get('Relative Error'):.6e}")
    print(f"  Residual Norm       : {result.get('Residual Norm'):.6e}")
    print(f"  Computation time    : {result.get('Computation Time (s)'):.4f} s")
    print(f"  Quantum solution    : {np.asarray(result.get('Solution State (Quantum)'))}")
    print(f"  Classical solution  : {np.asarray(result.get('Solution State (Classical)'))}")
    print(f"  Circuit path        : {result.get('circuit_path')}")
    for item in result.get("plot", []):
        print(f"  Output file         : {item['filename']} ({item['format']})")
    return result


if __name__ == "__main__":
    example_two_qubit_vqls()
