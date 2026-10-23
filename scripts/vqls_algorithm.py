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
    A = np.array(
        [
            [1.8, 0.2, 0.0, 0.0],
            [0.2, 1.6, 0.1, 0.0],
            [0.0, 0.1, 1.4, 0.2],
            [0.0, 0.0, 0.2, 1.2],
        ],
        dtype=complex,
    )
    b = np.array([1.0, 0.5, -0.25, 0.75], dtype=complex)

    algo = VQLSAlgorithm(text_mode="plain")
    result = algo.run(
        A=A,
        b=b,
        cost_function="local_classical",
        n_layers=4,
        maxiter=50,
        tol=1e-6,
        seed=42,
    )

    print("=" * 60)
    print("VQLS Example: 2-qubit variational linear solver")
    print("=" * 60)
    print(f"  Status              : {result.get('status')}")
    fidelity = result.get("Fidelity")
    fidelity_text = "None" if fidelity is None else f"{fidelity:.6f}"
    print(f"  Fidelity            : {fidelity_text}")
    print(f"  Ax fidelity         : {result.get('Ax Fidelity'):.6f}")
    print(f"  Cost function       : {result.get('Cost Function')}")
    print(f"  Condition number    : {result.get('Condition Number'):.6f}")
    print(f"  Early stopped flag  : {result.get('Early Stopped')}")
    print(f"  Computation time    : {result.get('Computation Time (s)'):.4f} s")
    print(f"  Quantum solution    : {np.asarray(result.get('Solution State (Quantum)'))}")
    print(f"  Classical solution  : {np.asarray(result.get('Solution State (Classical)'))}")
    print(f"  Circuit path        : {result.get('circuit_path')}")
    for item in result.get("plot", []):
        print(f"  Output file         : {item['filename']} ({item['format']})")
    return result


if __name__ == "__main__":
    example_two_qubit_vqls()
