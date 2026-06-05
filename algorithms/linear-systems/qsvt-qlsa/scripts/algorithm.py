"""QSVT QLSA verification script generated from the leaf skill."""

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

from unitarylab_algorithms.linear_algebra.qsvt_qlsa.algorithm import QSVTLinearSolverAlgorithm


def example_diagonal_system() -> dict:
    """Solve the small diagonal system from the skill parameter table."""
    A = np.array([[0.8, 0.0], [0.0, 0.4]])
    b = np.array([1.0, 2.0])

    algo = QSVTLinearSolverAlgorithm(text_mode="plain")
    result = algo.run(
        A=A,
        b=b,
        epsilon=0.0001,
        backend="torch",
        device="cpu",
        dtype=np.complex128,
    )

    print("=" * 60)
    print("QSVT QLSA Example: 2x2 diagonal linear system")
    print("=" * 60)
    print(f"  Status                 : {result.get('status')}")
    print(f"  Solution vector        : {np.asarray(result.get('Solution vector'))}")
    print(f"  Scaling factor applied : {result.get('Scaling factor applied')}")
    print(f"  Simulation time        : {result.get('Simulation time (s)')} s")
    print(f"  Circuit path           : {result.get('circuit_path')}")
    for item in result.get("plot", []):
        print(f"  Output file            : {item['filename']} ({item['format']})")
    return result


if __name__ == "__main__":
    example_diagonal_system()
