"""HHL Algorithm — solve linear system Ax = b on a quantum simulator."""

import numpy as np
from unitarylab_algorithms import HHLAlgorithm


def example_2x2():
    """Solve a 2x2 Hermitian system with HHL."""
    A = np.array([[1.5, 0.5],
                  [0.5, 1.5]])
    b = np.array([1.0, 0.0])

    algo = HHLAlgorithm(text_mode="plain")
    result = algo.run(A=A, b=b, d=4, backend="torch")

    print("=" * 50)
    print("HHL Example: 2x2 Hermitian System")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file       : {f['filename']} ({f['format']})")
    print(f"  Status            : {result['status']}")
    print(f"  Quantum solution  : {np.round(result['Estimated solution (quantum)'], 4)}")
    print(f"  Classical solution: {np.round(result['Exact solution (classical)'], 4)}")
    print(f"  L2 error          : {result['L2 error']:.4e}")
    print(f"  Post-select prob  : {result['Post-selection probability']:.4f}")
    print(f"  Computation time  : {result['Computation time (s)']:.4f} s")
    print(f"  Circuit path      : {result.get('circuit_path')}")


def example_4x4():
    """Solve a 4x4 tridiagonal system with HHL."""
    A = np.array([[2.0, -1.0,  0.0,  0.0],
                  [-1.0, 2.0, -1.0,  0.0],
                  [0.0, -1.0,  2.0, -1.0],
                  [0.0,  0.0, -1.0,  2.0]], dtype=complex)
    b = np.array([1.0, 0.0, 0.0, 0.0])

    algo = HHLAlgorithm(text_mode="plain")
    result = algo.run(A=A, b=b, d=6, backend="torch")

    print("=" * 50)
    print("HHL Example: 4x4 Tridiagonal System")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file       : {f['filename']} ({f['format']})")
    print(f"  Status            : {result['status']}")
    print(f"  Quantum solution  : {np.round(result['Estimated solution (quantum)'], 4)}")
    print(f"  Classical solution: {np.round(result['Exact solution (classical)'], 4)}")
    print(f"  L2 error          : {result['L2 error']:.4e}")
    print(f"  Post-select prob  : {result['Post-selection probability']:.4f}")
    print(f"  Computation time  : {result['Computation time (s)']:.4f} s")


if __name__ == "__main__":
    example_2x2()
    example_4x4()
