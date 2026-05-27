"""Quantum Signal Processing (QSP) — polynomial approximation of cos(t * x)."""

from unitarylab_algorithms import QSPAlgorithm


def example_low_degree():
    """Approximate cos(t * x) with t=1.0, degree d=6, at x=0.5."""
    algo = QSPAlgorithm(text_mode="legacy")
    result = algo.run(
        t=1.0,
        d=6,
        x=0.5,
        backend="torch",
    )

    print("=" * 50)
    print("QSP Example: t=1.0, d=6, x=0.5")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file      : {f['filename']} ({f['format']})")
    print(f"  Status           : {result['status']}")
    print(f"  Estimated value  : {result['Estimated value']}")
    print(f"  Ideal value      : {result['Ideal value']:.6f}")
    print(f"  Absolute error   : {result['Absolute error']:.6e}")
    print(f"  Computation time : {result['Computation time (s)']:.4f} s")
    print(f"  Circuit path     : {result.get('circuit_path')}")


def example_high_degree():
    """Higher degree for better accuracy: t=2.0, d=15, x=0.3."""
    algo = QSPAlgorithm(text_mode="legacy")
    result = algo.run(
        t=2.0,
        d=15,
        x=0.3,
        backend="torch",
    )

    print("=" * 50)
    print("QSP Example: t=2.0, d=15, x=0.3")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file      : {f['filename']} ({f['format']})")
    print(f"  Status           : {result['status']}")
    print(f"  Estimated value  : {result['Estimated value']}")
    print(f"  Ideal value      : {result['Ideal value']:.6f}")
    print(f"  Absolute error   : {result['Absolute error']:.6e}")
    print(f"  Computation time : {result['Computation time (s)']:.4f} s")
    print(f"  Circuit path     : {result.get('circuit_path')}")


if __name__ == "__main__":
    example_low_degree()
    example_high_degree()
