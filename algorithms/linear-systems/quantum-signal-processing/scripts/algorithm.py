"""Quantum Signal Processing (QSP) — polynomial approximation of exp(-i*tau*x)."""

from engine.algorithms import QSPAlgorithm


def example_low_degree():
    """Approximate exp(-i*x) with degree 6 at x=0.5."""
    algo = QSPAlgorithm(seed=42)
    result = algo.run(
        target_tau=1.0,
        degree=6,
        x_value=0.5,
        backend="torch",
    )

    print("=" * 50)
    print("QSP Example: tau=1.0, degree=6, x=0.5")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Error        : {result['error']:.6e}")
    print(f"  Circuit path : {result.get('circuit_path')}")


def example_high_degree():
    """Higher degree for better accuracy: tau=2.0, degree=15."""
    algo = QSPAlgorithm(seed=42)
    result = algo.run(
        target_tau=2.0,
        degree=15,
        x_value=0.3,
        backend="torch",
    )

    print("=" * 50)
    print("QSP Example: tau=2.0, degree=15, x=0.3")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Error        : {result['error']:.6e}")


if __name__ == "__main__":
    example_low_degree()
    example_high_degree()
