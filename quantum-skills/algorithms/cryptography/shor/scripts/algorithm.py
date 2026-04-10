"""Examples for ShorAlgorithm based on the SKILL guide."""

from engine.algorithms import ShorAlgorithm


def example_factor_15() -> None:
    """Factor N=15 using matrix method."""
    shor = ShorAlgorithm()
    result = shor.run(
        N=15,
        method="matrix",
        backend="torch",
        max_retries=15,
        algo_dir="./shor_results",
    )

    print("=" * 60)
    print("Shor Example 1: N=15")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"status: {result.get('status')}")
    print(f"factors: {result.get('factors')}")
    print(f"period: {result.get('period')}")
    print(f"circuit_path: {result.get('circuit_path')}")
    print(f"message: {result.get('message')}")


def example_factor_21() -> None:
    """Factor N=21 using matrix method."""
    shor = ShorAlgorithm()
    result = shor.run(
        N=21,
        method="matrix",
        backend="torch",
        max_retries=20,
        algo_dir="./shor_results",
    )

    print("=" * 60)
    print("Shor Example 2: N=21")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"status: {result.get('status')}")
    print(f"factors: {result.get('factors')}")
    print(f"period: {result.get('period')}")
    print(f"circuit_path: {result.get('circuit_path')}")
    print(f"message: {result.get('message')}")


if __name__ == "__main__":
    example_factor_15()
    example_factor_21()
