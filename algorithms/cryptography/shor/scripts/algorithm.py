"""Shor's Algorithm — factor a composite integer."""

from unitarylab.algorithms import ShorAlgorithm


def example_factor_15():
    """Factor N=15.  Expected factors: {3, 5}."""
    algo = ShorAlgorithm()
    result = algo.run(
        N=15,
        method="matrix",
        backend="torch",
        max_retries=15,
    )

    print("=" * 50)
    print("Shor Example: N = 15")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Factors      : {result['factors']}")
    print(f"  Period       : {result['period']}")
    print(f"  Circuit path : {result.get('circuit_path')}")
    print(f"  Message      : {result['message']}")


def example_factor_21():
    """Factor N=21.  Expected factors: {3, 7}."""
    algo = ShorAlgorithm()
    result = algo.run(
        N=21,
        method="matrix",
        backend="torch",
        max_retries=20,
    )

    print("=" * 50)
    print("Shor Example: N = 21")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Factors      : {result['factors']}")
    print(f"  Period       : {result['period']}")
    print(f"  Message      : {result['message']}")


if __name__ == "__main__":
    example_factor_15()
    example_factor_21()
