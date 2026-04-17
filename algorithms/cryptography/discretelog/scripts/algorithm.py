"""Discrete Logarithm — solve g^x = y (mod P) via quantum period-finding."""

from engine.algorithms import DiscreteLogAlgorithm


def example_small_prime():
    """Solve 3^x ≡ 6 (mod 7).  Expected x = 3."""
    algo = DiscreteLogAlgorithm()
    result = algo.run(g=3, y=6, P=7, backend="torch")

    print("=" * 50)
    print("Discrete Log: 3^x ≡ 6 (mod 7)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Found x      : {result['found_x']}")
    print(f"  Circuit path : {result.get('circuit_path')}")
    print(f"  Message      : {result['message']}")

    if result["found_x"] is not None:
        x = int(result["found_x"])
        print(f"  Verification : 3^{x} mod 7 = {pow(3, x, 7)}  (target = 6)")


if __name__ == "__main__":
    example_small_prime()
