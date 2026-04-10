"""Examples for DiscreteLogAlgorithm based on the SKILL guide."""

from engine.algorithms import DiscreteLogAlgorithm


def example_small_prime() -> None:
    """Solve 3^x = 6 (mod 7), expected x = 3."""
    dlg = DiscreteLogAlgorithm()
    result = dlg.run(g=3, y=6, P=7, backend="torch", algo_dir="./dlg_results")

    print("=" * 60)
    print("Discrete Log Example 1: g=3, y=6, P=7")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"status: {result.get('status')}")
    print(f"found_x: {result.get('found_x')}")
    print(f"circuit_path: {result.get('circuit_path')}")
    print(f"message: {result.get('message')}")


def example_second_case() -> None:
    """Solve 2^x = 13 (mod 29)."""
    g, y, p = 2, 13, 29
    dlg = DiscreteLogAlgorithm()
    result = dlg.run(g=g, y=y, P=p, backend="torch", algo_dir="./dlg_results")

    print("=" * 60)
    print("Discrete Log Example 2: g=2, y=13, P=29")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"status: {result.get('status')}")
    print(f"found_x: {result.get('found_x')}")
    if result.get("found_x") is not None:
        x = int(result["found_x"])
        print(f"verification: {g}^{x} mod {p} = {pow(g, x, p)} (target={y})")


if __name__ == "__main__":
    example_small_prime()
    example_second_case()
