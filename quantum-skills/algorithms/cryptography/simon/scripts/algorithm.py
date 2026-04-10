"""Examples for SimonAlgorithm based on the SKILL guide."""

from engine.algorithms import SimonAlgorithm


def example_secret_1010() -> None:
    """Recover secret s=1010."""
    simon = SimonAlgorithm()
    result = simon.run(s_target="1010", backend="torch", algo_dir="./simon_results")

    print("=" * 60)
    print("Simon Example 1: s_target=1010")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"status: {result.get('status')}")
    print(f"found_s: {result.get('found_s')}")
    print(f"circuit_path: {result.get('circuit_path')}")
    print(f"message: {result.get('message')}")


def example_secret_1101() -> None:
    """Recover secret s=1101."""
    simon = SimonAlgorithm()
    result = simon.run(s_target="1101", backend="torch", algo_dir="./simon_results")

    print("=" * 60)
    print("Simon Example 2: s_target=1101")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"status: {result.get('status')}")
    print(f"found_s: {result.get('found_s')}")
    print(f"circuit_path: {result.get('circuit_path')}")
    print(f"message: {result.get('message')}")


if __name__ == "__main__":
    example_secret_1010()
    example_secret_1101()
