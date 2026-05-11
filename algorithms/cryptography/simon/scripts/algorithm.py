"""Simon's Algorithm — find hidden string s such that f(x) = f(x ⊕ s)."""

from unitarylab-algorithms import SimonAlgorithm


def example_4bit():
    """Recover hidden string s = 1010."""
    algo = SimonAlgorithm()
    result = algo.run(s_target="1010", backend="torch")

    print("=" * 50)
    print("Simon Example: s = 1010")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Found s      : {result['found_s']}")
    print(f"  Circuit path : {result.get('circuit_path')}")
    print(f"  Message      : {result['message']}")


def example_3bit():
    """Recover hidden string s = 110."""
    algo = SimonAlgorithm()
    result = algo.run(s_target="110", backend="torch")

    print("=" * 50)
    print("Simon Example: s = 110")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Found s      : {result['found_s']}")


if __name__ == "__main__":
    example_4bit()
    example_3bit()
