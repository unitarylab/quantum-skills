"""Grover Search — find a marked computational-basis state."""

import sys
from pathlib import Path


for parent in Path(__file__).resolve().parents:
    if (parent / "unitarylab_algorithms").is_dir():
        sys.path.insert(0, str(parent))
        break

from unitarylab_algorithms import GroverAlgorithm


def example_3qubit():
    """Search for target |101> in an 8-state search space."""
    algo = GroverAlgorithm()
    result = algo.run(
        n=3,
        target="101",
        backend="torch",
    )

    print("=" * 50)
    print("Grover Search: 3-Qubit Target |101>")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status          : {result['status']}")
    print(f"  Found state     : {result['Result']}")
    print(f"  Target prob     : {result['Amplified target-state probability']:.4f}")
    print(f"  Circuit path    : {result.get('circuit_path')}")


def example_4qubit():
    """Search for target |1101> in a 16-state search space."""
    algo = GroverAlgorithm()
    result = algo.run(
        n=4,
        target="1101",
        backend="torch",
    )

    print("=" * 50)
    print("Grover Search: 4-Qubit Target |1101>")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status          : {result['status']}")
    print(f"  Found state     : {result['Result']}")
    print(f"  Target prob     : {result['Amplified target-state probability']:.4f}")
    print(f"  Circuit path    : {result.get('circuit_path')}")


if __name__ == "__main__":
    example_3qubit()
    example_4qubit()
