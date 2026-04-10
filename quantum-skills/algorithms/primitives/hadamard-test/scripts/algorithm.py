"""Examples for Hadamard Test modes: expectation, swap test, phase estimation."""

from engine.core import GateSequence
from engine.algorithms import HadamardTestAlgorithm


def run_expectation_example() -> None:
    """Estimate real and imaginary parts of <psi|U|psi>."""
    U = GateSequence(1)
    U.rz(0.8, 0)

    prepare_psi = GateSequence(1)
    prepare_psi.h(0)

    algo = HadamardTestAlgorithm()

    result_re = algo.run(
        mode="expectation",
        U=U,
        prepare_psi=prepare_psi,
        imag=False,
        shots=20000,
        backend="torch",
        algo_dir="./hadamard_test_results",
    )

    result_im = algo.run(
        mode="expectation",
        U=U,
        prepare_psi=prepare_psi,
        imag=True,
        shots=20000,
        backend="torch",
        algo_dir="./hadamard_test_results",
    )

    print("=" * 60)
    print("Hadamard Test Example: Expectation Mode")
    print("=" * 60)
    print(result_re.get("plot", "(no plot)"))
    print(f"Re estimate: {result_re.get('estimated_value')}")
    print(f"Im estimate: {result_im.get('estimated_value')}")
    print(f"Complex estimate: {result_re.get('estimated_value')} + {result_im.get('estimated_value')}j")


def run_swap_test_example() -> None:
    """Estimate overlap |<phi|psi>|^2 via swap test mode."""
    prepare_psi = GateSequence(1)
    prepare_psi.h(0)

    prepare_phi = GateSequence(1)
    prepare_phi.ry(0.4, 0)

    algo = HadamardTestAlgorithm()
    result = algo.run(
        mode="swap_test",
        prepare_psi=prepare_psi,
        prepare_phi=prepare_phi,
        shots=20000,
        backend="torch",
        algo_dir="./hadamard_test_results",
    )

    print("=" * 60)
    print("Hadamard Test Example: Swap Test Mode")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Overlap estimate: {result.get('estimated_value')}")


if __name__ == "__main__":
    run_expectation_example()
    run_swap_test_example()
