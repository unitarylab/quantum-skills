"""Hadamard Test — estimate expectation values and state overlaps."""

from unitarylab.algorithms import HadamardTestAlgorithm
from unitarylab.core import Circuit


def example_expectation():
    """Estimate Re and Im of <+|RZ(0.8)|+>.  Expected: cos(0.4), -sin(0.4)."""
    U = Circuit(1, name="RZ_0.8", backend="torch")
    U.rz(0.8, 0)

    prepare_psi = Circuit(1, name="|+>", backend="torch")
    prepare_psi.h(0)

    algo = HadamardTestAlgorithm()

    result_re = algo.run(
        mode="expectation", U=U, prepare_psi=prepare_psi,
        imag=False, shots=20000, backend="torch",
    )
    result_im = algo.run(
        mode="expectation", U=U, prepare_psi=prepare_psi,
        imag=True, shots=20000, backend="torch",
    )

    print("=" * 50)
    print("Hadamard Test: Expectation Mode")
    print("=" * 50)
    print(result_re.get("plot", ""))
    print(f"  Re estimate: {result_re['estimated_value']:.4f}  (expected ~0.9211)")
    print(f"  Im estimate: {result_im['estimated_value']:.4f}  (expected ~-0.3894)")


def example_swap_test():
    """Estimate overlap |<phi|psi>|^2 via swap test."""
    prepare_psi = Circuit(1, name="|+>", backend="torch")
    prepare_psi.h(0)

    prepare_phi = Circuit(1, name="ry_0.4", backend="torch")
    prepare_phi.ry(0.4, 0)

    algo = HadamardTestAlgorithm()
    result = algo.run(
        mode="swap_test",
        prepare_psi=prepare_psi,
        prepare_phi=prepare_phi,
        shots=20000,
        backend="torch",
    )

    print("=" * 50)
    print("Hadamard Test: Swap Test Mode")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Overlap estimate: {result['estimated_value']:.4f}")


if __name__ == "__main__":
    example_expectation()
    example_swap_test()
