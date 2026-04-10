"""
Quantum Signal Processing (QSP) Algorithm Example using UnitaryLab
==================================================================
Demonstrates polynomial approximation of functions using Quantum Signal 
Processing (QSP), a framework for applying arbitrary polynomial 
transformations to quantum signals via phase sequences.

Theory recap
------------
QSP solves the problem: given a target polynomial function P(x) and signal 
operator W(x), construct a quantum circuit that evaluates P(x) by:

  1. Optimize phase sequence Φ = (φ₀, φ₁, ..., φ_d) to match target function
  2. Build quantum circuit with alternating phases and rotations
  3. Execute on signal operator to compute P(x)
  4. Extract function value from quantum state measurement

The fundamental QSP result: there exists a phase sequence that implements

    U_Φ = e^(i·φ₀·Z) ∏ₖ [W(x)·e^(i·φₖ·Z)]

whose (0,0) block encodes the target polynomial P(x).

QSP is essential for:
  - Optimal linear systems solving (better than HHL)
  - Hamiltonian simulation and digital quantum simulation
  - Arbitrary polynomial function evaluation
  - Foundation for Quantum Singular Value Transformation (QSVT)
"""

from __future__ import annotations

import os

import numpy as np

from engine.algorithms.linear_algebra import QSPAlgorithm


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def print_separator(title: str = "") -> None:
    width = 60
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def report_result(result: dict, tau: float, degree: int, x_val: float) -> None:
    """Print a structured summary of a QSP result."""
    print(result["plot"])
    print(f"  Target function       : exp(-i·τ·x) with τ = {tau:.2f}")
    print(f"  Polynomial degree     : {degree}")
    print(f"  Test point x          : {x_val:.4f}")
    print(f"  Approximation error   : {result['error']:.6e}")
    print(f"  Circuit diagram       : {result['circuit_path']}")


# ---------------------------------------------------------------------------
# Example 1 — Low degree approximation: exp(-i·x) with degree 4
# ---------------------------------------------------------------------------

def example_low_degree(algo: QSPAlgorithm, output_dir: str) -> dict:
    """
    Approximate exp(-i·x) using a degree-4 polynomial.

    Parameters:
      τ = 1.0 (evolution parameter)
      degree = 4 (low polynomial degree)
      x_value = 0.5 (test at eigenvalue 0.5)

    Lower degree = faster circuit but larger approximation error.
    """
    print_separator("Example 1 — exp(-i·x), Degree 4 (Low Degree)")

    target_tau = 1.0
    degree = 4
    x_value = 0.5

    print(f"  Target: exp(-i·{target_tau}·x)")
    print(f"  Degree: {degree} (economical, but coarser approximation)")
    print(f"  Domain: x ∈ [-1, 1]")

    result = algo.run(
        target_tau=target_tau,
        degree=degree,
        x_value=x_value,
        backend="torch",
        algo_dir=os.path.join(output_dir, "low_degree"),
    )

    report_result(result, target_tau, degree, x_value)

    return result


# ---------------------------------------------------------------------------
# Example 2 — Medium degree: exp(-i·1.5·x) with degree 12
# ---------------------------------------------------------------------------

def example_medium_degree(algo: QSPAlgorithm, output_dir: str) -> dict:
    """
    Approximate exp(-i·1.5·x) using degree-12 polynomial.

    Parameters:
      τ = 1.5 (increased evolution parameter)
      degree = 12 (medium polynomial degree)
      x_value = 0.7 (test at eigenvalue 0.7)

    Medium degree balances circuit depth with approximation fidelity.
    """
    print_separator("Example 2 — exp(-i·1.5·x), Degree 12 (Medium Degree)")

    target_tau = 1.5
    degree = 12
    x_value = 0.7

    print(f"  Target: exp(-i·{target_tau}·x)")
    print(f"  Degree: {degree} (balanced complexity & accuracy)")
    print(f"  Domain: x ∈ [-1, 1]")

    result = algo.run(
        target_tau=target_tau,
        degree=degree,
        x_value=x_value,
        backend="torch",
        algo_dir=os.path.join(output_dir, "medium_degree"),
    )

    report_result(result, target_tau, degree, x_value)

    return result


# ---------------------------------------------------------------------------
# Example 3 — High degree, large τ: exp(-i·3.0·x) with degree 20
# ---------------------------------------------------------------------------

def example_high_degree(algo: QSPAlgorithm, output_dir: str) -> dict:
    """
    Approximate exp(-i·3.0·x) using degree-20 polynomial.

    Parameters:
      τ = 3.0 (larger evolution parameter)
      degree = 20 (higher polynomial degree)
      x_value = 0.3 (test at eigenvalue 0.3)

    Higher degree = more accurate function approximation but deeper circuit.
    Demonstrates that QSP can achieve high-precision approximations.
    """
    print_separator("Example 3 — exp(-i·3.0·x), Degree 20 (High Degree)")

    target_tau = 3.0
    degree = 20
    x_value = 0.3

    print(f"  Target: exp(-i·{target_tau}·x)")
    print(f"  Degree: {degree} (high precision, deeper circuit)")
    print(f"  Domain: x ∈ [-1, 1]")

    result = algo.run(
        target_tau=target_tau,
        degree=degree,
        x_value=x_value,
        backend="torch",
        algo_dir=os.path.join(output_dir, "high_degree"),
    )

    report_result(result, target_tau, degree, x_value)

    return result


# ---------------------------------------------------------------------------
# Example 4 — Zero function: exp(-i·0·x) = 1 (diagonal only)
# ---------------------------------------------------------------------------

def example_constant_function(algo: QSPAlgorithm, output_dir: str) -> dict:
    """
    Approximate the constant function exp(-i·0·x) = 1.

    Parameters:
      τ = 0.0 (zero evolution)
      degree = 6
      x_value = 0.5

    Expected result: polynomial is constant 1 everywhere on [-1, 1].
    This is the simplest verifiable test case.
    """
    print_separator("Example 4 — exp(-i·0·x) = 1 (Constant Function)")

    target_tau = 0.0
    degree = 6
    x_value = 0.5

    print(f"  Target: exp(-i·{target_tau}·x) = constant 1")
    print(f"  Degree: {degree}")
    print(f"  Expected: perfect evaluation ∀x (error ≈ 0)")

    result = algo.run(
        target_tau=target_tau,
        degree=degree,
        x_value=x_value,
        backend="torch",
        algo_dir=os.path.join(output_dir, "constant"),
    )

    report_result(result, target_tau, degree, x_value)

    return result


# ---------------------------------------------------------------------------
# Example 5 — Varying evaluation point: exp(-i·2.0·x) at multiple x
# ---------------------------------------------------------------------------

def example_varying_points(algo: QSPAlgorithm, output_dir: str) -> dict:
    """
    Approximate exp(-i·2.0·x) and evaluate at multiple test points.

    This demonstrates that a single trained QSP circuit works 
    accurately across the entire domain [-1, 1].
    """
    print_separator("Example 5 — exp(-i·2.0·x) at Multiple Evaluation Points")

    target_tau = 2.0
    degree = 16
    test_points = [0.0, 0.3, 0.6, 0.9]

    print(f"  Target: exp(-i·{target_tau}·x)")
    print(f"  Degree: {degree}")
    print(f"  Test points: {test_points}")
    print()

    errors = []
    for x_value in test_points:
        print(f"  Evaluating at x = {x_value:.1f}:")

        result = algo.run(
            target_tau=target_tau,
            degree=degree,
            x_value=x_value,
            backend="torch",
            algo_dir=os.path.join(output_dir, f"varying_x_{x_value:.2f}"),
        )

        error = result["error"]
        errors.append(error)
        print(f"    Approximation error: {error:.6e}")

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    print(f"\n  Average error: {avg_error:.6e}")
    print(f"  Max error:     {max_error:.6e}")

    return {"errors": errors, "test_points": test_points}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = "./qsp_results"
    os.makedirs(output_dir, exist_ok=True)

    algo = QSPAlgorithm()

    print_separator("QSP Algorithm — Example Script")
    print("  Quantum Signal Processing - Polynomial Function Approximation.")
    print("  Demonstrates applying arbitrary polynomial transformations")
    print("  to quantum signals via optimized phase sequences.")
    print("  Results (circuit diagrams, etc.) written to:")
    print(f"    {os.path.abspath(output_dir)}")

    results = {}
    results["low_degree"] = example_low_degree(algo, output_dir)
    results["medium_degree"] = example_medium_degree(algo, output_dir)
    results["high_degree"] = example_high_degree(algo, output_dir)
    results["constant"] = example_constant_function(algo, output_dir)
    results["varying"] = example_varying_points(algo, output_dir)

    # -------------------------------------------------------------------
    # Summary table for fixed-point examples
    # -------------------------------------------------------------------
    print_separator("Summary of Fixed Evaluation Points")
    print(f"  {'Function':<30} {'Degree':>8} {'Error':>12}")
    print(f"  {'-'*30} {'-'*8} {'-'*12}")

    configs = [
        ("exp(-i·1.0·x)", 4, results["low_degree"]),
        ("exp(-i·1.5·x)", 12, results["medium_degree"]),
        ("exp(-i·3.0·x)", 20, results["high_degree"]),
        ("exp(-i·0.0·x)", 6, results["constant"]),
    ]
    for label, deg, res in configs:
        err = res["error"]
        status = "GOOD" if err < 1e-3 else "FAIR" if err < 1e-2 else "POOR"
        print(f"  {label:<30} {deg:>8} {err:>12.4e}  [{status}]")

    print()


if __name__ == "__main__":
    main()
