"""
Quantum Singular Value Transformation Linear Systems Algorithm (QSVT-QLSA) Example
===================================================================================
Demonstrates solving linear systems Ax = b using Quantum Singular Value Transformation 
(QSVT), a cutting-edge algorithm that achieves better scaling than HHL by directly 
transforming singular values rather than eigenvalues.

Theory recap
------------
QSVT-QLSA solves: Given a matrix A and vector b, find x such that Ax = b by:

  1. Building a block-encoding of A: a unitary W in a larger space
  2. Approximating the inverse function f(x) = 1/x using Chebyshev polynomials
  3. Using Quantum Signal Processing (QSP) to construct the polynomial block-encoding
  4. Post-processing to extract the solution state |x⟩ ∝ A⁻¹|b⟩

QSVT-QLSA achieves better complexity than HHL:
  - HHL: O(κ²/ε) gates
  - QSVT-QLSA: O(κ·√(log(κ/ε))) gates, polylogarithmic in 1/ε

QSVT-QLSA advantages:
  - Works with non-Hermitian (arbitrary) matrices via SVD
  - Polynomial (not exponential!) dependence on error tolerance 1/ε
  - State-of-the-art theoretical complexity
"""

from __future__ import annotations

import os

import numpy as np

from engine import GateSequence
from engine.algorithms.linear_algebra import QSVTLinearSolverAlgorithm


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def print_separator(title: str = "") -> None:
    width = 60
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def report_result(result: dict, n_sys: int, cond_num: float) -> None:
    """Print a structured summary of a QSVT-QLSA result."""
    print(result["plot"])
    print(f"  System qubits         : {n_sys}")
    print(f"  Condition number κ    : {cond_num:.2f}")
    print(f"  Solution vector       : {result['solution_vector']}")
    print(f"  Circuit diagram       : {result['circuit_path']}")


# ---------------------------------------------------------------------------
# Simple block-encodings for demonstration
# ---------------------------------------------------------------------------

def build_simple_block_encoding_diag(
    eigenvalues: list[float],
    n_anc: int = 1,
) -> GateSequence:
    """
    Build a simple block-encoding for a diagonal matrix.
    
    For a 2-qubit system, encodes diag(a₀, a₁) via controlled rotations.
    The ancilla qubit's state determines which eigenvalue is encoded.
    """
    gs = GateSequence(1 + n_anc)
    
    for sys_state, ev in enumerate(eigenvalues):
        # For each possible system register state, encode eigenvalue via RY
        theta = 2.0 * np.arccos(np.clip(ev, 0, 1))
        
        # Controlled rotation: apply RY when sys register = sys_state
        control = list(range(1, 1 + n_anc))  # Ancilla qubits
        target = 0  # System qubit
        
        # Flip ancilla bits for sys_state
        ancilla_flips = []
        for bit_idx in range(n_anc):
            if not ((sys_state >> bit_idx) & 1):
                ancilla_flips.append(1 + bit_idx)
        
        for flip_qubit in ancilla_flips:
            gs.x(flip_qubit)
        
        # Multi-controlled RY
        if control:
            gs.mcry(theta, control, target)
        
        # Unflip
        for flip_qubit in ancilla_flips:
            gs.x(flip_qubit)
    
    return gs


# ---------------------------------------------------------------------------
# Example 1 — Simple 1-qubit system: diagonal A = diag(2, 1)
# ---------------------------------------------------------------------------

def example_1qubit_diagonal(algo: QSVTLinearSolverAlgorithm, output_dir: str) -> dict:
    """
    Solve Ax = b where A = [[2, 0], [0, 1]] is diagonal.

    Matrix properties:
      - Eigenvalues: 2, 1
      - Condition number κ = 2/1 = 2 (well-conditioned)
      - RHS: b = [1, 0]ᵀ (normalized)

    QSVT polynomial will approximate 1/λ for each eigenvalue.
    """
    print_separator("Example 1 — 1-Qubit Diagonal System: A = diag(2, 1)")

    eigenvalues = [2.0, 1.0]
    n_sys = 1
    n_anc_be = 1

    # Build block-encoding
    gs_be = build_simple_block_encoding_diag(eigenvalues, n_anc=n_anc_be)

    # Right-hand side
    b = np.array([1.0, 0.0])

    # Condition number
    kappa = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

    print(f"  Matrix A              : diag(2.0, 1.0)")
    print(f"  Eigenvalues           : {eigenvalues}")
    print(f"  Condition number κ    : {kappa:.2f}")
    print(f"  Right-hand side b     : {b}")
    print(f"  System qubits n_sys   : {n_sys}")
    print(f"  Ancilla qubits n_anc  : {n_anc_be}")

    # Run QSVT
    result = algo.run(
        gs_be=gs_be,
        b=b,
        kappa=kappa,
        alpha=1.0,
        epsilon=0.001,
        n_sys=n_sys,
        n_anc_be=n_anc_be,
        backend="torch",
        algo_dir=os.path.join(output_dir, "1qubit_diag"),
    )

    report_result(result, n_sys, kappa)

    # Expected solution: x = A⁻¹b = [1/2, 0]ᵀ (before normalization)
    expected = np.array([0.5, 0.0])
    print(f"  Expected solution     : {expected}")

    return result


# ---------------------------------------------------------------------------
# Example 2 — 2-qubit system: scaled diagonal coefficients
# ---------------------------------------------------------------------------

def example_2qubit_diagonal(algo: QSVTLinearSolverAlgorithm, output_dir: str) -> dict:
    """
    Solve Ax = b where A has 4 eigenvalues with controlled condition number.

    Matrix properties:
      - Eigenvalues: [4.0, 2.0, 1.0, 0.5] (geometrically decreasing)
      - Condition number κ = 4.0/0.5 = 8 (moderate conditioning)
      - RHS: b = [1, 1, 1, 1]ᵀ (uniform, normalized)

    This 2-qubit system demonstrates scaling to larger problems.
    """
    print_separator("Example 2 — 2-Qubit Diagonal System: κ = 8")

    eigenvalues = [4.0, 2.0, 1.0, 0.5]
    n_sys = 2  # 2 qubits = 4 states
    n_anc_be = 2

    # Build block-encoding
    gs_be = build_simple_block_encoding_diag(eigenvalues, n_anc=n_anc_be)

    # Right-hand side (uniform)
    b = np.ones(4)

    # Condition number
    kappa = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

    print(f"  Matrix A              : diag(4.0, 2.0, 1.0, 0.5)")
    print(f"  Eigenvalues           : {eigenvalues}")
    print(f"  Condition number κ    : {kappa:.2f}")
    print(f"  Right-hand side b     : all ones (normalized)")
    print(f"  System qubits n_sys   : {n_sys}")
    print(f"  Ancilla qubits n_anc  : {n_anc_be}")

    # Run QSVT
    result = algo.run(
        gs_be=gs_be,
        b=b,
        kappa=kappa,
        alpha=1.0,
        epsilon=0.001,
        n_sys=n_sys,
        n_anc_be=n_anc_be,
        backend="torch",
        algo_dir=os.path.join(output_dir, "2qubit_diag"),
    )

    report_result(result, n_sys, kappa)

    # Expected solution: x = A⁻¹b unnormalized: [1/4, 1/2, 1, 2]ᵀ
    expected_unnorm = np.array([0.25, 0.5, 1.0, 2.0])
    print(f"  Expected (unnormalized): {expected_unnorm}")

    return result


# ---------------------------------------------------------------------------
# Example 3 — Well-conditioned 2-qubit: κ = 2
# ---------------------------------------------------------------------------

def example_2qubit_wellcond(algo: QSVTLinearSolverAlgorithm, output_dir: str) -> dict:
    """
    Solve Ax = b where A is very well-conditioned: κ = 2.

    Matrix properties:
      - Eigenvalues: [2.0, 1.5, 1.2, 1.0] (all close together)
      - Condition number κ = 2.0/1.0 = 2 (excellent conditioning)
      - RHS: b = 1-norm vector = [0.5, 0.5, 0.5, 0.5]ᵀ

    Well-conditioned systems are easier for quantum algorithms to solve.
    QSVT should achieve high accuracy with moderate resources.
    """
    print_separator("Example 3 — 2-Qubit Well-Conditioned: κ = 2")

    eigenvalues = [2.0, 1.5, 1.2, 1.0]
    n_sys = 2
    n_anc_be = 2

    # Build block-encoding
    gs_be = build_simple_block_encoding_diag(eigenvalues, n_anc=n_anc_be)

    # Right-hand side (uniform after normalization)
    b = np.ones(4) / 2.0

    # Condition number
    kappa = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

    print(f"  Matrix A              : diag(2.0, 1.5, 1.2, 1.0)")
    print(f"  Eigenvalues           : {eigenvalues}")
    print(f"  Condition number κ    : {kappa:.2f}")
    print(f"  Right-hand side b     : uniform after norm")
    print(f"  System qubits n_sys   : {n_sys}")
    print(f"  Ancilla qubits n_anc  : {n_anc_be}")

    # Run QSVT
    result = algo.run(
        gs_be=gs_be,
        b=b,
        kappa=kappa,
        alpha=1.0,
        epsilon=0.001,
        n_sys=n_sys,
        n_anc_be=n_anc_be,
        backend="torch",
        algo_dir=os.path.join(output_dir, "2qubit_wellcond"),
    )

    report_result(result, n_sys, kappa)

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = "./qsvt_results"
    os.makedirs(output_dir, exist_ok=True)

    algo = QSVTLinearSolverAlgorithm()

    print_separator("QSVT-QLSA Algorithm — Example Script")
    print("  Quantum Singular Value Transformation Linear Systems.")
    print("  Solves Ax = b via polynomial SVD transformation with improved")
    print("  scaling: O(κ·√log(κ/ε)) vs. HHL's O(κ²/ε).")
    print("  Results (circuit diagrams, etc.) written to:")
    print(f"    {os.path.abspath(output_dir)}")

    results = {}
    results["1q_diag"] = example_1qubit_diagonal(algo, output_dir)
    results["2q_diag"] = example_2qubit_diagonal(algo, output_dir)
    results["2q_wellcond"] = example_2qubit_wellcond(algo, output_dir)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print_separator("Summary")
    print(f"  {'Example':<30} {'Condition':>12} {'Difficulty':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")

    configs = [
        ("1-Qubit Diagonal", 2.0, results["1q_diag"]),
        ("2-Qubit Diagonal", 8.0, results["2q_diag"]),
        ("2-Qubit Well-Cond.", 2.0, results["2q_wellcond"]),
    ]
    for label, kappa, res in configs:
        if kappa <= 2.0:
            status = "EASY"
        elif kappa <= 5.0:
            status = "MODERATE"
        else:
            status = "HARD"
        print(f"  {label:<30} {kappa:>12.2f} {status:>10}")

    print()


if __name__ == "__main__":
    main()
