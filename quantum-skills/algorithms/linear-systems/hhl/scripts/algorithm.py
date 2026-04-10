"""
HHL Algorithm Example using UnitaryLab
=======================================
Demonstrates solving linear systems Ax = b using the
Harrow-Hassidim-Lloyd (HHL) quantum algorithm via UnitaryLab's
engine.algorithms.linear_algebra.HHLAlgorithm class.

Theory recap
------------
Given a Hermitian matrix A and vector b, HHL prepares the quantum state:

    |x⟩ ∝ A⁻¹|b⟩

Key steps inside HHLAlgorithm.run():
  1. Initialise |b⟩ in the system register.
  2. Quantum Phase Estimation (QPE) on U = exp(iAt) to encode eigenvalues.
  3. Controlled reciprocal rotation: encodes 1/λ onto an ancilla qubit.
  4. Inverse QPE to disentangle the phase register.
  5. Post-select on ancilla = |1⟩ to extract the solution state.

The classical runtime for a sparse, well-conditioned N×N system scales as
O(N s κ log(1/ε)), while HHL scales as O(log(N) s² κ² / ε) — an
exponential improvement in N when κ and 1/ε are polylogarithmic.
"""

from __future__ import annotations

import os

import numpy as np

from engine.algorithms.linear_algebra import HHLAlgorithm


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def make_tridiagonal(n: int, diag: float = 2.0, off: float = -1.0) -> np.ndarray:
    """Return a real symmetric n×n tridiagonal matrix."""
    A = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(A, diag)
    np.fill_diagonal(A[1:], off)
    np.fill_diagonal(A[:, 1:], off)
    return A


def print_separator(title: str = "") -> None:
    width = 52
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def report_result(result: dict, A: np.ndarray, b: np.ndarray) -> None:
    """Print a structured summary of an HHL result dictionary."""
    x_q = result["solution_quantum"]
    x_c = result["solution_classical"]
    error = float(np.linalg.norm(x_q - x_c))
    rel_error = error / (float(np.linalg.norm(x_c)) + 1e-30)

    print(result["plot"])
    print(f"  Classical solution : {np.round(x_c, 6)}")
    print(f"  Quantum solution   : {np.round(x_q, 6)}")
    print(f"  Absolute error     : {error:.6e}")
    print(f"  Relative error     : {rel_error:.4%}")
    print(f"  Post-selection P   : {result['post_selection_prob']:.6f}")
    print(f"  Circuit diagram    : {result['circuit_path']}")

    residual = float(np.linalg.norm(A @ x_q - b / np.linalg.norm(b)))
    print(f"  Residual ‖Ax-b̂‖   : {residual:.6e}")


# ---------------------------------------------------------------------------
# Example 1 — 2×2 system (minimal, pedagogical)
# ---------------------------------------------------------------------------

def example_2x2(algo: HHLAlgorithm, output_dir: str) -> dict:
    """
    Solve a simple 2×2 Hermitian system.

    A = [[2, 1],   b = [1, 0]ᵀ
         [1, 2]]

    Classical solution: x = A⁻¹b = [2/3, -1/3]ᵀ
    """
    print_separator("Example 1 — 2×2 Hermitian System")

    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]], dtype=complex)

    b = np.array([1.0, 0.0], dtype=complex)
    b_norm = b / np.linalg.norm(b)

    print(f"  A =\n{A.real}\n  b = {b.real}")

    result = algo.run(
        A=A,
        b=b,
        d=5,                          # 5-bit phase register
        backend="torch",
        algo_dir=os.path.join(output_dir, "2x2"),
    )

    report_result(result, A, b_norm)
    return result


# ---------------------------------------------------------------------------
# Example 2 — 4×4 tridiagonal system
# ---------------------------------------------------------------------------

def example_4x4_tridiagonal(algo: HHLAlgorithm, output_dir: str) -> dict:
    """
    Solve a 4×4 tridiagonal (discrete Laplacian) system.

        A = tridiag(-1, 2, -1),  b = ones(4)

    This arises naturally from finite-difference discretisation of
    the 1D Poisson equation −u″ = f.
    """
    print_separator("Example 2 — 4×4 Tridiagonal (Discrete Laplacian)")

    A = make_tridiagonal(4, diag=2.0, off=-1.0)
    b = np.ones(4, dtype=complex)

    x_ref = np.linalg.solve(A, b)
    print(f"  A =\n{A.real}\n  b = {b.real}")
    print(f"  Reference solution: {np.round(x_ref, 6)}")

    result = algo.run(
        A=A,
        b=b,
        d=6,                          # 6-bit phase register for better precision
        backend="torch",
        algo_dir=os.path.join(output_dir, "4x4_tridiagonal"),
    )

    report_result(result, A, b / np.linalg.norm(b))
    return result


# ---------------------------------------------------------------------------
# Example 3 — 4×4 diagonal system (easy verification)
# ---------------------------------------------------------------------------

def example_4x4_diagonal(algo: HHLAlgorithm, output_dir: str) -> dict:
    """
    Solve a 4×4 diagonal Hermitian system.

        A = diag(1, 2, 3, 4),  b = [1, 1, 1, 1]ᵀ

    Known exact solution: x = [1, 1/2, 1/3, 1/4]ᵀ  (after b normalisation).
    Pure diagonal systems are the simplest possible test for the
    reciprocal-rotation step because eigenvalues = diagonal entries.
    """
    print_separator("Example 3 — 4×4 Diagonal System")

    eigenvalues = np.array([1.0, 2.0, 3.0, 4.0])
    A = np.diag(eigenvalues.astype(complex))
    b = np.ones(4, dtype=complex)

    x_ref = np.linalg.solve(A, b)
    print(f"  A = diag{tuple(eigenvalues)}")
    print(f"  b = {b.real}")
    print(f"  Reference solution: {np.round(x_ref, 6)}")

    result = algo.run(
        A=A,
        b=b,
        d=6,
        backend="torch",
        algo_dir=os.path.join(output_dir, "4x4_diagonal"),
    )

    b_norm = b / np.linalg.norm(b)
    report_result(result, A, b_norm)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = "./hhl_results"
    os.makedirs(output_dir, exist_ok=True)

    algo = HHLAlgorithm()

    print_separator("HHL Algorithm — Example Script")
    print("  Solving Ax = b on a quantum simulator via UnitaryLab.")
    print("  Results (circuit diagrams, state vectors) written to:")
    print(f"    {os.path.abspath(output_dir)}")

    results = {}
    results["2x2"] = example_2x2(algo, output_dir)
    results["4x4_tridiagonal"] = example_4x4_tridiagonal(algo, output_dir)
    results["4x4_diagonal"] = example_4x4_diagonal(algo, output_dir)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print_separator("Summary")
    print(f"  {'Example':<30} {'Abs error':>12} {'P_success':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")

    labels = {
        "2x2": "2×2 Hermitian",
        "4x4_tridiagonal": "4×4 Tridiagonal",
        "4x4_diagonal": "4×4 Diagonal",
    }
    for key, label in labels.items():
        r = results[key]
        err = float(np.linalg.norm(r["solution_quantum"] - r["solution_classical"]))
        p = r["post_selection_prob"]
        status = "OK" if err < 1e-2 else "WARN"
        print(f"  {label:<30} {err:>12.4e} {p:>12.6f}  [{status}]")

    print()


if __name__ == "__main__":
    main()
