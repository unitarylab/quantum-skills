"""
QSP-based Hamiltonian Simulation Example
=========================================
Demonstrates how to use QSPHSAlgorithm to approximate e^{-iHt} via
polynomial spectral transformation with a block-encoded Hamiltonian.

Covers:
  1. Basic single-run usage
  2. Parameter sweep over evolution time t and polynomial degree
  3. Inspecting the approximate vs. exact evolution matrices
"""

import numpy as np
from unitarylab-algorithms import QSPHSAlgorithm

# ---------------------------------------------------------------------------
# Hamiltonian definition
# ---------------------------------------------------------------------------
# 2x2 Hermitian matrix (eigenvalues ≈ 1.382 and 3.618)
H = np.array([[2, 1],
              [1, 3]], dtype=complex)

# ---------------------------------------------------------------------------
# 1. Basic single run
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Basic single run")
print("=" * 60)

algo = QSPHSAlgorithm(text_mode="plain")
result = algo.run(
    H=H,
    t=1.0,
    error=1e-8,
    degree=15,
    beta=0.7,
)

print("Status       :", result["status"])
print("Circuit path :", result["circuit_path"])
print("File path    :", result["file_path"])

U_approx = algo.output["Approximate evolution matrix"]
U_exact  = algo.output["Exact evolution matrix"]
frob_err = algo.output["Frobenius norm of error"]

print("\nExact evolution matrix e^{-iHt}:")
print(np.round(U_exact, 6))
print("\nApproximate evolution matrix (QSP):")
print(np.round(U_approx, 6))
print(f"\nFrobenius norm error: {frob_err:.4e}")

# ---------------------------------------------------------------------------
# 2. Parameter sweep over t and degree
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Sweep over evolution time t and polynomial degree")
print("=" * 60)
print(f"{'t':>5}  {'degree':>6}  {'error':>12}  {'status':>6}")
print("-" * 40)

for t in [1.0, 3.0, 5.0]:
    for degree in [10, 20, 40]:
        sweep_algo = QSPHSAlgorithm(text_mode="plain")
        sweep_result = sweep_algo.run(H=H, t=t, error=1e-8, degree=degree, beta=0.7)
        err = sweep_algo.output["Frobenius norm of error"]
        print(f"{t:>5.1f}  {degree:>6d}  {err:>12.4e}  {sweep_result['status']:>6}")

# ---------------------------------------------------------------------------
# 3. Larger 4x4 Hamiltonian (non-power-of-2 auto-padding not needed here)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 4x4 Hamiltonian simulation")
print("=" * 60)

# Random Hermitian matrix for a 2-qubit system
rng = np.random.default_rng(42)
A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
H4 = (A + A.conj().T) / 2  # symmetrize → Hermitian

algo4 = QSPHSAlgorithm(text_mode="plain")
result4 = algo4.run(H=H4, t=2.0, error=1e-6, degree=20, beta=0.7)

frob4 = algo4.output["Frobenius norm of error"]
print(f"Status: {result4['status']},  Frobenius error: {frob4:.4e}")
