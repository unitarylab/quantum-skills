"""
Linear Combination of Unitaries (LCU) Algorithm Example using UnitaryLab
=========================================================================
Demonstrates implementing non-unitary matrices as probabilistic applications
of unitary operations via the Linear Combination of Unitaries (LCU) method.

Theory recap
------------
LCU solves the problem: given a non-unitary matrix M = Σⱼ αⱼ Uⱼ where Uⱼ are 
unitary operators with non-negative coefficients αⱼ ≥ 0, probabilistically apply M 
to a quantum state by:

  1. Encode coefficients in auxiliary qubits using the V operator
  2. Conditionally apply each unitary Uⱼ using the SELECT operator
  3. Uncompute ancillas with V† and post-select on ancilla = |0⟩

The success probability is P_success = (Σⱼ αⱼ)² = s², where s normalizes the 
operation. LCU is essential for:
  - Foundation of HHL algorithm (linear systems)
  - Hamiltonian simulation via sum decomposition
  - Quantum machine learning block-encodings
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from engine import GateSequence
from engine.algorithms.linear_algebra import LCUAlgorithm


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def print_separator(title: str = "") -> None:
    width = 60
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def report_result(result: dict, name: str) -> None:
    """Print a structured summary of an LCU result."""
    print(result["plot"])
    print(f"  LCU Success Probability: {result['lcu_success_probability']:.6f}")
    print(f"  Circuit diagram         : {result['circuit_path']}")


# ---------------------------------------------------------------------------
# Example 1 — Simple coefficient encoding: M = 0.6*I + 0.4*X
# ---------------------------------------------------------------------------

def example_identity_plus_pauli(algo: LCUAlgorithm, output_dir: str) -> dict:
    """
    Implement M = 0.6*I + 0.4*X on a single qubit.

    Here:
      - U₀ = I (identity)
      - U₁ = X (Pauli-X flipper)
      - α₀ = 0.6, α₁ = 0.4

    LCU normalizes: s = 0.6 + 0.4 = 1.0
    Expected success probability: s² = 1.0
    """
    print_separator("Example 1 — M = 0.6*I + 0.4*X")

    # Step 1: Define unitaries
    U0 = GateSequence(1)
    U0.i(0)  # Identity

    U1 = GateSequence(1)
    U1.x(0)  # Pauli-X

    unitaries = [U0, U1]

    # Step 2: Define coefficients
    alphas = [0.6, 0.4]
    s_norm = np.sum(alphas)

    print(f"  Unitaries: U₀ = I, U₁ = X")
    print(f"  Coefficients: α₀ = {alphas[0]}, α₁ = {alphas[1]}")
    print(f"  Normalization: s = Σαⱼ = {s_norm:.4f}")

    # Step 3: Run LCU
    result = algo.run(
        alphas=alphas,
        unitaries=unitaries,
        n_sys=1,
        initial_state=None,
        backend="torch",
        algo_dir=os.path.join(output_dir, "identity_plus_pauli"),
    )

    report_result(result, "I + X")
    print(f"  Expected P_success     : {s_norm**2:.6f} (s² = {s_norm}²)")

    return result


# ---------------------------------------------------------------------------
# Example 2 — More complex: M = 0.5*I + 0.3*Z + 0.2*X
# ---------------------------------------------------------------------------

def example_three_unitaries(algo: LCUAlgorithm, output_dir: str) -> dict:
    """
    Implement M = 0.5*I + 0.3*Z + 0.2*X on a single qubit.

    Here:
      - U₀ = I (identity)
      - U₁ = Z (Pauli-Z)
      - U₂ = X (Pauli-X)
      - α₀ = 0.5, α₁ = 0.3, α₂ = 0.2

    Requires 2 auxiliary qubits to store 3 indices (⌈log₂(3)⌉ = 2).
    Success probability: s² = (0.5 + 0.3 + 0.2)² = 1.0
    """
    print_separator("Example 2 — M = 0.5*I + 0.3*Z + 0.2*X")

    # Step 1: Define three Pauli unitaries
    U0 = GateSequence(1)
    U0.i(0)  # Identity

    U1 = GateSequence(1)
    U1.z(0)  # Pauli-Z

    U2 = GateSequence(1)
    U2.x(0)  # Pauli-X

    unitaries = [U0, U1, U2]

    # Step 2: Define coefficients
    alphas = [0.5, 0.3, 0.2]
    s_norm = np.sum(alphas)

    print(f"  Unitaries: U₀ = I, U₁ = Z, U₂ = X")
    print(f"  Coefficients: α₀ = {alphas[0]}, α₁ = {alphas[1]}, α₂ = {alphas[2]}")
    print(f"  Normalization: s = Σαⱼ = {s_norm:.4f}")
    print(f"  Auxiliary qubits needed: ⌈log₂(3)⌉ = 2")

    # Step 3: Run LCU
    result = algo.run(
        alphas=alphas,
        unitaries=unitaries,
        n_sys=1,
        initial_state=None,
        backend="torch",
        algo_dir=os.path.join(output_dir, "three_unitaries"),
    )

    report_result(result, "I + Z + X")
    print(f"  Expected P_success     : {s_norm**2:.6f} (s² = {s_norm}²)")

    return result


# ---------------------------------------------------------------------------
# Example 3 — With initial state preparation
# ---------------------------------------------------------------------------

def example_with_initial_state(algo: LCUAlgorithm, output_dir: str) -> dict:
    """
    Implement M = 0.6*I + 0.4*Y with custom initial state |+⟩ = (|0⟩ + |1⟩)/√2.

    Initial state |+⟩ is maximally superposed, allowing us to observe
    the full effect of the linear combination across the superposition.
    """
    print_separator("Example 3 — M = 0.6*I + 0.4*Y, Initial State |+⟩")

    # Step 1: Define unitaries
    U0 = GateSequence(1)
    U0.i(0)  # Identity

    U1 = GateSequence(1)
    U1.y(0)  # Pauli-Y

    unitaries = [U0, U1]

    # Step 2: Define coefficients
    alphas = [0.6, 0.4]
    s_norm = np.sum(alphas)

    # Step 3: Prepare initial state |+⟩
    initial_state = GateSequence(1)
    initial_state.h(0)  # Hadamard creates |+⟩

    print(f"  Unitaries: U₀ = I, U₁ = Y")
    print(f"  Coefficients: α₀ = {alphas[0]}, α₁ = {alphas[1]}")
    print(f"  Initial state: H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"  Normalization: s = Σαⱼ = {s_norm:.4f}")

    # Step 4: Run LCU with initial state
    result = algo.run(
        alphas=alphas,
        unitaries=unitaries,
        n_sys=1,
        initial_state=initial_state,
        backend="torch",
        algo_dir=os.path.join(output_dir, "with_initial_state"),
    )

    report_result(result, "I + Y with |+⟩")
    print(f"  Expected P_success     : {s_norm**2:.6f} (s² = {s_norm}²)")

    return result


# ---------------------------------------------------------------------------
# Example 4 — Equal superposition (all coefficients the same)
# ---------------------------------------------------------------------------

def example_equal_coefficients(algo: LCUAlgorithm, output_dir: str) -> dict:
    """
    Implement M = (1/4)*(I + X + Y + Z), equal superposition of all Pauli matrices.

    This is a benchmark case where all unitaries have equal weight.
    Requires 2 auxiliary qubits (⌈log₂(4)⌉ = 2).
    Success probability: s² = 1.0 (since 4 * 0.25 = 1.0)
    """
    print_separator("Example 4 — M = 0.25*(I + X + Y + Z), Equal Superposition")

    # Step 1: Define all four single-qubit Paulis
    U0 = GateSequence(1)
    U0.i(0)  # Identity

    U1 = GateSequence(1)
    U1.x(0)  # Pauli-X

    U2 = GateSequence(1)
    U2.y(0)  # Pauli-Y

    U3 = GateSequence(1)
    U3.z(0)  # Pauli-Z

    unitaries = [U0, U1, U2, U3]

    # Step 2: Define equal coefficients
    alphas = [0.25, 0.25, 0.25, 0.25]
    s_norm = np.sum(alphas)

    print(f"  Unitaries: U₀ = I, U₁ = X, U₂ = Y, U₃ = Z")
    print(f"  Coefficients: all αⱼ = 0.25 (equal superposition)")
    print(f"  Normalization: s = Σαⱼ = {s_norm:.4f}")
    print(f"  Auxiliary qubits needed: ⌈log₂(4)⌉ = 2")

    # Step 3: Run LCU
    result = algo.run(
        alphas=alphas,
        unitaries=unitaries,
        n_sys=1,
        initial_state=None,
        backend="torch",
        algo_dir=os.path.join(output_dir, "equal_coefficients"),
    )

    report_result(result, "I + X + Y + Z")
    print(f"  Expected P_success     : {s_norm**2:.6f} (s² = {s_norm}²)")

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = "./lcu_results"
    os.makedirs(output_dir, exist_ok=True)

    algo = LCUAlgorithm()

    print_separator("LCU Algorithm — Example Script")
    print("  Linear Combination of Unitaries (LCU) - Probabilistic Application.")
    print("  Demonstrates encoding non-unitary matrices as superpositions of" )
    print("  unitary operations with classical post-selection.")
    print("  Results (circuit diagrams, etc.) written to:")
    print(f"    {os.path.abspath(output_dir)}")

    results = {}
    results["identity_x"] = example_identity_plus_pauli(algo, output_dir)
    results["three_paulis"] = example_three_unitaries(algo, output_dir)
    results["with_state"] = example_with_initial_state(algo, output_dir)
    results["equal"] = example_equal_coefficients(algo, output_dir)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print_separator("Summary")
    print(f"  {'Example':<35} {'P_success':>12}")
    print(f"  {'-'*35} {'-'*12}")

    labels = {
        "identity_x": "0.6*I + 0.4*X",
        "three_paulis": "0.5*I + 0.3*Z + 0.2*X",
        "with_state": "0.6*I + 0.4*Y (with |+⟩)",
        "equal": "0.25*(I+X+Y+Z)",
    }
    for key, label in labels.items():
        r = results[key]
        p = r["lcu_success_probability"]
        status = "OK" if p > 0.5 else "LOW"
        print(f"  {label:<35} {p:>12.6f}  [{status}]")

    print()


if __name__ == "__main__":
    main()
