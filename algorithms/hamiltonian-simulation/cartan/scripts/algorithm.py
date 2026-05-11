import numpy as np
from unitarylab_algorithms import CartanDecompositionAlgorithm

# ─────────────────────────────────────────────
# Example 1: 2-qubit (4×4) Heisenberg-like Hamiltonian
# H = XX + YY + ZZ  (real symmetric, entries in units where ℏ = 1)
# ─────────────────────────────────────────────
XX = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
], dtype=float)

YY = np.array([
    [ 0, 0, 0, -1],
    [ 0, 0, 1,  0],
    [ 0, 1, 0,  0],
    [-1, 0, 0,  0],
], dtype=float)

ZZ = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1],
], dtype=float)

H = XX + YY + ZZ          # 4×4 real symmetric Hamiltonian

t_evol  = 1.0             # evolution time
epsilon = 1e-3            # target approximation error

print("=" * 55)
print("Cartan Decomposition Hamiltonian Simulation Demo")
print("=" * 55)
print(f"Hamiltonian (4×4 Heisenberg):\n{H}\n")
print(f"Evolution time  : {t_evol}")
print(f"Target error    : {epsilon}\n")

# ─────────────────────────────────────────────
# Instantiate and run the algorithm
# ─────────────────────────────────────────────
algo = CartanDecompositionAlgorithm()

result = algo.run(
    H=H,
    t=t_evol,
    error=epsilon,
    lr=1e-3,
    max_steps=100_000,
    reps=5_000,
)

# ─────────────────────────────────────────────
# Basic result summary
# ─────────────────────────────────────────────
print("─" * 55)
print("Run result")
print("─" * 55)
print(f"  status       : {result['status']}")
print(f"  circuit_path : {result['circuit_path']}")
print(f"  file_path    : {result['file_path']}")

# ─────────────────────────────────────────────
# Detailed output
# ─────────────────────────────────────────────
U_approx   = algo.output["Evolution result"]
U_exact    = algo.output["Exact evolution"]
error_val  = algo.output["Final total error"]
comp_time  = algo.output["Computation time (s)"]

print("\n─" * 28)
print("Detailed output")
print("─" * 55)
print(f"  Final total error  : {error_val:.4e}")
print(f"  Computation time   : {comp_time:.3f} s")

# Frobenius-norm distance between approximate and exact unitary
frob_dist = np.linalg.norm(U_approx - U_exact, ord="fro")
print(f"  ‖U_approx - U_exact‖_F : {frob_dist:.4e}")

print("\nApproximate U(t):")
print(np.round(U_approx, 4))

print("\nExact      U(t) = expm(-iHt):")
print(np.round(U_exact, 4))
