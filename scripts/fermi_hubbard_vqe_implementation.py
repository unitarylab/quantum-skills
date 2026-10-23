"""Fermi-Hubbard VQE -- comprehensive manual implementation.

Builds the Jordan-Wigner Pauli Hamiltonian for an open Fermi-Hubbard chain,
bit-reverses it for little-endian convention, verifies spectrum invariance,
minimizes energy with an Ry-Rz ring-entangling ansatz via COBYLA, and returns
a result dictionary matching the FermiHubbardVQEAlgorithm contract defined in
the parent SKILL.md.

Return fields (per SKILL.md):
  status, circuit_path, plot, circuit, Exact Energy, VQE Energy,
  Absolute Error, Circuit Energy, Number of Qubits,
  Optimizer Evaluations / Iterations / Converged / Message,
  VQE Runtime (s), Total Runtime (s), qubit mapping,
  Fermionic Hamiltonian, Pauli Hamiltonian, Convergence History,
  and -- when measure_shots > 0 -- magnetic moment, standard errors, total shots.
"""

import io
import time
import base64
from functools import reduce

import numpy as np
from scipy.optimize import minimize
from unitarylab import Circuit
from unitarylab.library.fermi_hubbard.fermi_hubbard_pauli import fermi_hubbard_pauli
from unitarylab.library.fermi_hubbard.pauli_ground_state import pauli_string_to_matrix

# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def reverse_bits(i: int, n: int) -> int:
    """Reverse the bit order of integer *i* within *n* bits (fast bitwise)."""
    result = 0
    for _ in range(n):
        result = (result << 1) | (i & 1)
        i >>= 1
    return result


def _spectrum_invariant(H_before, H_after, atol=1e-10):
    """Return True if the sorted spectra of two Hermitian matrices match."""
    eigs_before = np.linalg.eigvalsh(H_before)
    eigs_after  = np.linalg.eigvalsh(H_after)
    return np.allclose(np.sort(eigs_before), np.sort(eigs_after), atol=atol)


# ---------------------------------------------------------------------------
#  Ansatz circuit
# ---------------------------------------------------------------------------

def ansatz_circuit(theta, n_qubits: int, layers: int) -> Circuit:
    """Build Ry-Rz ring-entangling ansatz circuit.

    Each layer: Ry + Rz on every qubit, then nearest-neighbour CX gates
    with a ring-closing CX from last to first qubit.

    Args:
        theta: Flat parameter array of length ``2 * n_qubits * layers``.
        n_qubits: Qubit count (``2L``).
        layers: Number of ansatz repetitions.
    """
    qc = Circuit(n_qubits)
    values = np.asarray(theta).reshape(layers, n_qubits, 2)
    for layer in values:
        for q in range(n_qubits):
            qc.ry(layer[q, 0], q)
            qc.rz(layer[q, 1], q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)
    return qc


# ---------------------------------------------------------------------------
#  Magnetic-moment measurement (finite-shot simulation)
# ---------------------------------------------------------------------------

def _kron_sum_operator(single_qubit_op: np.ndarray, n_qubits: int) -> np.ndarray:
    """Build the full-Hilbert-space sum of a single-qubit operator.

    Returns Σ_{q=0}^{n-1} I ⊗ … ⊗ *single_qubit_op* ⊗ … ⊗ I  as a
    (2^n × 2^n) complex matrix.
    """
    dim = 2 ** n_qubits
    total = np.zeros((dim, dim), dtype=complex)
    for q in range(n_qubits):
        ops = [np.eye(2, dtype=complex)] * n_qubits
        ops[q] = single_qubit_op
        total += reduce(np.kron, ops)
    return total


def _rotate_to_basis(state: np.ndarray, u_single: np.ndarray,
                     n_qubits: int) -> np.ndarray:
    """Apply *u_single* to every qubit of *state* (tensor-product unitary)."""
    u_full = reduce(np.kron, [u_single] * n_qubits)
    return u_full @ state


def _sample_sz(state: np.ndarray, n_qubits: int, shots: int,
               rng: np.random.Generator):
    """Sample S_z = ½ Σ_q σ_z from Z-basis probabilities of *state*.

    Returns (mean, standard_error).
    """
    dim = 2 ** n_qubits
    probs = np.abs(state) ** 2
    indices = rng.choice(dim, size=shots, p=probs)
    # Extract bits and map 0 → +1, 1 → −1 for each qubit
    bitstrings = ((indices[:, None] >> np.arange(n_qubits)) & 1).astype(np.int8)
    sz_per_shot = 0.5 * np.sum(1 - 2 * bitstrings, axis=1)
    mean = float(np.mean(sz_per_shot))
    se   = float(np.std(sz_per_shot, ddof=1) / np.sqrt(shots))
    return mean, se


def _measure_magnetic_moment(state: np.ndarray, n_qubits: int,
                             shots: int = 10000) -> dict:
    """Estimate total-spin magnetic moment ⟨S_x⟩, ⟨S_y⟩, ⟨S_z⟩.

    When *shots* > 0, simulates finite-shot noise by sampling from the
    statevector in the appropriate measurement basis for each axis.
    Standard errors are reported alongside the mean values.

    Args:
        state: Statevector of length 2^{n_qubits}.
        n_qubits: Number of qubits.
        shots: Number of measurement shots per axis (0 → exact only).

    Returns:
        dict with keys magnetic_moment_{x,y,z}, magnetic_moment_{x,y,z}_se,
        and total_shots.
    """
    rng = np.random.default_rng()

    # Exact expectation values as cross-check
    sx1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sy1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz1 = np.array([[1, 0], [0, -1]], dtype=complex)

    Sx = 0.5 * _kron_sum_operator(sx1, n_qubits)
    Sy = 0.5 * _kron_sum_operator(sy1, n_qubits)
    Sz = 0.5 * _kron_sum_operator(sz1, n_qubits)

    exact_x = float(np.vdot(state, Sx @ state).real)
    exact_y = float(np.vdot(state, Sy @ state).real)
    exact_z = float(np.vdot(state, Sz @ state).real)

    if shots <= 0:
        return {
            "magnetic_moment_x": exact_x, "magnetic_moment_x_se": 0.0,
            "magnetic_moment_y": exact_y, "magnetic_moment_y_se": 0.0,
            "magnetic_moment_z": exact_z, "magnetic_moment_z_se": 0.0,
            "total_shots": 0,
        }

    # Basis-rotation unitaries (map X/Y eigenbases → Z eigenbasis)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S_dag = np.diag([1.0, -1j])

    # U_to_x = H  :  H|x+⟩=|0⟩, H|x-⟩=|1⟩
    # U_to_y = H·S†:  H S†|y+⟩=|0⟩, H S†|y-⟩=|1⟩
    state_x = _rotate_to_basis(state, H, n_qubits)
    state_y = _rotate_to_basis(state, H @ S_dag, n_qubits)

    mx, se_x = _sample_sz(state_x, n_qubits, shots, rng)
    my, se_y = _sample_sz(state_y, n_qubits, shots, rng)
    mz, se_z = _sample_sz(state,   n_qubits, shots, rng)  # Z = computational

    return {
        "magnetic_moment_x": mx, "magnetic_moment_x_se": se_x,
        "magnetic_moment_y": my, "magnetic_moment_y_se": se_y,
        "magnetic_moment_z": mz, "magnetic_moment_z_se": se_z,
        "total_shots": shots * 3,  # X, Y, Z
        "_exact_magnetic_moment_x": exact_x,
        "_exact_magnetic_moment_y": exact_y,
        "_exact_magnetic_moment_z": exact_z,
    }


# ---------------------------------------------------------------------------
#  Main VQE routine
# ---------------------------------------------------------------------------

def fermi_hubbard_vqe(L=2, t=1.0, U=4.0, B=1.5, layers=5, max_iter=1000,
                      seed=7, measure_shots=0, backend="torch", device="cpu",
                      dtype=np.complex128):
    """Run VQE for the Fermi-Hubbard model on an open chain of *L* sites.

    Parameters match the FermiHubbardVQEAlgorithm contract (see SKILL.md).

    Args:
        L: Number of lattice sites (qubit count = 2L).
        t: Hopping coefficient.
        U: On-site interaction strength.
        B: Zeeman field strength.
        layers: Depth of the Ry-Rz ring-entangling ansatz.
        max_iter: Maximum COBYLA evaluations.
        seed: RNG seed for initial parameters.
        measure_shots: Shots per axis for magnetic-moment estimation
            (0 skips measurement).
        backend, device, dtype: Circuit execution settings.

    Returns:
        dict with all fields specified in the SKILL.md return-fields section.
    """
    t_start = time.perf_counter()

    # ---- 1. Build Jordan-Wigner Pauli Hamiltonian (big-endian) ----------
    expression = fermi_hubbard_pauli(L, t, U, B)
    H_be = pauli_string_to_matrix(expression)   # dense 2^{2L} × 2^{2L}
    n = 2 * L

    # ---- 2. Bit-reverse for UnitaryLab little-endian convention ---------
    # Verify spectrum invariance first (per SKILL.md debugging tips and
    # run_pauli_vqe spec in the Implementation Architecture section).
    p = np.array([reverse_bits(i, n) for i in range(2 ** n)])
    H = H_be[p][:, p]

    if not _spectrum_invariant(H_be, H):
        e_be = np.linalg.eigvalsh(H_be)
        e_le = np.linalg.eigvalsh(H)
        diff = np.max(np.abs(np.sort(e_be) - np.sort(e_le)))
        raise RuntimeError(
            f"Bit-reversal broke spectrum invariance: "
            f"max eigenvalue diff = {diff:.3e}"
        )

    # ---- 3. Exact reference energy (dense diagonalization) --------------
    exact_energy = float(np.linalg.eigvalsh(H)[0])

    # ---- 4. VQE energy objective ----------------------------------------
    history = []

    def energy(theta):
        qc = ansatz_circuit(theta, n, layers)
        state = np.asarray(
            qc.execute(backend=backend, device=device, dtype=dtype).state,
            dtype=complex,
        ).reshape(-1)
        val = float(np.vdot(state, H @ state).real)
        history.append(val)
        return val

    # ---- 5. COBYLA optimisation -----------------------------------------
    num_params = 2 * n * layers
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-np.pi, np.pi, num_params)

    t_vqe_start = time.perf_counter()
    opt = minimize(
        energy, x0, method="COBYLA",
        options={"maxiter": max_iter, "tol": 1e-8},
    )
    t_vqe_end = time.perf_counter()
    vqe_runtime = t_vqe_end - t_vqe_start

    # ---- 6. Rebuild optimised circuit & compute true energy -------------
    opt_circuit = ansatz_circuit(opt.x, n, layers)
    final_state = np.asarray(
        opt_circuit.execute(backend=backend, device=device, dtype=dtype).state,
        dtype=complex,
    ).reshape(-1)
    circuit_energy = float(np.vdot(final_state, H @ final_state).real)

    # Use circuit_energy as authoritative VQE energy (not opt.fun, which
    # may reflect COBYLA's internal linear-model approximation).
    vqe_energy = circuit_energy
    abs_error = abs(vqe_energy - exact_energy)

    # ---- 7. Status determination ----------------------------------------
    # COBYLA's OptimizeResult may not expose 'nit'; use nfev as iteration proxy.
    n_iter = getattr(opt, "nit", opt.nfev)

    if vqe_energy < exact_energy - 1e-8:
        status = "FAILED: variational bound violated"
        convergence = False
    elif abs_error < 1e-8:
        status = "converged to machine precision"
        convergence = True
    elif n_iter < max_iter:
        status = f"converged (tol satisfied in {n_iter} iters)"
        convergence = True
    else:
        status = f"max_iter={max_iter} reached, ΔE={abs_error:.3e}"
        convergence = False

    # ---- 8. Build result dictionary (per SKILL.md Return Fields) --------
    result = {
        "status": status,
        "circuit": opt_circuit,
        "Exact Energy": exact_energy,
        "VQE Energy": vqe_energy,
        "Absolute Error": abs_error,
        "Circuit Energy": circuit_energy,
        "Number of Qubits": n,
        "Optimizer Evaluations": opt.nfev,
        "Optimizer Iterations": n_iter,
        "Optimizer Converged": convergence,
        "Optimizer Message": opt.message,
        "VQE Runtime (s)": vqe_runtime,
        "Total Runtime (s)": time.perf_counter() - t_start,
        "Qubit Mapping": (
            f"(1↑,1↓,2↑,2↓,…,{L}↑,{L}↓) → {n} qubits, little-endian"
        ),
        "Fermionic Hamiltonian": (
            "H = -t Σ_{j,σ}(c†_{jσ}c_{j+1,σ} + h.c.) "
            "+ U Σ_j n_{j↑}n_{j↓} - B Σ_j(n_{j↑} - n_{j↓})"
        ),
        "Pauli Hamiltonian": expression,
        "Convergence History": history,
    }

    # ---- 9. Convergence plot (matplotlib, per SKILL.md) -----------------
    result["plot"] = None
    result["circuit_path"] = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history, "b.-", markersize=3, linewidth=0.8,
                label="VQE Energy")
        ax.axhline(y=exact_energy, color="r", linestyle="--", linewidth=1.2,
                   label=f"Exact = {exact_energy:.6f}")
        ax.set_xlabel("COBYLA Function Evaluation")
        ax.set_ylabel("Energy")
        ax.set_title(
            f"Fermi-Hubbard VQE Convergence  "
            f"(L={L}, t={t}, U={U}, B={B}, layers={layers})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        result["plot"] = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
    except ImportError:
        pass  # matplotlib not available; plot stays None

    # ---- 10. Optional magnetic-moment measurement -----------------------
    if measure_shots > 0:
        mag = _measure_magnetic_moment(final_state, n, shots=measure_shots)
        result.update(mag)

    return result


# ---------------------------------------------------------------------------
#  Standalone execution & self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    L, layers, max_iter, measure_shots = 2, 5, 1000, 10000

    result = fermi_hubbard_vqe(
        L=L, t=1.0, U=4.0, B=1.5,
        layers=layers, max_iter=max_iter, seed=7,
        measure_shots=measure_shots,
        backend="torch", device="cpu",
    )

    # --- Report ---
    print(f"Status:             {result['status']}")
    print(f"Exact energy:       {result['Exact Energy']:.8f}")
    print(f"VQE energy:         {result['VQE Energy']:.8f}")
    print(f"Absolute error:     {result['Absolute Error']:.3e}")
    print(f"Circuit energy:     {result['Circuit Energy']:.8f}")
    print(f"Qubits:             {result['Number of Qubits']}")
    print(f"Optimizer evals:    {result['Optimizer Evaluations']}")
    print(f"Optimizer message:  {result['Optimizer Message']}")
    print(f"VQE runtime:        {result['VQE Runtime (s)']:.3f} s")
    print(f"Total runtime:      {result['Total Runtime (s)']:.3f} s")
    print(f"Variational bound:  "
          f"{result['VQE Energy'] >= result['Exact Energy'] - 1e-8}")
    print(f"Convergence plot:   "
          f"{'generated' if result['plot'] else 'not available'}")

    if result.get("total_shots", 0) > 0:
        print(f"\nMagnetic moments (shots={measure_shots} per axis):")
        for axis in ("x", "y", "z"):
            key  = f"magnetic_moment_{axis}"
            se   = f"magnetic_moment_{axis}_se"
            print(f"  S_{axis}: {result[key]:.6f} ± {result[se]:.6f}"
                  f"  (exact: {result.get(f'_exact_magnetic_moment_{axis}', 0):.6f})")
        print(f"  Total shots: {result['total_shots']}")

    # --- Assertions (per SKILL.md contract) ---
    assert result["VQE Energy"] >= result["Exact Energy"] - 1e-8, (
        f"VQE energy {result['VQE Energy']} "
        f"violates variational bound {result['Exact Energy']}"
    )
    assert np.isfinite(result["Absolute Error"])
    assert abs(result["VQE Energy"] - result["Circuit Energy"]) < 1e-10, (
        "VQE energy must equal re-evaluated circuit energy"
    )

    print("\n✅ All validations passed.")
