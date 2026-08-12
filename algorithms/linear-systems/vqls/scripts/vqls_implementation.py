# -*- coding: utf-8 -*-
"""Reference implementation of the VQLS contract documented by SKILL.md.

This script builds the algorithm from its basic components rather than
delegating to ``unitarylab_algorithms``: Pauli decomposition, a hardware-
efficient ansatz, the general ``U_b`` preparation circuit, three cost paths,
COBYLA optimisation, post-processing, and artifact export.

VQLS Algorithm — Variational Quantum Linear Solver.

Features:
  - j=-1 shortcut: U_b†→controlled-Z→U_b is skipped for psi_norm.
  - A local-cost threshold flag derived from epsilon; COBYLA is not stopped.
  - U_b construction with a uniform-state fast path or Gram–Schmidt completion.
  - Defaults: n_layers=4, maxiter=500.

Accepts caller-provided square matrices with power-of-two dimensions, subject
to the validation boundaries and numerical limitations documented by SKILL.md:
  - Pauli decomposition via ``unitarylab.library.pauli_operator``
  - Hardware-efficient ansatz with ring CNOT entanglement
  - Three selectable cost-function methods
  - COBYLA classical optimisation
"""

import os
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize

try:
    from unitarylab.library.pauli_operator import pauli_string_decomposition
    from unitarylab.core import Circuit, Register
    from unitarylab.transpiler.unroll import Unroll
    from unitarylab.transpiler.basis import DEFAULT_BASIS
    from unitarylab_algorithms.algo_base import BaseAlgorithm
except ImportError as exc:
    raise ImportError(
        "This VQLS implementation requires the project 'unitarylab' and "
        "'unitarylab_algorithms' packages with their circuit, Pauli, "
        "transpiler, and BaseAlgorithm modules available."
    ) from exc

# ============================================================================
# 0. Pauli constants & helpers
# ============================================================================

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MAT = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _pauli_string_to_matrix(label: str) -> np.ndarray:
    mats = [_PAULI_MAT[ch] for ch in reversed(label)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# ============================================================================
# 1. State helpers
# ============================================================================

def _fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, complex).ravel()
    b = np.asarray(b, complex).ravel()
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) ** 2)


def _zero_state(n_qubits: int) -> np.ndarray:
    s = np.zeros(1 << n_qubits, dtype=complex)
    s[0] = 1.0 + 0.0j
    return s


def _ancilla_expval_z(state: np.ndarray, ancilla_idx: int) -> float:
    """⟨Z⟩ on the ancilla qubit via direct probability-vector arithmetic.

    Complexity O(2ⁿ) with no intermediate matrix construction.
    """
    probs = np.square(np.abs(state))
    indices = np.arange(len(state), dtype=np.int64)
    signs = np.where((indices >> ancilla_idx) & 1, -1.0, 1.0)
    return float(signs @ probs)


def _decompose_unitary(qc: Circuit) -> Circuit:
    """Decompose ``unitary`` gates into primitive gates (CX, RY, RZ, H).

    Uses the available project transpiler. If it reports that a decomposition
    is unsupported, retain the unitary gate so the mathematical operation is
    unchanged.
    """
    unroller = Unroll(gates_to_unroll="unitary", basis=DEFAULT_BASIS)
    try:
        return unroller.run(qc)
    except NotImplementedError:
        return qc


# ============================================================================
# 2. U_b circuit
# ============================================================================

def _build_Ub(b_state: np.ndarray, n_qubits: int,
              dagger: bool = False) -> Circuit:
    r"""Build a circuit whose first column equals |b> (or <b| if dagger).

    - Uniform |b>  : fast path  H^{\otimes n}.
    - General  |b>  : Gram-Schmidt completion, then decompose to primitives.
    """
    b = np.asarray(b_state, complex).ravel()
    name = "U_b_dagger" if dagger else "U_b"

    uniform = np.ones(1 << n_qubits, complex) / np.sqrt(1 << n_qubits)
    if np.allclose(b, uniform, atol=1e-10):
        reg = Register("q", n_qubits)
        qc = Circuit(reg, name=name)
        for q in range(n_qubits):
            qc.h(q)
        return qc

    # Gram-Schmidt: construct a full unitary whose first column is |b⟩
    v = b / np.linalg.norm(b)
    dim = len(v)
    U = np.zeros((dim, dim), dtype=complex)
    U[:, 0] = v
    col = 1
    for k in range(dim):
        if col >= dim:
            break
        w = np.zeros(dim, dtype=complex)
        w[k] = 1.0
        for j in range(col):
            w -= np.vdot(U[:, j], w) * U[:, j]
        nw = np.linalg.norm(w)
        if nw > 1e-12:
            U[:, col] = w / nw
            col += 1
    if col != dim:
        raise RuntimeError("Failed to complete unitary basis for U_b.")

    mat = U.conj().T if dagger else U
    reg = Register("q", n_qubits)
    qc = Circuit(reg, name=name)
    qc.unitary(matrix=mat, target=list(range(n_qubits)))
    return _decompose_unitary(qc)


# ============================================================================
# 3. Ansatz
# ============================================================================

def _ansatz_num_params(n_qubits: int, n_layers: int = 4) -> int:
    return 2 * n_qubits * n_layers


def _build_ansatz(theta: np.ndarray, n_qubits: int, n_layers: int = 4,
                  entangle: bool = True) -> Circuit:
    """Hardware-efficient ansatz: H → [RY-RZ-CNOT_ring] × n_layers.

    Each layer applies RY, RZ on every qubit followed by a ring of CNOTs
    that entangle adjacent qubits (q0→q1→...→q_{n-1}→q0).
    """
    theta = np.asarray(theta, float).ravel()
    p_needed = _ansatz_num_params(n_qubits, n_layers)
    if len(theta) < p_needed:
        theta = np.pad(theta, (0, p_needed - len(theta)), mode="constant")
    theta = theta[:p_needed]

    reg = Register("q", n_qubits)
    qc = Circuit(reg, name="V(theta)")

    # Initial superposition — start from a balanced state for better coverage
    for q in range(n_qubits):
        qc.h(q)

    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(theta[idx]), q)
            idx += 1
        for q in range(n_qubits):
            qc.rz(float(theta[idx]), q)
            idx += 1
        if entangle and n_qubits >= 2:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(n_qubits - 1, 0)          # ring CNOT
    return qc


def _ansatz_state(theta: np.ndarray, n_qubits: int, n_layers: int = 4,
                  entangle: bool = True) -> np.ndarray:
    qc = _build_ansatz(theta, n_qubits, n_layers, entangle)
    U = qc.get_matrix()
    x = U @ _zero_state(n_qubits)
    return x / np.linalg.norm(x)


# ============================================================================
# 4. Controlled Pauli-string helper
# ============================================================================

def _apply_controlled_pauli_string(qc: Circuit, label: str, n_sys: int,
                                   control: int) -> None:
    """Append controlled Pauli gates for each non-identity character of *label*.

    ``label[pos]∈{I,X,Y,Z}``; ``control`` is the ancilla qubit index.
    """
    if len(label) != n_sys:
        raise ValueError(
            f"Pauli label length ({len(label)}) != n_sys ({n_sys})")
    for pos, p in enumerate(label):
        if p == "I":
            continue
        elif p == "X":
            qc.cx(control, pos)
        elif p == "Y":
            qc.cy(control, pos)
        elif p == "Z":
            qc.cz(control, pos)
        else:
            raise ValueError(f"Invalid Pauli char '{p}'")


# ============================================================================
# 5. Hadamard-test circuit builder (PS-encoded C-Z_j)
# ============================================================================

def _build_ht_circuit(
    theta: np.ndarray, pauli_labels: List[str], n_sys: int,
    l: int, lp: int, z_angles: List[float], part: str,
    n_layers: int = 4, entangle: bool = True,
    ub_circuit: Optional[Circuit] = None,
    ub_dag_circuit: Optional[Circuit] = None,
) -> Circuit:
    """Build ONE Hadamard-test circuit with parameterised C-Phase gates.

    Instead of individual CZ(anc, j) we place a **controlled PhaseShift** on
    every system qubit.  The ``z_angles`` list encodes which qubit is active:
        z_angles[j] = π   →  PS(π) = Z  (qubit j measured)
        z_angles[j] = 0   →  PS(0) = I  (qubit j skipped)

    For the psi-norm term (all qubits inactive) pass all zeros.
    """
    anc = n_sys
    total = n_sys + 1
    reg = Register("q", total)
    qc = Circuit(reg, name=f"HT_l{l}_lp{lp}_{part}")

    # --- ancilla preamble ---
    qc.h(anc)
    if part.lower() == "im":
        qc.sdag(anc)

    # --- ansatz V(θ) ---
    ans_qc = _build_ansatz(theta, n_sys, n_layers, entangle)
    qc.append(ans_qc, list(range(n_sys)))

    # --- C-A_l ---
    _apply_controlled_pauli_string(qc, pauli_labels[l], n_sys, anc)

    # --- U_b† (skip for psi_norm: j=-1 → all z_angles=0, no Z needed) ---
    need_ub = any(abs(a) > 1e-9 for a in z_angles)
    if need_ub and ub_dag_circuit is not None:
        qc.append(ub_dag_circuit, list(range(n_sys)))

    # --- C-Z_j via PS-encoding ---
    if need_ub:
        for q, angle in enumerate(z_angles):
            if abs(angle) > 1e-9:
                # controlled PhaseShift(π) ≡ CZ (up to global phase on ancilla)
                qc.cz(anc, q) if abs(
                    angle - np.pi) < 1e-9 else qc.cp(angle, anc, q)

    # --- U_b ---
    if need_ub and ub_circuit is not None:
        qc.append(ub_circuit, list(range(n_sys)))

    # --- C-A_lp† (Pauli strings are self-inverse: A_lp† = A_lp) ---
    _apply_controlled_pauli_string(qc, pauli_labels[lp], n_sys, anc)

    # --- ancilla postamble ---
    qc.h(anc)

    return qc


# ============================================================================
# 6. Cost functions
# ============================================================================

def _cost_global(theta: np.ndarray, A: np.ndarray, b: np.ndarray,
                 n_qubits: int, n_layers: int, entangle: bool) -> float:
    """Global cost: C_G = 1 - |⟨b| Ax/‖Ax‖ ⟩|²."""
    x = _ansatz_state(theta, n_qubits, n_layers, entangle)
    Ax = A @ x
    nrm = np.linalg.norm(Ax)
    if nrm < 1e-12:
        return 1.0
    return float(1.0 - abs(np.vdot(b, Ax / nrm)) ** 2)


def _make_cost_local_classical(
    coeffs: np.ndarray, pauli_labels: List[str], b_state: np.ndarray,
    n_qubits: int, n_layers: int, entangle: bool,
) -> Callable[[np.ndarray], float]:
    """Local cost evaluated via classical matrix arithmetic (fast)."""
    dim = 1 << n_qubits
    L = len(coeffs)

    P_mats = np.array([_pauli_string_to_matrix(lbl) for lbl in pauli_labels])
    Ub_mat = _build_Ub(b_state, n_qubits, dagger=False).get_matrix()
    Ub_dag_mat = Ub_mat.conj().T

    Z_mats = np.zeros((n_qubits, dim, dim), dtype=complex)
    for q in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[q] = "Z"
        Z_mats[q] = _pauli_string_to_matrix("".join(chars))

    def cost_local_classical(theta: np.ndarray) -> float:
        x = _ansatz_state(theta, n_qubits, n_layers, entangle)
        mu_sum = 0.0 + 0.0j
        psi_norm = 0.0 + 0.0j

        for l in range(L):
            y_base = P_mats[l] @ x
            for lp in range(L):
                cp = coeffs[l] * np.conj(coeffs[lp])
                y = Ub_dag_mat @ y_base
                y = Ub_mat @ y
                y = P_mats[lp] @ y
                psi_norm += cp * np.vdot(x, y)
                for j in range(n_qubits):
                    y = Ub_dag_mat @ y_base
                    y = Z_mats[j] @ y
                    y = Ub_mat @ y
                    y = P_mats[lp] @ y
                    mu_sum += cp * np.vdot(x, y)

        pn = abs(psi_norm)
        if pn < 1e-12:
            return 1.0
        return float(np.real(0.5 - 0.5 * abs(mu_sum) / (n_qubits * pn)))

    return cost_local_classical


class _HTCircuitState:
    """Bookkeeping for local_ht representative circuits (for export)."""

    def __init__(self):
        self.circuits: Dict[str, Circuit] = {}
        self._build_rep: Optional[Callable[[np.ndarray], None]] = None


def _make_cost_local_ht(
    coeffs: np.ndarray, pauli_labels: List[str], b_state: np.ndarray,
    n_sys: int, n_layers: int, entangle: bool, state: _HTCircuitState,
    backend: str = 'torch', device: str = 'cpu', dtype: np.dtype = np.complex128
) -> Callable[[np.ndarray], float]:
    """Local cost via full Hadamard-test circuits (PS-encoded C-Z_j).

    For psi_norm (all z angles zero), the U_b† → controlled-Z → U_b segment
    is skipped. Circuits are rebuilt for the current theta on each objective
    evaluation.
    """
    anc = n_sys
    L = len(coeffs)

    _ub = _build_Ub(b_state, n_sys, dagger=False)
    _ub_dag = _build_Ub(b_state, n_sys, dagger=True)

    # Find representative indices for later display
    non_id = [(i, lbl) for i, lbl in enumerate(pauli_labels)
              if any(ch != "I" for ch in lbl)]
    if len(non_id) >= 2:
        rep_l, rep_label = non_id[0]
        rep_lp = non_id[1][0]
    elif len(non_id) == 1:
        rep_lp = non_id[0][0]
        rep_l = 0
        rep_label = pauli_labels[0]
    else:
        rep_l = rep_lp = 0
        rep_label = pauli_labels[0]

    rep_j = -1
    for pos, ch in enumerate(rep_label):
        if ch == "Z":
            rep_j = pos
            break
    if rep_j == -1:
        for pos, ch in enumerate(rep_label):
            if ch != "I":
                rep_j = pos
                break
    if rep_j == -1:
        rep_j = 0

    # Keep the documented setup circuits as representative construction
    # artifacts. Execution below rebuilds circuits for each current theta.
    z_angles_psi_norm = [0.0] * n_sys
    state.circuits["psi_norm_setup_re"] = _build_ht_circuit(
        np.zeros(_ansatz_num_params(n_sys, n_layers)),
        pauli_labels, n_sys, 0, 0, z_angles_psi_norm, "Re",
        n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
    state.circuits["psi_norm_setup_im"] = _build_ht_circuit(
        np.zeros(_ansatz_num_params(n_sys, n_layers)),
        pauli_labels, n_sys, 0, 0, z_angles_psi_norm, "Im",
        n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)

    def _exec_psi_norm(theta: np.ndarray, l: int, lp: int) -> complex:
        """Rebuild and execute psi_norm circuits for current theta."""
        qc_re = _build_ht_circuit(
            theta, pauli_labels, n_sys, l, lp, z_angles_psi_norm, "Re",
            n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
        qc_im = _build_ht_circuit(
            theta, pauli_labels, n_sys, l, lp, z_angles_psi_norm, "Im",
            n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
        re = _ancilla_expval_z(
            qc_re.execute(backend=backend, device=device, dtype=dtype).state, anc)
        im = _ancilla_expval_z(
            qc_im.execute(backend=backend, device=device, dtype=dtype).state, anc)
        return re + 1.0j * im

    def _exec_mu_j(theta: np.ndarray, l: int, lp: int, j: int) -> complex:
        """Execute µ(l,lp,j) — one qubit j active, others inactive."""
        z_angles = [0.0] * n_sys
        z_angles[j] = np.pi   # PS(π) = Z on qubit j
        qc_re = _build_ht_circuit(
            theta, pauli_labels, n_sys, l, lp, z_angles, "Re",
            n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
        qc_im = _build_ht_circuit(
            theta, pauli_labels, n_sys, l, lp, z_angles, "Im",
            n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
        re = _ancilla_expval_z(
            qc_re.execute(backend=backend, device=device, dtype=dtype).state, anc)
        im = _ancilla_expval_z(
            qc_im.execute(backend=backend, device=device, dtype=dtype).state, anc)
        return re + 1.0j * im

    def cost_local_ht(theta: np.ndarray) -> float:
        mu_sum = 0.0 + 0.0j
        psi_norm = 0.0 + 0.0j

        for l in range(L):
            for lp in range(L):
                cp = coeffs[l] * np.conj(coeffs[lp])

                # psi_norm — j=-1 path: no Z on any qubit
                psi_norm += cp * _exec_psi_norm(theta, l, lp)

                # µ(l,lp,j) — rebuild the current-theta circuit, z_angle[j]=π
                for j in range(n_sys):
                    mu_sum += cp * _exec_mu_j(theta, l, lp, j)

        pn = abs(psi_norm)
        if pn < 1e-12:
            return 1.0
        return float(np.real(0.5 - 0.5 * abs(mu_sum) / (n_sys * pn)))

    def _build_rep_circuits(theta_opt: np.ndarray):
        z_angles_rep = [0.0] * n_sys
        if rep_j >= 0:
            z_angles_rep[rep_j] = np.pi
        state.circuits["hadamard_test_re"] = _build_ht_circuit(
            theta_opt, pauli_labels, n_sys, rep_l, rep_lp, z_angles_rep, "Re",
            n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
        state.circuits["hadamard_test_im"] = _build_ht_circuit(
            theta_opt, pauli_labels, n_sys, rep_l, rep_lp, z_angles_rep, "Im",
            n_layers, entangle, ub_circuit=_ub, ub_dag_circuit=_ub_dag)
        state.circuits["ansatz"] = _build_ansatz(
            theta_opt, n_sys, n_layers, entangle)
        state.circuits["Ub"] = _ub
        state.circuits["Ub_dag"] = _ub_dag

    state._build_rep = _build_rep_circuits
    return cost_local_ht


# ============================================================================
# 7. VQLSAlgorithm (BaseAlgorithm wrapper)
# ============================================================================

class VQLSAlgorithm(BaseAlgorithm):
    """Variational Quantum Linear Solver.

    Solve  A·x = b  using a hybrid quantum-classical variational approach.

    Three cost-function modes
    -------------------------
    * ``"local_ht"`` — Hadamard-test circuits with PS-encoded C-Z_j.
      Representative circuits are retained for export; objective circuits are
      rebuilt for the current parameter vector.
    * ``"local_classical"`` — classical matrix arithmetic, fast.
    * ``"global"`` — C_G = 1 - |⟨b|Ax/‖Ax‖⟩|², simplest.

    Parameters
    ----------
    text_mode : str
        ``"plain"`` for clean output, ``"legacy"`` for verbose log style.
    algo_dir : str, optional
        Output directory for results / circuit SVGs.
    """

    def __init__(self, text_mode: str = "plain", algo_dir: str = None):
        if algo_dir is None:
            algo_dir = os.path.join(os.getcwd(), "results", "vqls")
        os.makedirs(algo_dir, exist_ok=True)
        super().__init__(name="VQLS Algorithm", prefix="VQLS",
                         text_mode=text_mode, algo_dir=algo_dir)

    def run(
        self,
        A: np.ndarray,
        b: np.ndarray,
        cost_function: str = "local_ht",
        n_layers: int = 4,
        maxiter: int = 500,
        tol: float = 1e-6,
        seed: int = 42,
        epsilon: Optional[float] = None, backend='torch', device='cpu', dtype=np.complex128
    ) -> Dict[str, Any]:
        """Execute the VQLS algorithm.

        Parameters
        ----------
        A : ndarray (2ⁿ, 2ⁿ)
            Coefficient matrix.
        b : ndarray (2ⁿ)
            Right-hand-side vector.
        cost_function : str
            ``"local_ht"`` / ``"local_classical"`` / ``"global"``.
        n_layers : int
            Number of ansatz layers (default 4).
        maxiter : int
            COBYLA maximum iterations (default 500).
        tol : float
            COBYLA convergence tolerance (default 1e-6).
        seed : int
            Random seed for initial parameters (default 42).
        epsilon : float, optional
            Local-cost threshold parameter. When set, the implementation
            records whether C_L ≤ (1/n)(ε/κ)²; COBYLA is not interrupted.
            Only used for ``local_ht`` and ``local_classical``.

        Returns
        -------
        dict
            Standard BaseAlgorithm result dict with keys:
            - status, circuit_path, plot, circuit
            - Fidelity, Ax Fidelity, Cost Function, Condition Number,
              Solution State (Quantum/Classical), Computation Time (s),
              Cost History
        """
        input_params = {
            "Matrix A shape": A.shape,
            "b norm": float(np.linalg.norm(b)),
            "Cost function": cost_function,
            "Ansatz layers": n_layers,
            "Max iterations": maxiter,
            "Tolerance": tol,
            "Seed": seed,
            "Epsilon threshold": epsilon,
        }
        self.update_input(input_params)

        start_time = time.time()
        entangle = True

        # ---- Validation ----
        A = np.asarray(A, dtype=complex)
        b = np.asarray(b, dtype=complex).ravel()
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")
        dim = A.shape[0]
        n_qubits = int(np.log2(dim))
        if 1 << n_qubits != dim:
            raise ValueError("A dimension must be a power of 2.")
        if b.shape[0] != dim:
            raise ValueError(f"b length ({b.shape[0]}) != A dim ({dim}).")

        valid_cost = {"local_ht", "local_classical", "global"}
        if cost_function not in valid_cost:
            raise ValueError(f"cost_function must be in {valid_cost}")

        bn = np.linalg.norm(b)
        if bn < 1e-12:
            raise ValueError("b is zero.")
        b_state = b / bn
        kappa = float(np.linalg.cond(A))

        # ---- Step 1: Pauli decomposition ----
        self.log("Stage 1/5: Pauli decomposition")
        is_real_sym = np.allclose(A, A.T) and np.allclose(A.imag, 0)
        terms = pauli_string_decomposition(
            A, partition_commuting=True, real_symmetric_hint=is_real_sym)
        pauli_labels = [t[0] for t in terms]
        coeffs = np.array([t[1] for t in terms], dtype=complex)
        self.log(f"  n_qubits={n_qubits}, dim={dim}, Pauli terms={len(terms)}")
        self.log("Stage 1/5 completed")

        # ---- Step 2: Ansatz & cost init ----
        self.log("Stage 2/5: Ansatz & cost-function setup")
        rng = np.random.default_rng(seed)
        n_params = _ansatz_num_params(n_qubits, n_layers)
        init_theta = rng.uniform(-0.5, 0.5, size=n_params)
        ht_state = _HTCircuitState() if cost_function == "local_ht" else None

        if cost_function == "global":
            def _raw_cost(t):
                return _cost_global(t, A, b_state, n_qubits, n_layers, entangle)
        elif cost_function == "local_classical":
            _raw_cost = _make_cost_local_classical(
                coeffs, pauli_labels, b_state, n_qubits, n_layers, entangle)
        elif cost_function == "local_ht":
            _raw_cost = _make_cost_local_ht(
                coeffs, pauli_labels, b_state, n_qubits, n_layers, entangle,
                ht_state, backend=backend, device=device, dtype=dtype)

        c0 = _raw_cost(init_theta)
        self.log(
            f"  n_layers={n_layers}, params={n_params}, "
            f"cost_function={cost_function}")
        self.log(f"  Init cost = {c0:.6e}")

        # Record the local-cost threshold flag; this does not interrupt COBYLA.
        gamma_stop = None
        if epsilon is not None and cost_function != "global":
            gamma_stop = (1.0 / n_qubits) * (float(epsilon) / kappa) ** 2
            self.log(f"  Stop threshold γ = {gamma_stop:.3e}  "
                     f"(ε={epsilon}, κ={kappa:.1f})")

        self.log("Stage 2/5 completed")

        # ---- Step 3: COBYLA with threshold recording ----
        self.log("Stage 3/5: COBYLA optimisation")
        cost_history: List[float] = []
        _early_stop_flag = False

        def _cost_tracked(t):
            nonlocal _early_stop_flag
            c = _raw_cost(t)
            cost_history.append(c)
            if gamma_stop is not None and c <= gamma_stop:
                _early_stop_flag = True
            return c

        result = minimize(
            _cost_tracked, init_theta, method="COBYLA",
            options={"maxiter": maxiter, "tol": tol, "rhobeg": 0.5})
        theta_opt = result.x
        final_cost = float(result.fun)
        sim_time = time.time() - start_time
        self.log(f"  success={result.success}, nfev={result.nfev}, "
                 f"final cost={final_cost:.6e}")
        if _early_stop_flag:
            self.log(
                "  Local-cost threshold reached; COBYLA was not interrupted."
            )
        self.log("Stage 3/5 completed")

        # ---- Step 4: Post-processing ----
        self.log("Stage 4/5: Post-processing")
        x_quantum = _ansatz_state(theta_opt, n_qubits, n_layers, entangle)

        try:
            x_cl = np.linalg.solve(A, b_state)
            x_cl /= np.linalg.norm(x_cl)
            fid = _fidelity(x_cl, x_quantum)
        except np.linalg.LinAlgError:
            x_cl = None
            fid = None

        Ax = A @ x_quantum
        nAx = np.linalg.norm(Ax)
        Ax_norm = Ax / nAx if nAx > 1e-12 else Ax
        ax_fid = _fidelity(b_state, Ax_norm)

        if fid is not None:
            self.log(f"  fidelity (x)  = {fid:.6f}")
        self.log(f"  fidelity (Ax) = {ax_fid:.6f}")
        self.log(f"  cond(A)       = {kappa:.2f}")
        self.log("Stage 4/5 completed")

        output = {
            "Fidelity": fid,
            "Ax Fidelity": ax_fid,
            "Cost Function": cost_function,
            "Condition Number": kappa,
            "Solution State (Quantum)": x_quantum,
            "Solution State (Classical)": x_cl,
            "Computation Time (s)": sim_time,
            "Cost History": cost_history,
            "Early Stopped": _early_stop_flag,
        }
        self.update_output(output)
        self.status = "success" if result.success else "failed"
        if fid is not None:
            self.summary = (f"VQLS completed — fidelity={fid:.6f}, "
                            f"Ax_fidelity={ax_fid:.6f}")
        else:
            self.summary = f"VQLS completed — Ax_fidelity={ax_fid:.6f}"

        # ---- Step 5: Circuits & export ----
        self.log("Stage 5/5: Export circuit diagrams")
        if cost_function == "local_ht" and ht_state is not None:
            ht_state._build_rep(theta_opt)

        example_circuit = None
        if ht_state and ht_state.circuits:
            example_circuit = ht_state.circuits.get("hadamard_test_re")
        if example_circuit is None:
            example_circuit = _build_ansatz(
                theta_opt, n_qubits, n_layers, entangle)

        # Decompose block gates so every gate is visible
        example_circuit = example_circuit.decompose(n=2)

        circuit_path = self.save_circuit(example_circuit)
        filename = self.save_txt()
        return self._build_return_dict(True, circuit_path, filename,
                                       example_circuit)


# ============================================================================
# 8. Convenience test function
# ============================================================================

def test(A=None, b=None, cost_function="local_classical", n_layers=4,
         maxiter=20, tol=1e-6, seed=42, epsilon=None,
         backend="torch", device="cpu", dtype=np.complex128,
         text_mode="plain", algo_dir=None):
    """Run the small documented VQLS smoke case.

    Parameters
    ----------
    A : ndarray, optional — 2×2 default if None.
    b : ndarray, optional — [1.0, 0.5] default if None.
    cost_function : str
    n_layers : int
    maxiter : int
    tol : float
    seed : int
    epsilon : float, optional — local-cost threshold parameter.

    Returns
    -------
    dict — Algorithm result dictionary.
    """
    if A is None:
        A = np.array([[1.5, 0.2], [0.2, 1.8]])
    if b is None:
        b = np.array([1.0, 0.5])

    algo = VQLSAlgorithm(text_mode=text_mode, algo_dir=algo_dir)
    result = algo.run(A=A, b=b, cost_function=cost_function,
                      n_layers=n_layers, maxiter=maxiter, tol=tol, seed=seed,
                      epsilon=epsilon, backend=backend, device=device,
                      dtype=dtype)
    return result


if __name__ == "__main__":
    result = test(cost_function="local_classical", maxiter=20)
    print(result["status"])
