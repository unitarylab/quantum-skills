"""Manual implementation of the Hadamard Test.

The Hadamard Test uses a single ancilla qubit to estimate the complex
expectation value ⟨ψ|U|ψ⟩ for a unitary U and state |ψ⟩. It is a
fundamental subroutine in quantum computing, supporting expectation
value estimation, state overlap testing (swap test), and single-bit
phase estimation.

Three supported modes:
    1. 'expectation': Estimates Re or Im of ⟨ψ|U|ψ⟩ by measuring
       ⟨Z⟩_anc = p(0) - p(1).
    2. 'swap_test': Estimates |⟨φ|ψ⟩|² using a controlled-SWAP circuit.
    3. 'phase_estimation': Runs both real and imaginary Hadamard tests
       to reconstruct the full complex eigenphase φ ∈ [0, 1).

Core circuit (mode='expectation'):
    1. Ancilla qubit |0⟩ → H → (S† for imag).
    2. Optionally prepare |ψ⟩ on target register.
    3. Controlled-U (ancilla controls, target receives U).
    4. H on ancilla.
    5. Measure ancilla: ⟨Z⟩ = p(0) - p(1) = Re(⟨ψ|U|ψ⟩) or Im part.

Components:
    - hadamard_test_circuit: Build single-ancilla Hadamard test circuit
    - build_swap_test_circuit: Build swap test circuit
    - hadamard_test_run: End-to-end Hadamard test pipeline
    - HadamardTestSolver: Class-based interface

Reference:
    SKILL.md — Hadamard Test
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Core Hadamard Test circuit builder
# ---------------------------------------------------------------------------

def hadamard_test_circuit(
    U: Circuit,
    prepare_psi: Optional[Circuit] = None,
    imag: bool = False,
) -> Circuit:
    """Build the single-ancilla Hadamard test circuit.

    Layout:
        - Ancilla: qubit 0
        - Target: qubits 1..1+n-1 where n = U.get_num_qubits()

    Gates applied:
        H(ancilla) → (S†(ancilla) if imag) → prepare_psi(target)
        → controlled-U(target, ctrl=ancilla) → H(ancilla)

    The ancilla measurement encodes the desired expectation value:
        ⟨Z⟩_anc = p(0) - p(1) = Re(⟨ψ|U|ψ⟩)   [imag=False]
        ⟨Z⟩_anc = p(0) - p(1) = Im(⟨ψ|U|ψ⟩)   [imag=True]

    Args:
        U: Unitary operator to measure (acts on target qubits).
        prepare_psi: Optional circuit preparing |ψ⟩ on target.
        imag: If True, insert S† after first H for imaginary part.

    Returns:
        Hadamard test Circuit with 1 + n qubits.

    Raises:
        ValueError: If prepare_psi qubit count doesn't match U.
    """
    n = U.get_num_qubits()
    qc = Circuit(1 + n, name="HadamardTest")
    anc = 0
    tgt = list(range(1, 1 + n))

    # First Hadamard on ancilla
    qc.h(anc)

    # Insert S† for imaginary part
    if imag:
        qc.sdag(anc)

    # Prepare target state |ψ⟩
    if prepare_psi is not None:
        if prepare_psi.get_num_qubits() != n:
            raise ValueError(
                f"prepare_psi has {prepare_psi.get_num_qubits()} qubits, "
                f"but U expects {n}"
            )
        qc.append(prepare_psi, target=tgt)

    # Controlled-U: U applied when ancilla = |1⟩
    qc.append(U, target=tgt, control=[anc], control_state="1")

    # Second Hadamard on ancilla (closes the interferometer)
    qc.h(anc)

    return qc


# ---------------------------------------------------------------------------
# 2. Swap test circuit builder
# ---------------------------------------------------------------------------

def build_swap_test_circuit(
    prepare_phi: Circuit,
    prepare_psi: Circuit,
) -> Tuple[Circuit, Circuit]:
    """Build the swap test components: joint preparation + SWAP unitary.

    The swap test estimates |⟨φ|ψ⟩|² using the identity:
        p(0) = (1 + |⟨φ|ψ⟩|²) / 2
    ⇒  |⟨φ|ψ⟩|² = 2·p(0) - 1 = ⟨Z⟩_anc

    Args:
        prepare_phi: Circuit preparing |φ⟩ on qubits 0..n-1.
        prepare_psi: Circuit preparing |ψ⟩ on qubits n..2n-1.

    Returns:
        Tuple of (U_swap, prepare_joint):
            - U_swap: SWAP circuit acting on 2n qubits
            - prepare_joint: Joint state prep |φ⟩⊗|ψ⟩

    Raises:
        ValueError: If prepare_phi and prepare_psi have different qubit counts.
    """
    n_phi = prepare_phi.get_num_qubits()
    n_psi = prepare_psi.get_num_qubits()
    if n_phi != n_psi:
        raise ValueError(
            f"prepare_phi ({n_phi} qubits) and prepare_psi ({n_psi} qubits) "
            f"must have the same number of qubits"
        )
    n = n_phi

    # Build SWAP gate between qubit i and i+n for i = 0..n-1
    U_swap = Circuit(2 * n, name="SWAP")
    for i in range(n):
        U_swap.swap(i, i + n)

    # Joint state preparation: |φ⟩ on first n qubits, |ψ⟩ on last n qubits
    prepare_joint = Circuit(2 * n, name="Prep(|phi⊗psi⟩)")
    prepare_joint.append(prepare_phi, target=list(range(n)))
    prepare_joint.append(prepare_psi, target=list(range(n, 2 * n)))

    return U_swap, prepare_joint


# ---------------------------------------------------------------------------
# 3. Shot noise simulation
# ---------------------------------------------------------------------------

def _simulate_measurement(p0_exact: float, shots: int, rng: Optional[np.random.Generator] = None) -> float:
    """Simulate shot noise on ancilla measurement.

    Args:
        p0_exact: Exact probability of measuring ancilla in |0⟩.
        shots: Number of measurement shots.
        rng: Optional random generator for reproducibility.

    Returns:
        Noisy estimate of p(0) after `shots` binomial trials.
    """
    if rng is None:
        rng = np.random.default_rng()
    c0 = int(rng.binomial(int(shots), p0_exact))
    return c0 / float(shots)


# ---------------------------------------------------------------------------
# 4. Phase estimation from real and imaginary parts
# ---------------------------------------------------------------------------

def _estimate_phi_from_real_imag(re_est: float, im_est: float) -> float:
    """Compute eigenphase φ ∈ [0, 1) from real and imaginary estimates.

    φ = atan2(Im, Re) / (2π) mod 1.0

    Args:
        re_est: Estimated real part ⟨ψ|U|ψ⟩.
        im_est: Estimated imaginary part ⟨ψ|U|ψ⟩.

    Returns:
        Eigenphase φ normalized to [0, 1).
    """
    angle = float(np.arctan2(im_est, re_est))
    return float((angle / (2.0 * np.pi)) % 1.0)


# ---------------------------------------------------------------------------
# 5. End-to-end Hadamard Test
# ---------------------------------------------------------------------------

def hadamard_test_run(
    mode: str = "expectation",
    U: Optional[Circuit] = None,
    prepare_psi: Optional[Circuit] = None,
    prepare_phi: Optional[Circuit] = None,
    imag: bool = False,
    shots: int = 20000,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the Hadamard Test in one of three modes.

    Modes:
        'expectation':     Estimate Re or Im of ⟨ψ|U|ψ⟩.
        'swap_test':       Estimate state overlap |⟨φ|ψ⟩|².
        'phase_estimation': Run both real and imag tests; reconstruct φ.

    Args:
        mode: One of 'expectation', 'swap_test', 'phase_estimation'.
        U: Unitary operator. Required for 'expectation' and
           'phase_estimation'.
        prepare_psi: Circuit preparing |ψ⟩. Required for 'swap_test'.
        prepare_phi: Circuit preparing |φ⟩. Required for 'swap_test'.
        imag: If True, estimate imaginary part (only for 'expectation').
        shots: Number of measurement shots for statistical sampling.
        backend: Simulation backend ('torch').
        device: Compute device ('cpu' or 'cuda').
        dtype: Numerical dtype.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok' on success, 'failed' otherwise
            - Estimated Value: Float estimate per mode
            - Computation Time (s): Wall-clock time
            - circuit: Primary Circuit object
            - measurements: Dict of measurement results per branch
            - p0_exact: Exact ancilla |0⟩ probability (no shot noise)

    Raises:
        ValueError: On invalid mode or missing required parameters.
    """
    # --- Stage 1: Validation ---
    valid_modes = {"expectation", "swap_test", "phase_estimation"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    if mode == "swap_test" and (prepare_phi is None or prepare_psi is None):
        raise ValueError("swap_test mode requires both prepare_phi and prepare_psi")
    if mode in {"expectation", "phase_estimation"} and U is None:
        raise ValueError(f"Unitary operator U is required in '{mode}' mode")

    if verbose:
        print(f"Hadamard Test")
        print(f"  Mode: {mode}")
        print(f"  Shots: {shots}")

    t_start = time.perf_counter()

    # --- Stage 2: Circuit construction ---
    circuits: Dict[str, Circuit] = {}
    rng = np.random.default_rng()

    if mode == "swap_test":
        U_swap, prepare_joint = build_swap_test_circuit(prepare_phi, prepare_psi)
        gs_real = hadamard_test_circuit(U_swap, prepare_joint, imag=False)
        circuits["real"] = gs_real
        if verbose:
            n = prepare_phi.get_num_qubits()
            print(f"  Built SWAP test circuit: {2 * n}-qubit target + 1 ancilla")

    elif mode == "phase_estimation":
        gs_real = hadamard_test_circuit(U, prepare_psi, imag=False)
        gs_imag = hadamard_test_circuit(U, prepare_psi, imag=True)
        circuits["real"] = gs_real
        circuits["imag"] = gs_imag
        if verbose:
            print(f"  Built Real & Imag circuits for phase estimation")

    else:  # expectation
        qc = hadamard_test_circuit(U, prepare_psi, imag=imag)
        circuits["main"] = qc
        if verbose:
            part = "Im" if imag else "Re"
            print(f"  Built Hadamard test circuit ({part} part)")

    # --- Stage 3: Simulation + Sampling ---
    measurements: Dict[str, float] = {}
    p0_exact_all: Dict[str, float] = {}

    for name, circ in circuits.items():
        # Execute statevector simulation
        state = circ.execute(backend=backend, device=device, dtype=dtype)
        # Extract ancilla (qubit 0) probability from calculate_state dict
        anc_probs = state.calculate_state([0])

        # calculate_state returns {bitstring: {prob: ..., int: ...}} or
        # {bitstring: float} depending on backend version
        p0_val = anc_probs.get("0", 0.0)
        if isinstance(p0_val, dict):
            p0_exact = float(p0_val.get("prob", p0_val.get("probability", 0.0)))
        elif isinstance(p0_val, (int, float)):
            p0_exact = float(p0_val)
        else:
            p0_exact = 0.0

        p0_exact_all[name] = p0_exact

        # Apply shot noise
        if shots is not None and shots > 0:
            p0 = _simulate_measurement(p0_exact, shots, rng)
        else:
            p0 = p0_exact

        p1 = 1.0 - p0
        exp_val = p0 - p1  # ⟨Z⟩ = p(0) - p(1)
        measurements[name] = exp_val

        if verbose:
            print(f"  [{name}] p0_exact={p0_exact:.6f}, p0_noisy={p0:.6f}, "
                  f"⟨Z⟩={exp_val:.6f}")

    # --- Stage 4: Classical post-processing ---
    est_val: float

    if mode == "expectation":
        est_val = measurements["main"]

    elif mode == "swap_test":
        # |⟨φ|ψ⟩|² = ⟨Z⟩_anc = p0 - p1
        # Clamp to valid range [0, 1]
        est_val = float(np.clip(measurements["real"], 0.0, 1.0))

    elif mode == "phase_estimation":
        re_est = measurements["real"]
        im_est = measurements["imag"]
        est_val = _estimate_phi_from_real_imag(re_est, im_est)

    comp_time = time.perf_counter() - t_start

    if verbose:
        print(f"  Estimated value: {est_val:.6f}")
        print(f"  Computation time: {comp_time:.4f}s")

    primary_circuit = next(iter(circuits.values()))

    return {
        "status": "ok",
        "Estimated Value": est_val,
        "Computation Time (s)": round(comp_time, 4),
        "circuit": primary_circuit,
        "circuits": circuits,
        "measurements": measurements,
        "p0_exact": p0_exact_all,
        "circuit_path": "",
        "plot": [],
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class HadamardTestSolver:
    """Class-based solver for the Hadamard Test.

    Usage:
        solver = HadamardTestSolver(text_mode='plain')
        result = solver.run(
            mode='expectation', U=circuit, prepare_psi=prep,
            shots=20000,
        )
        print(result['Estimated Value'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        mode: str = "expectation",
        U: Optional[Circuit] = None,
        prepare_psi: Optional[Circuit] = None,
        prepare_phi: Optional[Circuit] = None,
        imag: bool = False,
        shots: int = 20000,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run the Hadamard Test. See hadamard_test_run() for docs."""
        result = hadamard_test_run(
            mode=mode, U=U, prepare_psi=prepare_psi,
            prepare_phi=prepare_phi, imag=imag, shots=shots,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "Estimated Value": result.get("Estimated Value"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 7. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    # (mode, U_angle, psi_type, expected_label, imag, shots)
    {
        "mode": "expectation", "angle": 0.8, "psi_type": "plus",
        "imag": False, "label": "Re⟨+|RZ(0.8)|+⟩ = cos(0.4) ≈ 0.9211",
    },
    {
        "mode": "expectation", "angle": 0.8, "psi_type": "plus",
        "imag": True, "label": "Im⟨+|RZ(0.8)|+⟩ = -sin(0.4) ≈ -0.3894",
    },
    {
        "mode": "expectation", "angle": 1.2, "psi_type": "zero",
        "imag": False, "label": "Re⟨0|RZ(1.2)|0⟩ = cos(0.6) ≈ 0.8253",
    },
    {
        "mode": "phase_estimation", "angle": 0.8, "psi_type": "plus",
        "imag": False, "label": "Eigenphase of RZ(0.8) on |+⟩ → φ=0.4/π",
    },
    {
        "mode": "swap_test", "angle": 0.0, "psi_type": "bell",
        "imag": False, "label": "|⟨Bell|++⟩|² = 0.5",
    },
]


def _build_test_unitary(angle: float) -> Circuit:
    """Build RZ(angle) as test unitary on 1 qubit."""
    U = Circuit(1, name=f"RZ_{angle:.2f}")
    U.rz(angle, 0)
    return U


def _build_test_state(psi_type: str) -> Circuit:
    """Build test state preparation circuits.

    'plus':  H|0⟩ = |+⟩
    'zero':  I|0⟩ = |0⟩
    'bell':  2-qubit Bell state preparation for swap test
    """
    if psi_type == "plus":
        qc = Circuit(1, name="|+⟩")
        qc.h(0)
        return qc
    elif psi_type == "zero":
        qc = Circuit(1, name="|0⟩")
        return qc
    elif psi_type == "bell":
        qc = Circuit(2, name="|Bell⟩")
        qc.h(0)
        qc.cx(0, 1)
        return qc
    else:
        raise ValueError(f"Unknown psi_type: {psi_type}")


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case and print result."""
    mode = case["mode"]
    psi_type = case["psi_type"]
    label = case["label"]

    if mode == "swap_test":
        # Swap test: |Bell⟩ vs |++⟩ → |⟨φ|ψ⟩|² = 0.5
        prepare_phi = Circuit(2, name="|++⟩")
        prepare_phi.h(0)
        prepare_phi.h(1)
        prepare_psi = _build_test_state("bell")

        result = hadamard_test_run(
            mode="swap_test", prepare_phi=prepare_phi,
            prepare_psi=prepare_psi, shots=40000,
            backend="torch", verbose=False,
        )
        expected = 0.5
        est = result["Estimated Value"]
        error = abs(est - expected)
        ok = error < 0.1  # shot noise tolerance
        icon = "ok" if ok else "WARN"
        print(f"  [{icon}] {label}")
        print(f"       expected={expected:.4f}, est={est:.4f}, error={error:.4f}")

    elif mode == "phase_estimation":
        angle = case["angle"]
        U = _build_test_unitary(angle)
        prepare_psi = _build_test_state(psi_type)

        result = hadamard_test_run(
            mode="phase_estimation", U=U, prepare_psi=prepare_psi,
            shots=20000, backend="torch", verbose=False,
        )
        # For RZ(α)|+⟩: ⟨+|RZ(α)|+⟩ = cos(α/2)
        # φ = atan2(-sin(α/2), cos(α/2)) / (2π) mod 1
        # For α=0.8: cos(0.4)≈0.9211, -sin(0.4)≈-0.3894
        # φ = atan2(-0.3894, 0.9211)/(2π) = -0.4/(2π) mod 1 = 1 - 0.4/(2π) ≈ 0.9363
        alpha = angle
        expected_phi = float(((-alpha / 2.0) / (2.0 * np.pi)) % 1.0)
        est = result["Estimated Value"]
        error = abs(est - expected_phi)
        ok = error < 0.05
        icon = "ok" if ok else "WARN"
        print(f"  [{icon}] {label}")
        print(f"       expected φ={expected_phi:.4f}, est={est:.4f}, error={error:.4f}")

    else:  # expectation
        angle = case["angle"]
        imag = case["imag"]
        U = _build_test_unitary(angle)
        prepare_psi = _build_test_state(psi_type)

        result = hadamard_test_run(
            mode="expectation", U=U, prepare_psi=prepare_psi,
            imag=imag, shots=20000, backend="torch", verbose=False,
        )

        if psi_type == "plus":
            # ⟨+|RZ(α)|+⟩ = cos(α/2)
            # Re: cos(α/2), Im: -sin(α/2)
            if imag:
                expected = -math.sin(angle / 2.0)
            else:
                expected = math.cos(angle / 2.0)
        elif psi_type == "zero":
            # ⟨0|RZ(α)|0⟩ = e^{-iα/2}
            # Re: cos(α/2), Im: -sin(α/2)
            if imag:
                expected = -math.sin(angle / 2.0)
            else:
                expected = math.cos(angle / 2.0)

        est = result["Estimated Value"]
        error = abs(est - expected)
        ok = error < 0.05  # shot noise tolerance
        icon = "ok" if ok else "WARN"
        part = "Im" if imag else "Re"
        print(f"  [{icon}] {label}")
        print(f"       expected={expected:.4f}, est({part})={est:.4f}, error={error:.4f}")

    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Hadamard Test — Manual Implementation")
    print("=" * 60)

    solver = HadamardTestSolver()

    # --- Demo 1: Expectation value (real part) ---
    print("\n--- Demo 1: Expectation mode — Re⟨+|RZ(0.8)|+⟩ ---")
    U1 = Circuit(1, name="RZ_0.8")
    U1.rz(0.8, 0)
    prep1 = Circuit(1, name="|+⟩")
    prep1.h(0)

    result1_re = solver.run(
        mode='expectation', U=U1, prepare_psi=prep1,
        imag=False, shots=20000,
    )
    expected_re = math.cos(0.4)
    print(f"  Expected:  cos(0.4) = {expected_re:.6f}")
    print(f"  Estimated: {result1_re['Estimated Value']:.6f}")
    print(f"  Error:     {abs(result1_re['Estimated Value'] - expected_re):.6e}")

    # --- Demo 2: Expectation value (imaginary part) ---
    print("\n--- Demo 2: Expectation mode — Im⟨+|RZ(0.8)|+⟩ ---")
    result1_im = solver.run(
        mode='expectation', U=U1, prepare_psi=prep1,
        imag=True, shots=20000,
    )
    expected_im = -math.sin(0.4)
    print(f"  Expected:  -sin(0.4) = {expected_im:.6f}")
    print(f"  Estimated: {result1_im['Estimated Value']:.6f}")
    print(f"  Error:     {abs(result1_im['Estimated Value'] - expected_im):.6e}")

    # --- Demo 3: Default state (|0⟩) ---
    print("\n--- Demo 3: Expectation mode — ⟨0|RZ(1.2)|0⟩ (no prepare_psi) ---")
    U3 = Circuit(1, name="RZ_1.2")
    U3.rz(1.2, 0)

    result3 = solver.run(
        mode='expectation', U=U3, prepare_psi=None,
        imag=False, shots=20000,
    )
    expected3 = math.cos(0.6)  # ⟨0|RZ(α)|0⟩ = e^{-iα/2}, Re = cos(α/2)
    print(f"  Expected:  cos(0.6) = {expected3:.6f}")
    print(f"  Estimated: {result3['Estimated Value']:.6f}")
    print(f"  Error:     {abs(result3['Estimated Value'] - expected3):.6e}")

    # --- Demo 4: Swap test ---
    print("\n--- Demo 4: Swap test — |⟨Bell|++⟩|² = 0.5 ---")
    prep_phi = Circuit(2, name="|++⟩")
    prep_phi.h(0)
    prep_phi.h(1)
    prep_psi_bell = Circuit(2, name="|Bell⟩")
    prep_psi_bell.h(0)
    prep_psi_bell.cx(0, 1)

    result4 = solver.run(
        mode='swap_test', prepare_phi=prep_phi,
        prepare_psi=prep_psi_bell, shots=40000,
    )
    print(f"  Expected:  0.5")
    print(f"  Estimated: {result4['Estimated Value']:.4f}")
    print(f"  Error:     {abs(result4['Estimated Value'] - 0.5):.4f}")

    # --- Demo 5: Phase estimation ---
    print("\n--- Demo 5: Phase estimation — eigenphase of RZ(0.8) on |+⟩ ---")
    result5 = solver.run(
        mode='phase_estimation', U=U1, prepare_psi=prep1,
        shots=20000,
    )
    # ⟨+|RZ(α)|+⟩ = cos(α/2) - i·sin(α/2) = e^{-iα/2}
    # φ = (-α/2) / (2π) mod 1 = (1 - α/(4π)) mod 1
    expected_phi = float(((-0.8 / 2.0) / (2.0 * np.pi)) % 1.0)
    print(f"  Expected φ:  {expected_phi:.6f}")
    print(f"  Estimated φ: {result5['Estimated Value']:.6f}")
    print(f"  Error:       {abs(result5['Estimated Value'] - expected_phi):.6e}")

    # --- Demo 6: Effect of shot count ---
    print("\n--- Demo 6: Effect of shot count on precision ---")
    U6 = Circuit(1, name="RZ_0.8")
    U6.rz(0.8, 0)
    prep6 = Circuit(1, name="|+⟩")
    prep6.h(0)

    for s in [100, 500, 2000, 10000, 50000]:
        result6 = hadamard_test_run(
            mode="expectation", U=U6, prepare_psi=prep6,
            imag=False, shots=s, backend="torch", verbose=False,
        )
        err6 = abs(result6["Estimated Value"] - math.cos(0.4))
        # Expected shot noise: σ ≈ 1/√shots
        sigma = 1.0 / math.sqrt(s)
        print(f"  shots={s:>6}: est={result6['Estimated Value']:.6f}, "
              f"error={err6:.6f}, expected σ≈{sigma:.6f}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")

    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
