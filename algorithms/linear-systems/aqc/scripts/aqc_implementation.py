import numpy as np
from unitarylab import Circuit, Register


def prepare_b_unitary(b):
    """Build the documented real-vector Householder preparation U_b."""
    b = np.asarray(b, dtype=complex).reshape(-1, 1)
    if len(b) == 0 or not np.all(np.isfinite(b)):
        raise ValueError("b must be a non-empty finite vector")

    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-15:
        raise ValueError("b must be nonzero")
    if not np.isclose(norm_b, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("b must be normalized before Householder preparation")
    if abs(b[0, 0].imag) > 1e-12:
        raise ValueError("This documented Householder method expects a real vector")
    if abs(b[0, 0].real + 1.0) < 1e-12:
        raise RuntimeError("Householder construction is unstable when b[0] is near -1")

    v = np.zeros_like(b)
    v[0, 0] = np.sqrt((b[0, 0] + 1.0) / 2.0)
    for k in range(1, len(b)):
        v[k, 0] = b[k, 0] / (2.0 * v[0, 0])
    return 2.0 * (v @ v.conj().T) - np.eye(len(b))


def block_encode(A):
    """Build the documented SVD block encoding for a normalized square matrix."""
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    if np.linalg.norm(A, ord=2) > 1.0:
        raise ValueError("A must satisfy ||A||_2 <= 1")

    U, singular_values, Vh = np.linalg.svd(A)
    S = np.diag(singular_values)
    R = np.diag(np.sqrt(np.clip(1.0 - singular_values**2, 0.0, None)))
    middle = np.block([[S, R], [R, -S]])
    identity = np.eye(len(singular_values))
    zero = np.zeros_like(identity)
    block = (
        np.block([[U, zero], [zero, identity]])
        @ middle
        @ np.block([[Vh, zero], [zero, identity]])
    )

    # SKILL.md 未说明以下可选调试检查应使用的数值容差；这里不自行规定。
    return block


def add_adiabatic_step(qc, k, T, n, sys, anc, Ub, BlA, kappa, p, mcz):
    """Append the documented 11-stage logical AQC step in the required order."""
    s = k / T
    f = kappa / (kappa - 1.0) * (
        1.0
        - (1.0 + s * (kappa ** (p - 1.0) - 1.0)) ** (1.0 / (1.0 - p))
    )
    theta = 2.0 * np.arctan2(f, 1.0 - f)

    # Stage 1: open the interference branch.
    qc.h(anc[2])

    # Stage 2: first U_b reflection sequence.
    qc.unitary(Ub.conj().T, target=sys[:])
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.append(
        mcz,
        target=list(range(n)) + [n + 4],
        control=[n + 1, n + 2],
        control_state=[1, 1],
    )
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.unitary(Ub, target=sys[:])

    # Stage 3: scheduled controlled rotation.
    qc.cz(control=anc[1], target=anc[3], control_state=0)
    qc.cry(theta, control=anc[1], target=anc[3], control_state=0)

    # Stage 4: controlled Hadamard.
    qc.ch(control=anc[1], target=anc[3], control_state=1)

    # Stage 5: controlled block encoding and its adjoint.
    qc.cz(target=anc[4], control=anc[3], control_state=0)
    qc.unitary(
        BlA,
        target=list(range(n + 1)),
        control=[n + 3, n + 4],
        control_state=[1, 1],
    )
    qc.unitary(
        BlA.conj().T,
        target=list(range(n + 1)),
        control=[n + 3, n + 4],
        control_state=[1, 0],
    )
    qc.cx(anc[3], anc[4])

    # Stage 6: switch the symmetric branch.
    qc.x(anc[1])

    # Stage 7: mirrored controlled Hadamard.
    qc.ch(control=anc[1], target=anc[3], control_state=1)

    # Stage 8: mirrored scheduled rotation.
    qc.cz(control=anc[1], target=anc[3], control_state=0)
    qc.cry(theta, control=anc[1], target=anc[3], control_state=0)

    # Stage 9: second U_b reflection sequence.
    qc.unitary(Ub.conj().T, target=sys[:])
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.append(
        mcz,
        target=list(range(n)) + [n + 4],
        control=[n + 1, n + 2],
        control_state=[1, 1],
    )
    for i in range(n):
        qc.x(sys[i])
    qc.x(anc[4])
    qc.unitary(Ub, target=sys[:])

    # Stage 10: close the interference branch.
    qc.h(anc[2])

    # Stage 11: final reflection and global phase.
    qc.x(anc[3])
    qc.mcz(controls=[n, n + 2], target=anc[3], control_state=[0, 0])
    qc.x(anc[3])
    qc.gp(np.pi)


def solve_manual_aqc(
    n=2,
    T=0,
    p=1.4,
    backend="torch",
    device="cpu",
    dtype=np.complex128,
):
    """Build, execute, post-select, rescale, and validate the documented AQC solver."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer, not bool")
    if isinstance(T, (bool, np.bool_)) or not isinstance(T, (int, np.integer)):
        raise TypeError("T must be an integer, not bool")
    if isinstance(p, (bool, np.bool_)) or not isinstance(
        p, (int, float, np.integer, np.floating)
    ):
        raise TypeError("p must be a finite real number")

    n, T, p = int(n), int(T), float(p)
    if not 1 <= n <= 6:
        raise ValueError("n must be in [1, 6]")
    if T < 0:
        raise ValueError("T must be a non-negative integer")
    if not np.isfinite(p) or p <= 1.0:
        raise ValueError("p must be finite and greater than 1")

    np.random.seed(42)
    N = 2**n
    random_A = np.random.randn(N, N)
    A_orig = ((random_A + random_A.T) / 2.0 + 10.0 * np.eye(N)).astype(complex)
    b_orig = np.random.randn(N).astype(complex)

    a_scale = float(np.linalg.norm(A_orig, ord=2))
    b_scale = float(np.linalg.norm(b_orig))
    A = A_orig / a_scale
    b = b_orig / b_scale
    kappa = float(np.linalg.cond(A))
    classical = np.linalg.solve(A_orig, b_orig)

    # Avoid the singular kappa/(kappa-1) schedule and do not execute a circuit.
    if abs(kappa - 1.0) < 1e-12:
        return {
            "status": "ok",
            "Quantum Solution (x)": classical.copy(),
            "Classical Solution": classical,
            "Residual Norm ||Ax-b||": float(
                np.linalg.norm(A_orig @ classical - b_orig)
            ),
            "Error vs Classical (L2)": 0.0,
            "Phase-aligned Direction Error (L2)": 0.0,
            "Fidelity": 1.0,
            "Internal Scale Factor": 1.0,
            "Adiabatic Steps (T)": 0,
            "AQC Circuit Executed": False,
            "circuit": None,
        }

    T_value = int(np.ceil(10.0 * kappa)) if T == 0 else T
    if T == 0 and T_value % 2:
        T_value += 1

    sys = Register("sys", n)
    anc = Register("anl", 5)
    qc = Circuit(sys, anc, name="manual_aqc")
    Ub = prepare_b_unitary(b)
    BlA = block_encode(A)
    qc.unitary(Ub, target=sys[:])

    mcz = Circuit(n + 1)
    mcz.mcz(controls=list(range(n)), target=n, control_state=[1] * n)
    for k in range(1, T_value + 1):
        add_adiabatic_step(
            qc, k, T_value, n, sys, anc, Ub, BlA, kappa, p, mcz
        )

    state = np.asarray(
        qc.execute(backend=backend, device=device, dtype=dtype).state,
        dtype=complex,
    ).reshape(-1)
    if len(state) != 2 ** (n + 5):
        raise RuntimeError("Unexpected state-vector length")

    # Little-endian post-selection: anl[0:4]=0 and anl[4]=1.
    postselected = state[2 ** (n + 4):][:N]
    postselection_amplitude = float(np.linalg.norm(postselected))
    if postselection_amplitude < 1e-15:
        raise RuntimeError("Near-zero post-selection amplitude; increase T")
    direction = postselected / postselection_amplitude

    # Complex least-squares scale for the normalized problem.
    A_direction = A @ direction
    denominator = np.vdot(A_direction, A_direction)
    if abs(denominator) < 1e-15:
        raise RuntimeError("Least-squares scaling failed: <Av,Av> is near zero")
    internal_scale = np.vdot(A_direction, b) / denominator

    normalized_problem_solution = internal_scale * direction
    quantum_solution = normalized_problem_solution * (b_scale / a_scale)

    quantum_direction = quantum_solution / np.linalg.norm(quantum_solution)
    classical_direction = classical / np.linalg.norm(classical)
    overlap = np.vdot(classical_direction, quantum_direction)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1.0

    source_direction_error = float(
        np.linalg.norm(direction - classical_direction)
    )
    phase_aligned_error = float(
        np.linalg.norm(np.conj(phase) * quantum_direction - classical_direction)
    )

    # SKILL.md 未说明手工实现的 circuit_path、plot 及精确计时/导出格式，
    # 因此不自行设计这些返回字段。
    return {
        "status": "ok",
        "Quantum Solution (x)": quantum_solution,
        "Classical Solution": classical,
        "Residual Norm ||Ax-b||": float(
            np.linalg.norm(A_orig @ quantum_solution - b_orig)
        ),
        "Error vs Classical (L2)": source_direction_error,
        "Phase-aligned Direction Error (L2)": phase_aligned_error,
        "Fidelity": float(abs(overlap) ** 2),
        "Internal Scale Factor": internal_scale,
        "Post-selection Amplitude": postselection_amplitude,
        "Post-selection Probability": postselection_amplitude**2,
        "Adiabatic Steps (T)": T_value,
        "AQC Circuit Executed": True,
        "circuit": qc,
    }


if __name__ == "__main__":
    result = solve_manual_aqc(
        n=2,
        T=20,
        p=1.4,
        backend="torch",
        device="cpu",
        dtype=np.complex128,
    )
    print(result["status"])
    print(result["Residual Norm ||Ax-b||"])
    print(result["Error vs Classical (L2)"])
    print(result["Phase-aligned Direction Error (L2)"])
    print(result["Fidelity"])
    print(result["Post-selection Amplitude"])
    print(result["Post-selection Probability"])
