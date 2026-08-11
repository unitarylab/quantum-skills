import numpy as np
from unitarylab import Circuit, Register


def prepare_b_unitary(b):
    """Source-equivalent Householder U_b with U_b|0> = |b>."""
    b = np.asarray(b, dtype=complex).reshape(-1, 1)
    v = np.zeros_like(b)
    v[0, 0] = np.sqrt((b[0, 0] + 1.0) / 2.0)
    for k in range(1, len(b)):
        v[k, 0] = b[k, 0] / (2.0 * v[0, 0])
    return 2 * (v @ v.conj().T) - np.eye(len(b))


def block_encode(A):
    """Source-equivalent SVD block encoding for ||A||_2 <= 1."""
    U, s, Vh = np.linalg.svd(A)
    S = np.diag(s)
    R = np.diag(np.sqrt(np.clip(1 - s**2, 0, None)))
    middle = np.block([[S, R], [R, -S]])
    I = np.eye(len(s))
    Z = np.zeros_like(I)
    return np.block([[U, Z], [Z, I]]) @ middle @ np.block([[Vh, Z], [Z, I]])


def add_adiabatic_step(qc, k, T, n, sys, anc, Ub, BlA, kappa, p, mcz):
    """Append the source's complete 11-operation AQC step."""
    s = k / T
    f = kappa / (kappa - 1) * (
        1 - (1 + s * (kappa ** (p - 1) - 1)) ** (1 / (1 - p))
    )
    theta = 2 * np.arctan2(f, 1 - f)

    qc.h(anc[2])
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
    qc.cz(control=anc[1], target=anc[3], control_state=0)
    qc.cry(theta, control=anc[1], target=anc[3], control_state=0)
    qc.ch(control=anc[1], target=anc[3], control_state=1)
    qc.cz(target=anc[4], control=anc[3], control_state=0)
    qc.unitary(BlA, target=list(range(n + 1)),
               control=[n + 3, n + 4], control_state=[1, 1])
    qc.unitary(BlA.conj().T, target=list(range(n + 1)),
               control=[n + 3, n + 4], control_state=[1, 0])
    qc.cx(anc[3], anc[4])

    qc.x(anc[1])
    qc.ch(control=anc[1], target=anc[3], control_state=1)
    qc.cz(control=anc[1], target=anc[3], control_state=0)
    qc.cry(theta, control=anc[1], target=anc[3], control_state=0)
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
    qc.h(anc[2])
    qc.x(anc[3])
    qc.mcz(controls=[n, n + 2], target=anc[3], control_state=[0, 0])
    qc.x(anc[3])
    qc.gp(np.pi)


def solve_manual_aqc(n=2, T=0, p=1.4, backend="torch", device="cpu"):
    """Build, run, post-select, and validate AQC without AQCAlgorithm."""
    np.random.seed(42)
    N = 2**n
    random_A = np.random.randn(N, N)
    A_orig = ((random_A + random_A.T) / 2 + 10 * np.eye(N)).astype(complex)
    b_orig = np.random.randn(N).astype(complex)

    a_scale = float(np.linalg.norm(A_orig, ord=2))
    b_scale = float(np.linalg.norm(b_orig))
    A = A_orig / a_scale
    b = b_orig / b_scale
    kappa = np.linalg.cond(A)
    if abs(kappa - 1) < 1e-12:
        x = b * (b_scale / a_scale)
        return {
            "status": "ok",
            "Quantum Solution (x)": x,
            "Classical Solution": np.linalg.solve(A_orig, b_orig),
            "Residual Norm ||Ax-b||": float(np.linalg.norm(A_orig @ x - b_orig)),
            "Error vs Classical (L2)": float(
                np.linalg.norm(x - np.linalg.solve(A_orig, b_orig))
            ),
            "Internal Scale Factor": 1.0,
            "circuit": None,
        }

    T_value = int(np.ceil(10 * kappa)) if T == 0 else int(T)
    if T == 0 and T_value % 2:
        T_value += 1

    sys = Register("sys", n)
    anc = Register("anl", 5)
    qc = Circuit(sys, anc, name="manual_aqc")
    Ub, BlA = prepare_b_unitary(b), block_encode(A)
    qc.unitary(Ub, target=sys[:])

    mcz = Circuit(n + 1)
    mcz.mcz(controls=list(range(n)), target=n, control_state=[1] * n)
    for k in range(1, T_value + 1):
        add_adiabatic_step(qc, k, T_value, n, sys, anc, Ub, BlA, kappa, p, mcz)

    state = np.asarray(
        qc.execute(backend=backend, device=device, dtype=np.complex128).state,
        dtype=complex,
    ).reshape(-1)

    # Little-endian ancilla post-selection: anl[0:4]=0 and anl[4]=1.
    x_postselected = state[2 ** (n + 4):][:N]
    postselection_amplitude = float(np.linalg.norm(x_postselected))
    if postselection_amplitude < 1e-15:
        raise RuntimeError("Near-zero post-selection amplitude; increase T")
    x_direction = x_postselected / postselection_amplitude

    internal_scale = 0j
    for i in range(N):
        if abs(b[i]) > 1e-12:
            internal_scale = b[i] / (A[i] @ x_direction)
            break
    if abs(internal_scale) < 1e-15:
        internal_scale = 1.0

    x_normalized_problem = internal_scale * x_direction
    x_final = x_normalized_problem * (b_scale / a_scale)
    x_classical = np.linalg.solve(A, b)
    x_classical_direction = x_classical / np.linalg.norm(x_classical)

    return {
        "status": "ok",
        "Quantum Solution (x)": x_final,
        "Classical Solution": x_classical,
        "Residual Norm ||Ax-b||": float(np.linalg.norm(A_orig @ x_final - b_orig)),
        "Error vs Classical (L2)": float(
            np.linalg.norm(x_direction - x_classical_direction)
        ),
        "Internal Scale Factor": internal_scale,
        "Post-selection Amplitude": postselection_amplitude,
        "Adiabatic Steps (T)": T_value,
        "circuit": qc,
    }


if __name__ == "__main__":
    result = solve_manual_aqc(n=2, T=20, p=1.4)
    print(result["status"])
    print(result["Residual Norm ||Ax-b||"])
    print(result["Error vs Classical (L2)"])
