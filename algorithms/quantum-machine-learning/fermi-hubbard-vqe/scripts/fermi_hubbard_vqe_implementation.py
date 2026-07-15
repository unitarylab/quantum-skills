"""Standalone Fermi-Hubbard VQE validation implementation.

This manual implementation follows only the contract documented in the adjacent
Fermi-Hubbard VQE SKILL.md. It performs the energy workflow without fabricating
the public workflow's SVG, NumPy parameter file, convergence plot, or circuit
path.
"""

import time

import numpy as np
from scipy.optimize import minimize
from unitarylab import Circuit
from unitarylab.library.fermi_hubbard.fermi_hubbard_pauli import (
    fermi_hubbard_pauli,
)
from unitarylab.library.fermi_hubbard.pauli_ground_state import (
    pauli_string_to_matrix,
)


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer, not bool")
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer, not bool")
    if int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _reverse_bits(index, width):
    return int(f"{index:0{width}b}"[::-1], 2)


def _bit_reversal_permutation(n_qubits):
    dimension = 2**n_qubits
    permutation = np.array(
        [_reverse_bits(index, n_qubits) for index in range(dimension)],
        dtype=int,
    )

    # Bit reversal must be an involution and therefore round-trip exactly.
    if not np.array_equal(permutation[permutation], np.arange(dimension)):
        raise RuntimeError("bit-reversal round trip failed")
    return permutation


def _validate_parameters(theta, n_qubits, layers):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    expected_size = 2 * n_qubits * layers
    if theta.size != expected_size:
        raise ValueError(
            f"theta has the wrong size: expected {expected_size}, got {theta.size}"
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain only finite values")
    return theta


def _validate_state(state, n_qubits):
    state = np.asarray(state, dtype=complex).reshape(-1)
    expected_shape = (2**n_qubits,)
    if state.shape != expected_shape:
        raise RuntimeError(
            f"final state has the wrong shape: expected {expected_shape}, "
            f"got {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise RuntimeError("final state contains a non-finite value")
    return state


def _ansatz(theta, n_qubits, layers, *, backend, device, dtype):
    """Build and execute the documented Ry-Rz ring-CX ansatz."""
    n_qubits = _positive_int("n_qubits", n_qubits)
    layers = _positive_int("layers", layers)
    theta = _validate_parameters(theta, n_qubits, layers)

    circuit = Circuit(n_qubits)
    for layer in theta.reshape(layers, n_qubits, 2):
        for qubit in range(n_qubits):
            circuit.ry(layer[qubit, 0], qubit)
            circuit.rz(layer[qubit, 1], qubit)
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        if n_qubits > 1:
            circuit.cx(n_qubits - 1, 0)

    state = circuit.execute(
        backend=backend,
        device=device,
        dtype=dtype,
    ).state
    return circuit, _validate_state(state, n_qubits)


def combine_paired_mode_correlators(xx, yy, xy, yx, z):
    """Combine already measured paired-mode correlators as specified.

    SKILL.md specifies only these combinations. It does not specify the
    measurement circuits, the minimum enabled shot count, the finite-shot
    estimator, the standard-error estimator, or the public value container.
    """
    values = np.asarray([xx, yy, xy, yx, z], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired-mode correlators must be finite")
    return (
        float((xx + yy) / 4.0),
        float((xy - yx) / 4.0),
        float(z / 4.0),
    )


def _measurement_not_specified(measure_shots):
    if measure_shots <= 0:
        return
    raise NotImplementedError(
        "SKILL.md 未说明：paired-mode XX/YY/XY/YX/Z 的测量电路、最低 "
        "shots、有限采样方式、标准误差估计方法及 Measured Magnetic "
        "Moment 的值结构；手动验证实现不得猜测这些细节。"
    )


def fermi_hubbard_vqe(
    L=2,
    t=1.0,
    U=4.0,
    B=1.5,
    layers=5,
    max_iter=1000,
    seed=7,
    measure_shots=0,
    backend="torch",
    device="cpu",
    dtype=np.complex128,
):
    """Run the documented manual full-Fock-space energy validation workflow."""
    total_start = time.perf_counter()

    L = _positive_int("L", L)
    layers = _positive_int("layers", layers)
    max_iter = _positive_int("max_iter", max_iter)
    measure_shots = _nonnegative_int("measure_shots", measure_shots)
    _measurement_not_specified(measure_shots)

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer, not bool")
    seed = int(seed)
    if not isinstance(backend, str) or not isinstance(device, str):
        raise TypeError("backend and device must be strings")
    try:
        np.dtype(dtype)
    except TypeError as error:
        raise TypeError("dtype must be a valid NumPy dtype") from error

    # SKILL.md fixes the mode order as (1↑,1↓,2↑,2↓,...) and uses the
    # complete Fock space of all 2L modes. No particle-number restriction or
    # number-preserving initialization is added here.
    n_qubits = 2 * L
    dimension = 2**n_qubits

    # The prescribed helper supplies the open-chain Hamiltonian with
    # H = -t hopping + U onsite - B(n_up-n_down).
    pauli_expression = fermi_hubbard_pauli(L, t, U, B)
    h_big_endian = np.asarray(
        pauli_string_to_matrix(pauli_expression),
        dtype=np.complex128,
    )
    if h_big_endian.shape != (dimension, dimension):
        raise ValueError("Hamiltonian shape does not match the qubit count")
    if not np.all(np.isfinite(h_big_endian)):
        raise ValueError("Hamiltonian contains a non-finite value")
    if not np.allclose(h_big_endian, h_big_endian.conj().T, atol=1e-12):
        raise ValueError("Hamiltonian must be Hermitian")

    # UnitaryLab q0 is least significant, so bit-reverse both matrix axes.
    permutation = _bit_reversal_permutation(n_qubits)
    h_little_endian = h_big_endian[np.ix_(permutation, permutation)]
    h_round_trip = h_little_endian[np.ix_(permutation, permutation)]
    if not np.array_equal(h_round_trip, h_big_endian):
        raise RuntimeError("bit-reversed Hamiltonian failed its round trip")

    exact_energy = float(np.linalg.eigvalsh(h_little_endian)[0])

    # History and best parameters are internal only. The return contract does
    # not expose Convergence History or optimized parameters as standalone keys.
    convergence_history = []
    best_energy = np.inf
    best_parameters = None

    def energy(theta):
        nonlocal best_energy, best_parameters
        theta = _validate_parameters(theta, n_qubits, layers)
        _, state = _ansatz(
            theta,
            n_qubits,
            layers,
            backend=backend,
            device=device,
            dtype=dtype,
        )
        value = float(np.vdot(state, h_little_endian @ state).real)
        if not np.isfinite(value):
            raise RuntimeError("objective energy is non-finite")
        convergence_history.append(value)
        if value < best_energy:
            best_energy = value
            best_parameters = theta.copy()
        return value

    parameter_count = 2 * n_qubits * layers
    initial_parameters = np.random.default_rng(seed).uniform(
        -np.pi,
        np.pi,
        parameter_count,
    )
    initial_parameters = _validate_parameters(
        initial_parameters,
        n_qubits,
        layers,
    )

    vqe_start = time.perf_counter()
    optimizer_result = minimize(
        energy,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": max_iter},
    )
    vqe_runtime = time.perf_counter() - vqe_start

    if best_parameters is None or not np.isfinite(best_energy):
        raise RuntimeError("optimizer produced no finite objective evaluation")
    best_parameters = _validate_parameters(
        best_parameters,
        n_qubits,
        layers,
    )

    # Rebuild the final circuit from the lowest-energy parameters observed,
    # not from optimizer_result.x unless it is independently the tracked best.
    final_circuit, final_state = _ansatz(
        best_parameters,
        n_qubits,
        layers,
        backend=backend,
        device=device,
        dtype=dtype,
    )
    circuit_energy = float(
        np.vdot(final_state, h_little_endian @ final_state).real
    )
    if not np.isfinite(circuit_energy):
        raise RuntimeError("final circuit energy is non-finite")
    if abs(circuit_energy - best_energy) > 1e-8:
        raise RuntimeError("final circuit energy does not match tracked best energy")

    vqe_energy = best_energy
    absolute_error = abs(vqe_energy - exact_energy)
    if vqe_energy < exact_energy - 1e-8:
        raise RuntimeError("variational upper bound violated")

    # Do not infer convergence from nfev, absolute error, or the variational
    # bound. SKILL.md does not specify the source's message-text matching rule,
    # so this independent manual implementation exposes SciPy's optimizer state.
    optimizer_converged = bool(optimizer_result.success)
    optimizer_message = str(optimizer_result.message)

    # SKILL.md requires real SVG/.npy file descriptors from the public workflow.
    # This independent manual implementation creates no such files and therefore
    # leaves circuit_path and plot as None instead of fabricating paths or PNGs.
    result = {
        "status": "ok",
        "circuit_path": None,
        "plot": None,
        "circuit": final_circuit,
        "Exact Energy": exact_energy,
        "VQE Energy": vqe_energy,
        "Absolute Error": absolute_error,
        "Circuit Energy": circuit_energy,
        "Number of Qubits": n_qubits,
        "Optimizer Evaluations": int(optimizer_result.nfev),
        "Optimizer Converged": optimizer_converged,
        "Optimizer Message": optimizer_message,
        "VQE Runtime": float(vqe_runtime),
        "Total Runtime": float(time.perf_counter() - total_start),
        "Qubit Mapping": (
            f"(1↑,1↓,2↑,2↓,…,{L}↑,{L}↓); q0 is the least-significant bit"
        ),
        "Fermionic Hamiltonian": (
            "H=-t Σ(c†jσ c(j+1)σ+h.c.) + U Σ n(j↑)n(j↓) "
            "- B Σ(n(j↑)-n(j↓))"
        ),
        "Pauli Hamiltonian": pauli_expression,
    }

    # With measure_shots > 0, the public contract would additionally require
    # Measured Magnetic Moment, Magnetic Moment Standard Errors, and
    # Measurement Total Shots. The function raises above because SKILL.md does
    # not define enough detail to produce those values without guessing.
    return result


if __name__ == "__main__":
    output = fermi_hubbard_vqe(
        L=2,
        t=1.0,
        U=4.0,
        B=1.5,
        layers=5,
        max_iter=1000,
        seed=7,
        measure_shots=0,
        backend="torch",
        device="cpu",
        dtype=np.complex128,
    )
    print(output["status"])
    print(output["Exact Energy"])
    print(output["VQE Energy"])
    print(output["Absolute Error"])
    print(output["Circuit Energy"])
    print(output["Optimizer Converged"])
    print(output["Optimizer Message"])
    print(output["VQE Runtime"])
    print(output["Total Runtime"])
