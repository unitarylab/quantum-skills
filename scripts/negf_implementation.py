"""Independent NEGF reconstruction for a clean rectangular square lattice.

Analytic semi-infinite lead self-energies replace Kwant for this separable model.
Default: NumPy classical benchmark and diagnostics. --execute: actual UnitaryLab
QSVT circuits plus classical acceptance checks. No packaged algorithm is imported.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import json
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NEGFConfig:
    L: int = 2
    W: int = 2
    a: float = 1.0
    t: float = 1.0
    inverse_tolerance: float = 2e-3

    def validate(self):
        for value in (self.L, self.W):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError("L and W must be integers >= 2")
        # Stronger educational input checks than the source wrapper's a/t checks.
        if not np.isfinite(self.a) or self.a <= 0 or not np.isfinite(self.t) or self.t <= 0:
            raise ValueError("a and t must be finite and positive")
        if not np.isfinite(self.inverse_tolerance) or not 0 < self.inverse_tolerance < 1:
            raise ValueError("inverse_tolerance must be finite and in (0, 1)")

    @property
    def energy(self):
        return self.t * (2 - np.cos(np.pi / (self.W + 1)) - np.cos(2 * np.pi / (self.W + 1)))


def build_device(config):
    """Build H, A and interface broadenings in (x, y) site order.

    The analytic retarded surface self-energy is exact for the same clean leads
    used by the source, but does not generalize to arbitrary Kwant systems.
    """
    config.validate()
    L, W, t = config.L, config.W, config.t
    hx = 2*t*np.eye(L) - t*np.eye(L, k=1) - t*np.eye(L, k=-1)
    hy = 2*t*np.eye(W) - t*np.eye(W, k=1) - t*np.eye(W, k=-1)
    h = np.kron(hx, np.eye(W)) + np.kron(np.eye(L), hy)
    modes = np.arange(1, W+1)
    q = modes*np.pi/(W+1)
    u = np.sqrt(2/(W+1))*np.sin(np.outer(np.arange(1, W+1), q))
    transverse = 2*t*(1-np.cos(q))
    z = config.energy - transverse - 2*t
    # Select Im(sigma) <= 0 and the decaying root outside each lead band.
    sigma_modes = np.empty(W, dtype=complex)
    inside = np.abs(z) <= 2*t
    sigma_modes[inside] = (z[inside] - 1j*np.sqrt(np.maximum(0, 4*t*t-z[inside]**2)))/2
    sigma_modes[~inside] = (z[~inside] - np.sign(z[~inside])*np.sqrt(z[~inside]**2-4*t*t))/2
    sigma = (u*sigma_modes) @ u.T
    gamma = 1j*(sigma-sigma.conj().T)
    interfaces = (np.arange(W), np.arange((L-1)*W, L*W))
    A = np.asarray(config.energy*np.eye(L*W)-h, dtype=complex)
    for interface in interfaces:
        A[np.ix_(interface, interface)] -= sigma
    # This deterministic ordering is local to this reconstruction, not fsyst.graph.
    bonds = np.argwhere((h != 0) & ~np.eye(L*W, dtype=bool))
    return {"H": h, "A": A, "sigma": sigma, "gamma": gamma,
            "interfaces": interfaces, "bonds": bonds}


def injection_vectors(gamma, sigma, interface, size):
    """Factor the local broadening and embed retained channels in device space."""
    interface = np.asarray(interface)
    if (isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer))
            or size < 0 or interface.ndim != 1
            or (interface.size and not np.issubdtype(interface.dtype, np.integer))):
        raise ValueError("size must be a nonnegative integer and interface a 1D integer array")
    interface = interface.astype(int)
    gamma, sigma = np.asarray(gamma), np.asarray(sigma)
    if (gamma.shape != (len(interface), len(interface)) or sigma.shape != gamma.shape
            or len(np.unique(interface)) != len(interface)
            or np.any(interface < 0) or np.any(interface >= size)
            or not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(sigma))):
        raise ValueError("invalid interface, broadening or self-energy")
    if not len(interface):
        return np.zeros((size, 0), dtype=complex)
    scale = max(np.linalg.norm(gamma, 2), 2*np.linalg.norm(sigma, 2))
    hermitian_tolerance = 100*np.finfo(float).eps*max(1, len(interface))*scale
    if np.linalg.norm(gamma-gamma.conj().T, 2) > hermitian_tolerance:
        raise ValueError("lead broadening must be Hermitian within roundoff")
    eigenvalues, eigenvectors = np.linalg.eigh((gamma+gamma.conj().T)/2)
    scale = max(np.max(np.abs(eigenvalues), initial=0.), 2*np.linalg.norm(sigma, 2))
    tolerance = 100*np.finfo(float).eps*max(1, len(interface))*scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError("retarded lead broadening must be positive semidefinite")
    keep = eigenvalues > tolerance
    C = np.zeros((size, np.count_nonzero(keep)), dtype=complex)
    C[interface] = eigenvectors[:, keep]*np.sqrt(eigenvalues[keep])
    assert np.allclose(C[interface] @ C[interface].conj().T, gamma,
                       atol=10*max(tolerance, np.finfo(float).tiny), rtol=1e-12)
    return C


def build_injections(model):
    size = len(model["A"])
    left, right = [injection_vectors(model["gamma"], model["sigma"], interface, size)
                   for interface in model["interfaces"]]
    rhs = np.column_stack((left, right))
    if rhs.shape[1] == 0:
        raise ValueError("no propagating injection channels")
    return left, right, rhs


def validate_matrix(matrix):
    """Reject malformed systems before importing or allocating SDK circuits."""
    matrix = np.asarray(matrix, dtype=complex)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix))):
        raise ValueError("matrix must be a finite square matrix of dimension >= 2")
    return matrix


def validate_tolerance(tolerance):
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("inverse_tolerance must be finite and in (0, 1)")


def calibrate_epsilon(matrix, tolerance):
    """Mirror the SDK's reciprocal polynomial calibration, including its beta.

    These two SDK helpers are version-sensitive; there is no silent fallback.
    """
    matrix = validate_matrix(matrix)
    validate_tolerance(tolerance)
    from unitarylab.library.block_encoding.tools import normalize_matrix
    from unitarylab.library.linear_solver.qsvt import _indicator

    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps*len(matrix)*singular[0]:
        raise ValueError("A must be numerically invertible")
    normalized = matrix/np.linalg.norm(matrix, 2)
    singular = np.linalg.svd(normalized, compute_uv=False)
    kappa = np.linalg.cond(normalized)
    _, alpha = normalize_matrix(normalized, norm_type="spectral")
    sigma = singular/alpha
    grid = np.unique(np.r_[np.linspace(sigma[-1], sigma[0], 20001), sigma])
    log_ratio = -np.log(tolerance)/singular[0]**2
    for _ in range(32):
        epsilon = float(kappa*alpha*np.exp(-log_ratio))
        order = np.ceil((kappa*alpha)**2*log_ratio)
        if not np.isfinite(epsilon) or not 0 < epsilon < kappa*alpha or not np.isfinite(order):
            raise RuntimeError("QSVT calibration exceeded floating-point range")
        if 4*order > epsilon:
            odd, beta = _indicator(kappa, alpha, epsilon, 2.)
            coefficients = np.zeros(2*len(odd))
            coefficients[1::2] = odd
            values = np.polynomial.chebyshev.chebval(grid, coefficients)
            error = float(np.max(np.abs(grid*values*beta*kappa*alpha-1)))
            if np.isfinite(error) and error <= tolerance:
                return epsilon, error, len(coefficients)-1
        log_ratio *= 2
    raise RuntimeError("reciprocal polynomial did not meet tolerance")


def qsvt_responses(A, rhs, tolerance):
    """Reuse one SDK QSVT core, preserving input norm and complex phase."""
    A = validate_matrix(A)
    validate_tolerance(tolerance)
    rhs = np.asarray(rhs, dtype=complex)
    if (rhs.ndim != 2 or rhs.shape[0] != len(A) or rhs.shape[1] == 0
            or not np.all(np.isfinite(rhs)) or np.any(np.linalg.norm(rhs, axis=0) == 0)):
        raise ValueError("rhs must contain finite nonzero columns with one row per matrix row")
    from unitarylab import Circuit
    from unitarylab.library import solve

    n = len(A)
    padded_size = 1 << (n-1).bit_length()
    padded = np.asarray(np.linalg.norm(A, 2)*np.eye(padded_size), dtype=complex)
    padded[:n, :n] = A
    epsilon, polynomial_error, degree = calibrate_epsilon(padded, tolerance)
    basis = np.zeros(padded_size, dtype=complex)
    basis[0] = 1
    solver = solve(padded, basis, method="qsvt", epsilon=epsilon,
                   backend="numpy", dtype=np.complex128, logging=False)
    core = solver.circuit
    system_qubits = list(range(padded_size.bit_length()-1))
    responses = np.zeros_like(rhs, dtype=complex)
    circuits = []
    for j, vector in enumerate(rhs.T):
        norm = np.linalg.norm(vector)
        state = np.zeros(padded_size, dtype=complex)
        state[:n] = vector/norm
        circuit = Circuit(core.get_num_qubits(), name=f"NEGF injection {j+1}")
        circuit.initialize(state, system_qubits)
        circuit.append(core, list(range(core.get_num_qubits())))
        out = np.asarray(circuit.execute(backend="numpy", dtype=np.complex128).state)
        assert out.shape == (1 << core.get_num_qubits(),), "unexpected SDK state shape"
        assert np.all(np.isfinite(out)), "SDK state contains nonfinite amplitudes"
        assert np.isclose(np.linalg.norm(out), 1., atol=1e-8, rtol=0)
        # Low-order system bits, all ancillas zero: take the first n amplitudes.
        responses[:, j] = norm*solver.scaling_factor*out[:n]
        circuits.append(circuit)
    assert np.all(np.isfinite(responses))
    return responses, {"epsilon": epsilon, "inverse_polynomial_error": polynomial_error,
                       "polynomial_degree": degree, "scaling_factor": float(solver.scaling_factor),
                       "circuits": circuits}


def observables(model, left, right, responses):
    X_L = responses[:, :left.shape[1]]
    rhs = np.column_stack((left, right))
    currents = np.array([2*np.imag(model["H"][i, j]*np.vdot(X_L[i], X_L[j]))
                         for i, j in model["bonds"]])
    divergence = np.zeros(len(model["A"]))
    np.add.at(divergence, model["bonds"][:, 0], currents)
    return {"current_divergence": divergence,
            "transmission": float(np.linalg.norm(right.conj().T @ X_L, "fro")**2),
            "ldos": np.sum(np.abs(responses)**2, axis=1)/(2*np.pi),
            "bond_current": currents,
            "injection_residuals": np.linalg.norm(model["A"] @ responses-rhs, axis=0)/np.linalg.norm(rhs, axis=0)}


def run_negf(config=NEGFConfig(), execute=False, acceptance_tolerance=0.05):
    """Return benchmark evidence; optionally grade actual QSVT against it.

    acceptance_tolerance is an educational validation threshold independent of
    the source's reciprocal polynomial tolerance (the source has no such gate).
    """
    if not np.isfinite(acceptance_tolerance) or acceptance_tolerance <= 0:
        raise ValueError("acceptance_tolerance must be finite and positive")
    model = build_device(config)
    left, right, rhs = build_injections(model)
    classical = np.linalg.solve(model["A"], rhs)
    exact = observables(model, left, right, classical)
    # Caroli/Green function checks independently verify the response formulas.
    G = np.linalg.inv(model["A"])
    spectral_ldos = -np.imag(np.diag(G))/np.pi
    # LDOS has units 1/energy: scale by t before using an absolute tolerance.
    assert np.allclose(config.t*exact["ldos"], config.t*spectral_ldos, atol=1e-10, rtol=1e-10)
    assert np.max(exact["injection_residuals"]) < 1e-10
    assert np.isclose(exact["transmission"], 1., atol=1e-10)
    assert left.shape[1] == right.shape[1] == 1
    currents = dict(zip(map(tuple, model["bonds"]), exact["bond_current"]))
    assert all(np.isclose(value, -currents[(j, i)], atol=1e-10)
               for (i, j), value in currents.items())
    divergence = exact["current_divergence"].reshape(config.L, config.W)
    flux_error = max(abs(divergence[0].sum()+exact["transmission"]),
                     abs(divergence[-1].sum()-exact["transmission"]),
                     np.max(np.abs(divergence[1:-1]), initial=0.))
    assert flux_error < 1e-10, "source/drain flux or bulk current conservation failed"
    report = {"mode": "qsvt" if execute else "classical_benchmark",
              "parameters": {"L": int(config.L), "W": int(config.W), "a": float(config.a),
                             "t": float(config.t), "inverse_tolerance": float(config.inverse_tolerance)},
              "ENERGY": float(config.energy),
              "system_size": int(config.L*config.W),
              "padded_system_size": 1 << (int(config.L*config.W)-1).bit_length(),
              "left_channels": left.shape[1], "right_channels": right.shape[1],
              "classical_transmission": exact["transmission"],
              "classical_max_residual": float(max(exact["injection_residuals"])),
              "classical_current_conservation_error": float(flux_error),
              "classical_ldos": exact["ldos"].tolist(),
              "spectral_ldos_max_error": float(np.max(np.abs(exact["ldos"]-spectral_ldos))),
              "directed_bonds": model["bonds"].tolist(),
              "classical_bond_current": exact["bond_current"].tolist(),
              "sdk_available": importlib.util.find_spec("unitarylab") is not None}
    if execute:
        responses, metadata = qsvt_responses(model["A"], rhs, config.inverse_tolerance)
        actual = observables(model, left, right, responses)
        relative_error = float(np.linalg.norm(responses-classical)/np.linalg.norm(classical))
        assert max(actual["injection_residuals"]) < acceptance_tolerance
        assert relative_error < acceptance_tolerance
        assert abs(actual["transmission"]-exact["transmission"]) < acceptance_tolerance
        scaled_ldos_error = float(config.t*np.max(np.abs(actual["ldos"]-exact["ldos"])))
        current_error = float(np.max(np.abs(actual["bond_current"]-exact["bond_current"])))
        assert scaled_ldos_error < acceptance_tolerance
        assert current_error < acceptance_tolerance
        circuits = metadata.pop("circuits")
        report.update(metadata)
        report.update({"transmission": actual["transmission"],
                       "injection_residuals": actual["injection_residuals"].tolist(),
                       "ldos": actual["ldos"].tolist(),
                       "bond_current": actual["bond_current"].tolist(),
                       "relative_response_error": relative_error,
                       "scaled_ldos_max_error": scaled_ldos_error,
                       "bond_current_max_error": current_error,
                       "transmission_absolute_error": abs(actual["transmission"]-exact["transmission"]),
                       "acceptance_tolerance": acceptance_tolerance,
                       "executed_circuits": len(circuits),
                       "num_qubits": circuits[0].get_num_qubits()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run actual QSVT circuits.")
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--W", type=int, default=2)
    parser.add_argument("--a", type=float, default=1.)
    parser.add_argument("--t", type=float, default=1.)
    parser.add_argument("--inverse-tolerance", type=float, default=2e-3)
    parser.add_argument("--acceptance-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not np.isfinite(args.acceptance_tolerance) or args.acceptance_tolerance <= 0:
        parser.error("--acceptance-tolerance must be finite and positive")
    config = NEGFConfig(args.L, args.W, args.a, args.t, args.inverse_tolerance)
    # Keep SDK progress on stderr, so stdout remains one parseable JSON report.
    with redirect_stdout(sys.stderr):
        report = run_negf(config, args.execute, args.acceptance_tolerance)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
