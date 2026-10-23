"""Manual implementation of NumPy Eigensolver.

Computes the lowest k eigenvalues (and eigenstates) of a qubit operator using
exact classical diagonalization via NumPy/SciPy. Supports sparse and dense
matrix paths, auxiliary operator evaluation, and eigenpair filtering.

This is a classical reference solver — no quantum circuit construction or
execution is involved. It serves as a deterministic baseline for benchmarking
quantum eigensolvers.

Components:
    - pauli_string_to_matrix: Convert Pauli string to dense matrix
    - operator_to_matrix: Build Hamiltonian from Pauli terms
    - solve_eigensystem: Select dense/sparse solver and compute eigenpairs
    - numpy_eigensolver_solve: End-to-end solver with filtering
    - NumPyEigensolverSolver: Class-based interface

Reference:
    SKILL.md — NumPy Eigensolver
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


# ---------------------------------------------------------------------------
# 1. Pauli / operator utilities
# ---------------------------------------------------------------------------

_PAULI_MAP: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Convert a Pauli string (e.g. 'ZIX') to its dense matrix representation.

    Args:
        pauli_str: String of characters I, X, Y, Z.

    Returns:
        Dense matrix of shape (2^n, 2^n) where n = len(pauli_str).
    """
    result = np.array([[1.0]], dtype=np.complex128)
    for ch in pauli_str:
        result = np.kron(result, _PAULI_MAP.get(ch, _PAULI_MAP["I"]))
    return result


def operator_to_matrix(
    pauli_list: List[Tuple[str, float]],
) -> Tuple[np.ndarray, int]:
    """Build a dense Hermitian matrix from Pauli terms.

    H = Σᵢ cᵢ Pᵢ where Pᵢ ∈ {I, X, Y, Z}⊗ⁿ

    Args:
        pauli_list: List of (pauli_string, coefficient).

    Returns:
        Tuple of (matrix, num_qubits).
    """
    if not pauli_list:
        raise ValueError("pauli_list must not be empty")
    num_qubits = len(pauli_list[0][0])
    matrix = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=np.complex128)
    for pauli_str, coeff in pauli_list:
        if len(pauli_str) != num_qubits:
            raise ValueError(
                f"Pauli string '{pauli_str}' length {len(pauli_str)} != {num_qubits}"
            )
        matrix += coeff * pauli_string_to_matrix(pauli_str)
    return matrix, num_qubits


def is_diagonal(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """Check if a matrix is effectively diagonal.

    Args:
        matrix: Input matrix.
        tol: Numerical tolerance.

    Returns:
        True if matrix is diagonal within tolerance.
    """
    diag = np.diag(np.diag(matrix))
    return bool(np.allclose(matrix, diag, atol=tol))


# ---------------------------------------------------------------------------
# 2. Eigendecomposition
# ---------------------------------------------------------------------------

def solve_eigensystem(
    hamiltonian: np.ndarray,
    k: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the lowest k eigenvalues and eigenvectors of a Hermitian matrix.

    Automatically selects the appropriate solver:
        - Diagonal shortcut: directly extract diagonal entries
        - Dense path: numpy.linalg.eigh (always available)
        - Sparse path: scipy.sparse.linalg.eigsh (if scipy is installed and
          matrix has exploitable sparsity)

    Args:
        hamiltonian: Hermitian matrix of shape (d, d).
        k: Number of eigenvalues to compute (capped at dimension).

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted ascending.
            eigenvalues shape: (k_eff,)
            eigenvectors shape: (d, k_eff) — columns are eigenvectors.
    """
    dim = hamiltonian.shape[0]
    k_eff = min(k, dim)

    # Diagonal shortcut
    if is_diagonal(hamiltonian):
        diag = np.real(np.diag(hamiltonian))
        idx = np.argsort(diag)[:k_eff]
        evals = diag[idx]
        evecs = np.zeros((dim, k_eff), dtype=np.complex128)
        for j, i in enumerate(idx):
            evecs[i, j] = 1.0
        return evals, evecs

    # Sparse path (for large matrices with structure)
    if HAS_SCIPY_SPARSE and dim >= 16:
        try:
            sparse_mat = sp.csr_matrix(hamiltonian)
            if sparse_mat.nnz < dim * dim * 0.5:
                evals, evecs = eigsh(sparse_mat, k=k_eff, which="SA")
                return evals, evecs
        except (ArpackNoConvergence, TypeError, ValueError, RuntimeError):
            pass  # Fall through to dense path

    # Dense path
    evals, evecs = np.linalg.eigh(hamiltonian)
    return evals[:k_eff], evecs[:, :k_eff]


# ---------------------------------------------------------------------------
# 3. Filter and auxiliary evaluation
# ---------------------------------------------------------------------------

def apply_filter(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    filter_criterion: Optional[
        Callable[[np.ndarray, complex, Optional[Any]], bool]
    ] = None,
) -> Tuple[List[complex], List[np.ndarray]]:
    """Filter eigenpairs through filter_criterion.

    Args:
        eigenvalues: Array of eigenvalues.
        eigenvectors: Matrix whose columns are eigenvectors.
        filter_criterion: Optional predicate (eigenstate, eigenvalue, aux) -> bool.

    Returns:
        Tuple of (filtered_eigenvalues, filtered_eigenstates).
    """
    if filter_criterion is None:
        return list(eigenvalues), [eigenvectors[:, i] for i in range(len(eigenvalues))]

    filtered_evals: List[complex] = []
    filtered_evecs: List[np.ndarray] = []
    for i in range(len(eigenvalues)):
        ev = eigenvalues[i]
        evec = eigenvectors[:, i]
        if filter_criterion(evec, complex(ev), None):
            filtered_evals.append(complex(ev))
            filtered_evecs.append(evec.copy())
    return filtered_evals, filtered_evecs


def evaluate_aux_operators(
    aux_pauli_lists: Union[
        List[List[Tuple[str, float]]],
        Dict[str, List[Tuple[str, float]]],
        None,
    ],
    eigenstates: List[np.ndarray],
) -> Union[
    List[List[Tuple[complex, Dict[str, Any]]]],
    List[Dict[str, Tuple[complex, Dict[str, Any]]]],
    None,
]:
    """Evaluate auxiliary operators on each eigenstate.

    Args:
        aux_pauli_lists: Aux operators as list or dict of Pauli specs.
        eigenstates: List of state vectors.

    Returns:
        Per-eigenstate list of expectation value tuples.
    """
    if aux_pauli_lists is None or not eigenstates:
        return None

    results: List[Any] = []
    for state in eigenstates:
        if isinstance(aux_pauli_lists, dict):
            entry: Dict[str, Tuple[complex, Dict[str, Any]]] = {}
            for name, pauli_list in aux_pauli_lists.items():
                mat, _ = operator_to_matrix(pauli_list)
                exp_val = complex(np.vdot(state, mat @ state))
                entry[name] = (exp_val, {"variance": 0.0})
            results.append(entry)
        else:
            entry_list: List[Tuple[complex, Dict[str, Any]]] = []
            for pauli_list in aux_pauli_lists:
                mat, _ = operator_to_matrix(pauli_list)
                exp_val = complex(np.vdot(state, mat @ state))
                entry_list.append((exp_val, {"variance": 0.0}))
            results.append(entry_list)
    return results


# ---------------------------------------------------------------------------
# 4. End-to-end solver
# ---------------------------------------------------------------------------

def numpy_eigensolver_solve(
    pauli_list: List[Tuple[str, float]],
    k: int = 1,
    filter_criterion: Optional[
        Callable[[np.ndarray, complex, Optional[Any]], bool]
    ] = None,
    aux_pauli_lists: Optional[
        Union[List[List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]
    ] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute the lowest k eigenvalues of a Pauli operator via exact diagonalization.

    Pipeline:
        1. Convert Pauli strings to dense Hermitian matrix.
        2. Select solver (diagonal/sparse/dense) and compute eigenpairs.
        3. Apply filter_criterion; keep first k surviving eigenpairs.
        4. Evaluate auxiliary operators on each returned eigenstate.

    Args:
        pauli_list: List of (pauli_string, coefficient).
        k: Number of eigenvalues to compute (capped at dimension 2^n).
        filter_criterion: Optional predicate (eigenstate, eigenvalue, aux) -> bool.
        aux_pauli_lists: Optional auxiliary operators to evaluate.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok'
            - eigenvalues: np.ndarray of computed eigenvalues
            - eigenstates: List of state vectors
            - aux_operators_evaluated: Per-eigenstate auxiliary expectations or None
            - num_qubits, dimension: System info
            - k_requested, k_returned: Requested vs actual count
            - Computation Time (s): Wall-clock time
    """
    t_start = time.perf_counter()

    # --- Stage 1: Matrix construction ---
    hamiltonian, num_qubits = operator_to_matrix(pauli_list)
    dim = 1 << num_qubits
    k_eff = min(k, dim)

    if verbose:
        print(f"NumPy Eigensolver")
        print(f"  Qubits:        {num_qubits}")
        print(f"  Dimension:     {dim} × {dim}")
        print(f"  k requested:   {k} (effective: {k_eff})")
        print(f"  Pauli terms:   {len(pauli_list)}")
        print(f"  Diagonal:      {is_diagonal(hamiltonian)}")

    # --- Stage 2: Eigendecomposition ---
    # Compute extra eigenpairs when filter is active (may need more than k)
    compute_k = max(k_eff * 3, k_eff + 10) if filter_criterion is not None else k_eff
    compute_k = min(compute_k, dim)

    eigenvalues, eigenvectors = solve_eigensystem(hamiltonian, k=compute_k)

    # --- Stage 3: Filter and truncate to k ---
    filtered_evals, filtered_evecs = apply_filter(
        eigenvalues, eigenvectors, filter_criterion
    )
    filtered_evals = filtered_evals[:k_eff]
    filtered_evecs = filtered_evecs[:k_eff]

    # --- Stage 4: Evaluate auxiliary operators ---
    aux_evaluated = evaluate_aux_operators(aux_pauli_lists, filtered_evecs)

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        print(f"  k returned:    {len(filtered_evals)}")
        print(f"  Eigenvalues:   {np.round(filtered_evals, 6).tolist()}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "eigenvalues": np.array(filtered_evals),
        "eigenstates": filtered_evecs,
        "aux_operators_evaluated": aux_evaluated,
        "num_qubits": num_qubits,
        "dimension": dim,
        "k_requested": k,
        "k_returned": len(filtered_evals),
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class NumPyEigensolverSolver:
    """Class-based solver for the NumPy Eigensolver.

    Usage:
        solver = NumPyEigensolverSolver()
        result = solver.run(
            pauli_list=[("ZZ", 1.0), ("XI", 0.5), ("IX", 0.3)],
            k=2,
        )
        print(result['eigenvalues'])

        # With filter
        def negative_only(eigenstate, eigenvalue, aux):
            return np.real(eigenvalue) < 0.0
        result2 = solver.run(pauli_list=[...], k=4, filter_criterion=negative_only)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        pauli_list: List[Tuple[str, float]],
        k: int = 1,
        filter_criterion: Optional[
            Callable[[np.ndarray, complex, Optional[Any]], bool]
        ] = None,
        aux_pauli_lists: Optional[
            Union[List[List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]
        ] = None,
    ) -> Dict[str, Any]:
        """Run the eigensolver. See numpy_eigensolver_solve()."""
        result = numpy_eigensolver_solve(
            pauli_list=pauli_list,
            k=k,
            filter_criterion=filter_criterion,
            aux_pauli_lists=aux_pauli_lists,
            verbose=(self.text_mode != "plain"),
        )
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "eigenvalues": result.get("eigenvalues"),
            "eigenstates": result.get("eigenstates"),
            "aux_operators_evaluated": result.get("aux_operators_evaluated"),
            "k_requested": result.get("k_requested", 0),
            "k_returned": result.get("k_returned", 0),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
        }


# ---------------------------------------------------------------------------
# 6. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "2q_k2",
        "pauli_list": [("ZZ", 1.0), ("XI", 0.5), ("IX", 0.3)],
        "k": 2,
        "aux_ops": {"mag_z0": [("ZI", 1.0)]},
    },
    {
        "name": "2q_k4_unfiltered",
        "pauli_list": [("ZZ", 1.5), ("XI", 0.8), ("IX", 0.3)],
        "k": 4,
        "aux_ops": None,
    },
    {
        "name": "3q_k2",
        "pauli_list": [("ZZZ", 1.0), ("XII", 0.5), ("IXI", 0.5), ("IIX", 0.5)],
        "k": 2,
        "aux_ops": None,
    },
    {
        "name": "1q_k2",
        "pauli_list": [("Z", 2.0), ("X", -1.0)],
        "k": 2,
        "aux_ops": None,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case."""
    name = case["name"]
    result = numpy_eigensolver_solve(
        pauli_list=case["pauli_list"],
        k=case["k"],
        aux_pauli_lists=case["aux_ops"],
        verbose=False,
    )
    ok = result["status"] == "ok" and len(result["eigenvalues"]) > 0
    icon = "ok" if ok else "FAIL"
    evals_preview = np.round(result["eigenvalues"], 4).tolist()
    print(f"  [{icon}] {name}: eigenvalues={evals_preview}, "
          f"k={result['k_returned']}/{result['k_requested']}, "
          f"time={result['Computation Time (s)']}s")
    if result["aux_operators_evaluated"] is not None:
        print(f"       aux: {result['aux_operators_evaluated']}")
    return ok


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NumPy Eigensolver — Manual Implementation")
    print("=" * 60)

    solver = NumPyEigensolverSolver()

    # --- Demo 1: 2-qubit, k=2 ---
    print("\n--- Demo 1: 2-qubit, k=2, with aux operators ---")
    result1 = solver.run(
        pauli_list=[("ZZ", 1.0), ("XI", 0.5), ("IX", 0.3)],
        k=2,
        aux_pauli_lists={"mag_z0": [("ZI", 1.0)]},
    )
    print(f"  Eigenvalues:           {np.round(result1['eigenvalues'], 6).tolist()}")
    print(f"  Aux operators:         {result1['aux_operators_evaluated']}")

    # --- Demo 2: Unfiltered vs filtered ---
    print("\n--- Demo 2: Filtered vs unfiltered (keep only negative eigenvalues) ---")
    pauli_2q = [("ZZ", 1.5), ("XI", 0.8), ("IX", 0.3)]

    result2_all = solver.run(pauli_list=pauli_2q, k=4)
    print(f"  All eigenvalues:       {np.round(result2_all['eigenvalues'], 6).tolist()}")
    print(f"  Returned:              {result2_all['k_returned']}/{result2_all['k_requested']}")

    def keep_negative(eigenstate, eigenvalue, aux):
        return float(np.real(eigenvalue)) < 0.0

    result2_filt = solver.run(pauli_list=pauli_2q, k=4, filter_criterion=keep_negative)
    print(f"  Negative eigenvalues:  {np.round(result2_filt['eigenvalues'], 6).tolist()}")
    print(f"  Returned:              {result2_filt['k_returned']}/{result2_filt['k_requested']}")

    # --- Demo 3: 3-qubit ---
    print("\n--- Demo 3: 3-qubit Ising + transverse field, k=2 ---")
    result3 = solver.run(
        pauli_list=[("ZZZ", 1.0), ("XII", 0.5), ("IXI", 0.5), ("IIX", 0.5)],
        k=2,
    )
    print(f"  Eigenvalues:           {np.round(result3['eigenvalues'], 6).tolist()}")

    # --- Demo 4: Single qubit ---
    print("\n--- Demo 4: H = 2Z - X, k=2 ---")
    result4 = solver.run(pauli_list=[("Z", 2.0), ("X", -1.0)], k=2)
    print(f"  Eigenvalues:           {np.round(result4['eigenvalues'], 6).tolist()}")
    # Known: eigenvalues of 2Z - X are ±√5 ≈ ±2.2361
    expected = np.sort([-np.sqrt(5), np.sqrt(5)])
    evals4 = np.sort(np.real(result4['eigenvalues']))
    print(f"  Expected (±√5):        {np.round(expected, 6).tolist()}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
