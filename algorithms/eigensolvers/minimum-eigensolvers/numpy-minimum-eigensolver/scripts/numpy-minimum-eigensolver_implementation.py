"""Manual implementation of NumPy Minimum Eigensolver.

Finds the minimum eigenvalue (and corresponding eigenstate) of a qubit operator
using exact classical diagonalization via NumPy, then selecting the feasible
eigenpair with the smallest eigenvalue after optional filtering.

This is a classical reference solver — no quantum circuit construction or
execution is involved. It serves as a deterministic baseline for benchmarking
quantum eigensolvers like VQE.

Components:
    - operator_to_matrix: Convert a Pauli-string operator to dense matrix
    - compute_all_eigenpairs: Full eigendecomposition via numpy.linalg.eigh
    - find_minimum_eigenpair: Apply filter and select minimum eigenvalue
    - numpy_minimum_eigensolver_solve: End-to-end solver
    - NumPyMinimumEigensolverSolver: Class-based interface

Reference:
    SKILL.md — NumPy Minimum Eigensolver
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# 1. Operator utilities
# ---------------------------------------------------------------------------

# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI_MAP: Dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Convert a Pauli string (e.g. 'ZIX') to its dense matrix representation.

    Args:
        pauli_str: String of characters I, X, Y, Z. Length determines qubit count.

    Returns:
        Dense matrix of shape (2^n, 2^n) where n = len(pauli_str).
    """
    result = np.array([[1.0]], dtype=np.complex128)
    for ch in pauli_str:
        result = np.kron(result, _PAULI_MAP.get(ch, _I))
    return result


def operator_to_matrix(
    pauli_list: List[Tuple[str, float]],
) -> Tuple[np.ndarray, int]:
    """Build a dense Hermitian matrix from a list of (pauli_string, coefficient).

    H = Σᵢ cᵢ Pᵢ where Pᵢ ∈ {I, X, Y, Z}⊗ⁿ

    Args:
        pauli_list: List of (pauli_string, coefficient) tuples.

    Returns:
        Tuple of (matrix, num_qubits).

    Raises:
        ValueError: If pauli strings have inconsistent lengths.
    """
    if not pauli_list:
        raise ValueError("pauli_list must not be empty")

    num_qubits = len(pauli_list[0][0])
    matrix = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=np.complex128)

    for pauli_str, coeff in pauli_list:
        if len(pauli_str) != num_qubits:
            raise ValueError(
                f"Pauli string '{pauli_str}' has length {len(pauli_str)}, "
                f"expected {num_qubits}"
            )
        matrix += coeff * pauli_string_to_matrix(pauli_str)

    return matrix, num_qubits


# ---------------------------------------------------------------------------
# 2. Eigendecomposition and minimum selection
# ---------------------------------------------------------------------------

def compute_all_eigenpairs(
    hamiltonian: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all eigenvalues and eigenvectors of a Hermitian matrix.

    Uses numpy.linalg.eigh for Hermitian matrices — eigenvalues are guaranteed
    real and sorted ascending.

    Args:
        hamiltonian: Hermitian matrix of shape (d, d).

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvalues[i] corresponds
        to eigenvectors[:, i] (column i).
    """
    evals, evecs = np.linalg.eigh(hamiltonian)
    return evals, evecs


def find_minimum_eigenpair(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    filter_criterion: Optional[
        Callable[[np.ndarray, complex, Optional[Any]], bool]
    ] = None,
    aux_values: Optional[List[Any]] = None,
) -> Tuple[Optional[complex], Optional[np.ndarray], Optional[int]]:
    """Select the minimum eigenvalue (and eigenstate) that passes the filter.

    Iterates eigenvalues in ascending order; returns the first pair that
    satisfies filter_criterion. If no pair passes, returns (None, None, None).

    Args:
        eigenvalues: 1-D array of eigenvalues (sorted ascending).
        eigenvectors: 2-D array where columns are eigenvectors.
        filter_criterion: Callable (eigenstate, eigenvalue, aux_value) -> bool.
            If None, all eigenpairs are accepted.
        aux_values: Optional auxiliary values to pass to filter.

    Returns:
        Tuple of (eigenvalue, eigenstate, index) or (None, None, None).
    """
    for i, ev in enumerate(eigenvalues):
        evec = eigenvectors[:, i]
        aux = aux_values[i] if aux_values is not None else None
        if filter_criterion is None or filter_criterion(evec, ev, aux):
            return complex(ev), evec.copy(), i
    return None, None, None


def evaluate_aux_operators(
    aux_pauli_lists: Union[
        List[List[Tuple[str, float]]],
        Dict[str, List[Tuple[str, float]]],
        None,
    ],
    eigenstate: np.ndarray,
) -> Union[List[Tuple[complex, Dict[str, Any]]],
           Dict[str, Tuple[complex, Dict[str, Any]]],
           None]:
    """Evaluate auxiliary operators on a given eigenstate.

    Computes ⟨ψ|Aⱼ|ψ⟩ for each auxiliary operator Aⱼ.

    Args:
        aux_pauli_lists: List or dict of Pauli operator specs, or None.
        eigenstate: State vector to evaluate on.

    Returns:
        List/dict of (expectation_value, {"variance": 0.0}) tuples, or None.
    """
    if aux_pauli_lists is None:
        return None

    if isinstance(aux_pauli_lists, dict):
        result: Dict[str, Tuple[complex, Dict[str, Any]]] = {}
        for name, pauli_list in aux_pauli_lists.items():
            mat, _ = operator_to_matrix(pauli_list)
            exp_val = complex(np.vdot(eigenstate, mat @ eigenstate))
            result[name] = (exp_val, {"variance": 0.0})
        return result
    else:
        result_list: List[Tuple[complex, Dict[str, Any]]] = []
        for pauli_list in aux_pauli_lists:
            mat, _ = operator_to_matrix(pauli_list)
            exp_val = complex(np.vdot(eigenstate, mat @ eigenstate))
            result_list.append((exp_val, {"variance": 0.0}))
        return result_list


# ---------------------------------------------------------------------------
# 3. End-to-end solver
# ---------------------------------------------------------------------------

def numpy_minimum_eigensolver_solve(
    pauli_list: List[Tuple[str, float]],
    aux_pauli_lists: Optional[
        Union[List[List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]
    ] = None,
    filter_criterion: Optional[
        Callable[[np.ndarray, complex, Optional[Any]], bool]
    ] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Find the minimum eigenvalue of a Pauli operator via exact diagonalization.

    Pipeline:
        1. Convert Pauli strings to dense Hermitian matrix.
        2. Compute all eigenpairs via numpy.linalg.eigh.
        3. Apply filter_criterion and select the minimum feasible eigenpair.
        4. Evaluate auxiliary operators on the selected eigenstate.

    Args:
        pauli_list: List of (pauli_string, coefficient) defining the operator.
        aux_pauli_lists: Optional auxiliary operators to evaluate.
        filter_criterion: Optional callback (eigenstate, eigenvalue, aux)
            -> bool. Only eigenpairs passing the filter are eligible.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok' if a feasible eigenpair was found, else 'no_feasible'
            - eigenvalue: Minimum eigenvalue (complex or None)
            - eigenstate: Corresponding eigenstate (np.ndarray or None)
            - aux_operators_evaluated: Auxiliary expectation values or None
            - all_eigenvalues: All computed eigenvalues
            - Computation Time (s): Wall-clock time
            - num_qubits, dimension: System size info
    """
    t_start = time.perf_counter()

    # --- Stage 1: Matrix construction ---
    hamiltonian, num_qubits = operator_to_matrix(pauli_list)
    dim = 1 << num_qubits

    if verbose:
        print(f"NumPy Minimum Eigensolver")
        print(f"  Qubits:        {num_qubits}")
        print(f"  Dimension:     {dim} × {dim}")
        print(f"  Pauli terms:   {len(pauli_list)}")

    # --- Stage 2: Eigendecomposition ---
    eigenvalues, eigenvectors = compute_all_eigenpairs(hamiltonian)

    if verbose:
        print(f"  Eigenvalues:   [{eigenvalues[0]:.6f}, ..., {eigenvalues[-1]:.6f}]")

    # --- Stage 3: Select minimum feasible eigenpair ---
    eigenvalue, eigenstate, idx = find_minimum_eigenpair(
        eigenvalues, eigenvectors, filter_criterion
    )

    # --- Stage 4: Evaluate auxiliary operators ---
    aux_evaluated = None
    if eigenstate is not None and aux_pauli_lists is not None:
        aux_evaluated = evaluate_aux_operators(aux_pauli_lists, eigenstate)

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        status_str = "ok" if eigenvalue is not None else "no_feasible"
        print(f"  Status:        {status_str}")
        if eigenvalue is not None:
            print(f"  Min eigenvalue: {eigenvalue:.6f}")
            print(f"  Index:          {idx}")
        print(f"  Time:           {comp_time}s")

    return {
        "status": "ok" if eigenvalue is not None else "no_feasible",
        "eigenvalue": eigenvalue,
        "eigenstate": eigenstate,
        "aux_operators_evaluated": aux_evaluated,
        "all_eigenvalues": eigenvalues,
        "Computation Time (s)": comp_time,
        "num_qubits": num_qubits,
        "dimension": dim,
    }


# ---------------------------------------------------------------------------
# 4. Class-based interface
# ---------------------------------------------------------------------------

class NumPyMinimumEigensolverSolver:
    """Class-based solver for the NumPy Minimum Eigensolver.

    Usage:
        solver = NumPyMinimumEigensolverSolver()
        result = solver.run(
            pauli_list=[("ZI", 1.0), ("IZ", 1.0), ("XX", 0.5)]
        )
        print(result['eigenvalue'])

        # With filter criterion
        def sym_filter(eigenstate, eigenvalue, aux_values):
            return np.real(eigenvalue) < 0.0
        result2 = solver.run(pauli_list=[...], filter_criterion=sym_filter)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        pauli_list: List[Tuple[str, float]],
        aux_pauli_lists: Optional[
            Union[List[List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]
        ] = None,
        filter_criterion: Optional[
            Callable[[np.ndarray, complex, Optional[Any]], bool]
        ] = None,
    ) -> Dict[str, Any]:
        """Run the minimum eigensolver. See numpy_minimum_eigensolver_solve()."""
        result = numpy_minimum_eigensolver_solve(
            pauli_list=pauli_list,
            aux_pauli_lists=aux_pauli_lists,
            filter_criterion=filter_criterion,
            verbose=(self.text_mode != "plain"),
        )
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "eigenvalue": result.get("eigenvalue"),
            "eigenstate": result.get("eigenstate"),
            "aux_operators_evaluated": result.get("aux_operators_evaluated"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
        }


# ---------------------------------------------------------------------------
# 5. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "H=ZI+IZ+0.5XX",
        "pauli_list": [("ZI", 1.0), ("IZ", 1.0), ("XX", 0.5)],
        "aux_ops": {"magnetization": [("ZZ", 1.0)]},
        "description": "2-qubit Heisenberg-like model",
    },
    {
        "name": "H=ZZ+XI+IX",
        "pauli_list": [("ZZ", 2.0), ("XI", 1.0), ("IX", -0.5)],
        "aux_ops": None,
        "description": "2-qubit with Zeeman terms",
    },
    {
        "name": "H=Z (single qubit)",
        "pauli_list": [("Z", 1.0)],
        "aux_ops": {"X_expectation": [("X", 1.0)]},
        "description": "Single-qubit Pauli Z",
    },
    {
        "name": "H=ZZZ+XII+IXI+IIX",
        "pauli_list": [("ZZZ", 1.0), ("XII", 0.5), ("IXI", 0.5), ("IIX", 0.5)],
        "aux_ops": None,
        "description": "3-qubit Ising + transverse field",
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case."""
    name = case["name"]
    result = numpy_minimum_eigensolver_solve(
        pauli_list=case["pauli_list"],
        aux_pauli_lists=case["aux_ops"],
        verbose=False,
    )
    ok = result["status"] == "ok" and result["eigenvalue"] is not None
    ev_str = f"{result['eigenvalue']:.6f}" if result["eigenvalue"] is not None else "None"
    icon = "ok" if ok else "FAIL"
    print(f"  [{icon}] {name}: eigenvalue={ev_str}, "
          f"time={result['Computation Time (s)']}s")
    if result["aux_operators_evaluated"] is not None:
        print(f"       aux: {result['aux_operators_evaluated']}")
    return ok


def run_filter_example() -> bool:
    """Demonstrate the filter_criterion feature."""
    print("\n  --- Filter Example: keep only eigenvectors with negative ZZ expectation ---")
    pauli_list = [("ZZ", 2.0), ("XI", 1.0), ("IX", -0.5)]

    def negative_zz_filter(eigenstate, eigenvalue, aux_values):
        zz_mat = pauli_string_to_matrix("ZZ")
        zz_exp = float(np.real(np.vdot(eigenstate, zz_mat @ eigenstate)))
        return zz_exp < 0.0

    result = numpy_minimum_eigensolver_solve(
        pauli_list=pauli_list,
        filter_criterion=negative_zz_filter,
        verbose=False,
    )
    print(f"  Status: {result['status']}")
    print(f"  Eigenvalue: {result['eigenvalue']}")
    if result["eigenstate"] is not None:
        zz_mat = pauli_string_to_matrix("ZZ")
        zz_exp = float(np.real(np.vdot(result["eigenstate"], zz_mat @ result["eigenstate"])))
        print(f"  ⟨ZZ⟩ of selected state: {zz_exp:.6f} (< 0, as required)")
    return result["status"] == "ok"


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NumPy Minimum Eigensolver — Manual Implementation")
    print("=" * 60)

    solver = NumPyMinimumEigensolverSolver()

    # --- Demo 1: Basic 2-qubit operator ---
    print("\n--- Demo 1: H = ZI + IZ + 0.5·XX ---")
    result1 = solver.run(
        pauli_list=[("ZI", 1.0), ("IZ", 1.0), ("XX", 0.5)],
        aux_pauli_lists={"magnetization": [("ZZ", 1.0)]},
    )
    print(f"  Eigenvalue:     {result1['eigenvalue']:.6f}")
    print(f"  Aux operators:  {result1['aux_operators_evaluated']}")

    # --- Demo 2: 3-qubit operator ---
    print("\n--- Demo 2: H = ZZZ + 0.5·(XII + IXI + IIX) ---")
    result2 = solver.run(
        pauli_list=[("ZZZ", 1.0), ("XII", 0.5), ("IXI", 0.5), ("IIX", 0.5)],
    )
    print(f"  Eigenvalue:     {result2['eigenvalue']:.6f}")

    # --- Demo 3: Single qubit ---
    print("\n--- Demo 3: H = Z (single qubit) ---")
    result3 = solver.run(
        pauli_list=[("Z", 1.0)],
        aux_pauli_lists={"X_expectation": [("X", 1.0)]},
    )
    print(f"  Eigenvalue:     {result3['eigenvalue']:.6f}")
    print(f"  ⟨X⟩ on |0⟩:     {result3['aux_operators_evaluated']}")

    # --- Filter example ---
    run_filter_example()

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
