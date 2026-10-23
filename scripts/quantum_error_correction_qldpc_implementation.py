"""Manual implementation of qLDPC (quantum Low-Density Parity-Check) codes.

Covers the full chain from classical LDPC to CSS and Hypergraph Product (HGP)
code construction, with practical validation and visualization.

Topics:
    1. Classical LDPC: parity-check matrix, Tanner graph, syndrome computation
    2. CSS codes: two binary matrices H_X, H_Z with commutation H_X H_Z^T = 0 (mod 2)
    3. Hypergraph Product (HGP): construct qLDPC checks from two classical codes
    4. Binary matrix rank over Z_2 for code dimension estimation

HGP Construction (standard form):
    H_X = [H1 ⊗ I_{n2} | I_{r1} ⊗ H2^T]
    H_Z = [I_{n1} ⊗ H2 | H1^T ⊗ I_{r2}]
    where H1: (r1, n1), H2: (r2, n2)

Prerequisites:
    pip install numpy networkx matplotlib pennylane

Components:
    - hamming_code / rep_code: Classical parity-check matrix generators
    - binary_matrix_rank: Z_2 rank via Gaussian elimination
    - hgp_code: Hypergraph Product CSS construction
    - css_commutes: Commutation check for CSS codes
    - build_tanner_graph / save_tanner_graph: Visualization
    - compute_syndrome: Classical syndrome computation
    - code_dimension: CSS code dimension estimate
    - QLDPCSolver: Class-based interface

Reference:
    SKILL.md — PennyLane qLDPC Skill
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


# ===================================================================
# 1. Classical code generators
# ===================================================================


def hamming_code(rank: int) -> np.ndarray:
    """Generate the binary Hamming parity-check matrix of size (rank, 2^rank - 1).

    Each column is the binary representation of the column index (1-indexed).

    Args:
        rank: Hamming code rank (>= 2).

    Returns:
        Binary matrix of shape (rank, 2^rank - 1).

    Raises:
        ValueError: If rank < 2.
    """
    if rank < 2:
        raise ValueError("rank must be >= 2 for a non-trivial Hamming code")

    n_cols = (1 << rank) - 1
    cols = []
    for val in range(1, n_cols + 1):
        bits = [(val >> bit) & 1 for bit in range(rank)]
        cols.append(bits)

    return np.array(cols, dtype=int).T % 2


def rep_code(distance: int) -> np.ndarray:
    """Construct repetition-code parity checks of shape (d-1, d).

    Each row checks two adjacent bits: row i couples bit i and bit i+1.

    Args:
        distance: Code distance (>= 2).

    Returns:
        Binary matrix of shape (distance-1, distance).

    Raises:
        ValueError: If distance < 2.
    """
    if distance < 2:
        raise ValueError("distance must be >= 2")

    h = np.zeros((distance - 1, distance), dtype=int)
    for i in range(distance - 1):
        h[i, i] = 1
        h[i, i + 1] = 1
    return h


def ring_code(n_bits: int) -> np.ndarray:
    """Construct a ring (cycle) parity-check matrix of shape (n_bits, n_bits).

    Each check couples bit i and bit (i+1) mod n_bits.

    Args:
        n_bits: Number of bits (>= 3).

    Returns:
        Binary matrix of shape (n_bits, n_bits).
    """
    if n_bits < 3:
        raise ValueError("n_bits must be >= 3 for a ring code")
    h = np.zeros((n_bits, n_bits), dtype=int)
    for i in range(n_bits):
        h[i, i] = 1
        h[i, (i + 1) % n_bits] = 1
    return h


# ===================================================================
# 2. Binary linear algebra over Z_2
# ===================================================================


def binary_matrix_rank(binary_matrix: np.ndarray) -> int:
    """Compute matrix rank over Z_2 using Gaussian elimination modulo 2.

    Args:
        binary_matrix: Binary matrix (0/1 entries).

    Returns:
        Rank over Z_2.
    """
    m = np.array(binary_matrix, dtype=int) % 2
    rows, cols = m.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find pivot
        pivot = None
        for r in range(pivot_row, rows):
            if m[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        # Swap to pivot row
        if pivot != pivot_row:
            m[[pivot_row, pivot]] = m[[pivot, pivot_row]]

        # Eliminate other rows
        for r in range(rows):
            if r != pivot_row and m[r, col] == 1:
                m[r, :] ^= m[pivot_row, :]

        pivot_row += 1

    return pivot_row


def compute_syndrome(h: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Compute the classical syndrome s = H e mod 2.

    Args:
        h: Parity-check matrix of shape (r, n).
        error: Error vector of length n.

    Returns:
        Syndrome vector of length r.
    """
    h = np.array(h, dtype=int) % 2
    error = np.array(error, dtype=int) % 2
    return (h @ error) % 2


def nullspace_dimension(h: np.ndarray) -> int:
    """Compute dimension of the nullspace of H over Z_2.

    dim(null(H)) = n - rank(H).

    Args:
        h: Binary matrix of shape (r, n).

    Returns:
        Dimension of the nullspace.
    """
    n = h.shape[1]
    return n - binary_matrix_rank(h)


# ===================================================================
# 3. CSS code construction and validation
# ===================================================================


def css_commutes(h_x: np.ndarray, h_z: np.ndarray) -> bool:
    """Check CSS commutation condition: H_X H_Z^T = 0 (mod 2).

    Args:
        h_x: X-type stabilizer matrix.
        h_z: Z-type stabilizer matrix.

    Returns:
        True if H_X H_Z^T ≡ 0 (mod 2).
    """
    return bool(
        np.all(
            (np.array(h_x, dtype=int) @ np.array(h_z, dtype=int).T) % 2 == 0
        )
    )


def code_dimension(h_x: np.ndarray, h_z: np.ndarray) -> int:
    """Estimate CSS code dimension: k = n - rank(H_X) - rank(H_Z).

    This is the number of encoded logical qubits.

    Args:
        h_x: X-type stabilizer matrix of shape (r_x, n).
        h_z: Z-type stabilizer matrix of shape (r_z, n).

    Returns:
        Estimated code dimension k (non-negative).

    Raises:
        ValueError: If h_x and h_z have different column counts.
    """
    if h_x.shape[1] != h_z.shape[1]:
        raise ValueError(
            f"H_X has {h_x.shape[1]} columns but H_Z has {h_z.shape[1]}; "
            f"both must act on the same number of qubits"
        )
    n = h_x.shape[1]
    rank_x = binary_matrix_rank(h_x)
    rank_z = binary_matrix_rank(h_z)
    return max(0, n - rank_x - rank_z)


def validate_css_code(
    h_x: np.ndarray,
    h_z: np.ndarray,
    label: str = "CSS",
) -> Dict[str, Any]:
    """Validate a CSS code: commutation check and dimension estimate.

    Args:
        h_x: X stabilizer matrix.
        h_z: Z stabilizer matrix.
        label: Label for print output.

    Returns:
        Dict with keys: commutes, n_qubits, rank_x, rank_z, code_dimension,
        n_x_stabilizers, n_z_stabilizers.
    """
    commutes = css_commutes(h_x, h_z)
    n = h_x.shape[1]
    rank_x = binary_matrix_rank(h_x)
    rank_z = binary_matrix_rank(h_z)
    k = max(0, n - rank_x - rank_z)

    result = {
        "commutes": commutes,
        "n_qubits": n,
        "rank_x": rank_x,
        "rank_z": rank_z,
        "code_dimension": k,
        "n_x_stabilizers": h_x.shape[0],
        "n_z_stabilizers": h_z.shape[0],
    }

    print(f"--- {label} Validation ---")
    print(f"  Qubits (n):            {n}")
    print(f"  H_X shape:             {h_x.shape}")
    print(f"  H_Z shape:             {h_z.shape}")
    print(f"  rank(H_X):             {rank_x}")
    print(f"  rank(H_Z):             {rank_z}")
    print(f"  Code dimension (k):    {k}")
    print(f"  H_X H_Z^T = 0 mod 2:   {commutes}")
    print()

    return result


# ===================================================================
# 4. Hypergraph Product (HGP) construction
# ===================================================================


def hgp_code(
    h1: np.ndarray,
    h2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct HGP CSS matrices using the standard block form.

    H_X = [H1 ⊗ I_{n2} | I_{r1} ⊗ H2^T]
    H_Z = [I_{n1} ⊗ H2 | H1^T ⊗ I_{r2}]

    where H1 has shape (r1, n1), H2 has shape (r2, n2).

    The resulting H_X has shape:
        (r1*n2 + n1*r2, n1*n2 + r1*r2)
    wait — let me be precise:
        H1 ⊗ I_{n2}:  (r1*n2, n1*n2)
        I_{r1} ⊗ H2^T: (r1*r2, r1*n2) → wait, H2^T is (n2, r2), so I_{r1} ⊗ H2^T is (r1*n2, r1*r2)

    Actually:
        left_x = H1 ⊗ I_{n2}:    shape (r1, n1) ⊗ (n2, n2) = (r1*n2, n1*n2)
        right_x = I_{r1} ⊗ H2^T: shape (r1, r1) ⊗ (n2, r2) = (r1*n2, r1*r2)
        H_X = [left_x | right_x]: shape (r1*n2, n1*n2 + r1*r2)

        left_z = I_{n1} ⊗ H2:    shape (n1, n1) ⊗ (r2, n2) = (n1*r2, n1*n2)
        right_z = H1^T ⊗ I_{r2}:  shape (n1, r1) ⊗ (r2, r2) = (n1*r2, r1*r2)
        H_Z = [left_z | right_z]: shape (n1*r2, n1*n2 + r1*r2)

    Args:
        h1: First classical parity-check matrix of shape (r1, n1).
        h2: Second classical parity-check matrix of shape (r2, n2).

    Returns:
        Tuple of (H_X, H_Z) as binary matrices.
    """
    h1 = np.array(h1, dtype=int) % 2
    h2 = np.array(h2, dtype=int) % 2

    r1, n1 = h1.shape
    r2, n2 = h2.shape

    # H_X blocks
    left_x = np.kron(h1, np.eye(n2, dtype=int))
    right_x = np.kron(np.eye(r1, dtype=int), h2.T)
    h_x = np.hstack([left_x, right_x]) % 2

    # H_Z blocks
    left_z = np.kron(np.eye(n1, dtype=int), h2)
    right_z = np.kron(h1.T, np.eye(r2, dtype=int))
    h_z = np.hstack([left_z, right_z]) % 2

    return h_x, h_z


def hgp_info(
    h1: np.ndarray,
    h2: np.ndarray,
) -> Dict[str, Any]:
    """Compute detailed information about an HGP construction.

    Args:
        h1: First classical code matrix.
        h2: Second classical code matrix.

    Returns:
        Dict with shapes, ranks, code dimension, and commutation status.
    """
    r1, n1 = h1.shape
    r2, n2 = h2.shape
    h_x, h_z = hgp_code(h1, h2)

    n_qubits = h_x.shape[1]
    k1_classical = nullspace_dimension(h1)  # n1 - rank(h1)
    k2_classical = nullspace_dimension(h2)  # n2 - rank(h2)

    return {
        "h1_shape": (r1, n1),
        "h2_shape": (r2, n2),
        "h_x_shape": h_x.shape,
        "h_z_shape": h_z.shape,
        "n_physical_qubits": n_qubits,
        "n_x_stabilizers": h_x.shape[0],
        "n_z_stabilizers": h_z.shape[0],
        "classical_k1": k1_classical,
        "classical_k2": k2_classical,
        "quantum_k": code_dimension(h_x, h_z),
        "commutes": css_commutes(h_x, h_z),
        "sparsity_x": float(np.mean(h_x)) if h_x.size > 0 else 0.0,
        "sparsity_z": float(np.mean(h_z)) if h_z.size > 0 else 0.0,
    }


# ===================================================================
# 5. Tanner graph visualization
# ===================================================================


def build_tanner_graph(h: np.ndarray) -> Any:
    """Build a bipartite Tanner graph from parity-check matrix H.

    Variable nodes: v0..v_{n-1} (bipartite=0)
    Check nodes:    c0..c_{r-1} (bipartite=1)
    Edges:          (c_i, v_j) if H[i, j] == 1

    Args:
        h: Parity-check matrix of shape (r, n).

    Returns:
        NetworkX Graph object.

    Raises:
        ImportError: If networkx is not installed.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required for Tanner graph construction")

    checks, bits = h.shape
    graph = nx.Graph()

    for j in range(bits):
        graph.add_node(f"v{j}", bipartite=0)
    for i in range(checks):
        graph.add_node(f"c{i}", bipartite=1)

    for i in range(checks):
        for j in range(bits):
            if h[i, j] == 1:
                graph.add_edge(f"c{i}", f"v{j}")

    return graph


def save_tanner_graph(
    h: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Tanner Graph",
) -> Path:
    """Render and save Tanner graph to an image file.

    Args:
        h: Parity-check matrix.
        output_path: Output file path. Defaults to 'tanner_graph.png' in cwd.
        title: Plot title.

    Returns:
        Path to the saved image.

    Raises:
        ImportError: If matplotlib or networkx is not installed.
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        raise ImportError(
            "matplotlib and networkx are required for Tanner graph visualization"
        )

    if output_path is None:
        output_path = Path("tanner_graph.png")

    graph = build_tanner_graph(h)

    check_nodes = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]
    var_nodes = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]

    pos = {}
    for idx, n in enumerate(check_nodes):
        pos[n] = (0.0, float(-idx))
    for idx, n in enumerate(var_nodes):
        pos[n] = (2.0, float(-idx))

    plt.figure(figsize=(8, 5))
    nx.draw_networkx_nodes(
        graph, pos, nodelist=check_nodes, node_color="#d95f02", node_size=800,
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=var_nodes, node_color="#1b9e77", node_size=800,
    )
    nx.draw_networkx_edges(graph, pos, width=1.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="white")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return output_path


# ===================================================================
# 6. Sparsity analysis
# ===================================================================


def matrix_sparsity(h: np.ndarray) -> Dict[str, Any]:
    """Analyze sparsity of a parity-check matrix.

    Args:
        h: Binary matrix.

    Returns:
        Dict with density, nonzeros, total_entries, sparsity_ratio.
    """
    total = h.size
    nonzeros = int(np.sum(h))
    density = nonzeros / total if total > 0 else 0.0
    return {
        "density": density,
        "nonzero_entries": nonzeros,
        "total_entries": total,
        "sparsity_ratio": 1.0 - density,
    }


# ===================================================================
# 7. Class-based interface
# ===================================================================


class QLDPCSolver:
    """Class-based solver for qLDPC code construction and validation.

    Usage:
        solver = QLDPCSolver()
        result = solver.build_hgp(h1=rep_code(3), h2=rep_code(4))
        print(result['code_dimension'])
    """

    def __init__(self):
        self._check_dependencies()

    @staticmethod
    def _check_dependencies() -> None:
        """Check that required packages are available."""
        missing = []
        if not HAS_NETWORKX:
            missing.append("networkx")
        if not HAS_MATPLOTLIB:
            missing.append("matplotlib")
        if not HAS_PENNYLANE:
            missing.append("pennylane")
        if missing:
            print(f"Warning: missing optional packages: {', '.join(missing)}")
            print("  Install with: pip install " + " ".join(missing))

    def classical_demo(
        self,
        h: Optional[np.ndarray] = None,
        error: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run classical LDPC demo: matrix, syndrome, Tanner graph.

        Args:
            h: Parity-check matrix. Defaults to Hamming(3).
            error: Error vector. Defaults to [1,0,1,0,0,0,0].

        Returns:
            Dict with matrix, syndrome, tanner_graph_path.
        """
        if h is None:
            h = hamming_code(3)
        if error is None:
            error = np.array([1, 0, 1, 0, 0, 0, 0], dtype=int)

        syndrome = compute_syndrome(h, error)

        print("=== Classical LDPC ===")
        print(f"H shape: {h.shape}")
        print(f"H:\n{h}")
        print(f"Error e: {error}")
        print(f"Syndrome s = H e mod 2: {syndrome}")

        graph_path = None
        if HAS_MATPLOTLIB and HAS_NETWORKX:
            graph_path = save_tanner_graph(h, Path("tanner_graph.png"))
            print(f"Tanner graph saved: {graph_path}")

        return {
            "h": h,
            "error": error,
            "syndrome": syndrome,
            "tanner_graph_path": graph_path,
        }

    def css_demo(
        self,
        h_x: Optional[np.ndarray] = None,
        h_z: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run CSS demo with commutation check and code dimension.

        Args:
            h_x: X stabilizer matrix. Defaults to rep_code(3).
            h_z: Z stabilizer matrix. Defaults to [[1,1,1]].

        Returns:
            Validation result dict.
        """
        if h_x is None:
            h_x = rep_code(3)
        if h_z is None:
            h_z = np.array([[1, 1, 1]], dtype=int)

        return validate_css_code(h_x, h_z, label="CSS")

    def build_hgp(
        self,
        h1: Optional[np.ndarray] = None,
        h2: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Build and validate an HGP code.

        Args:
            h1: First classical code. Defaults to rep_code(3).
            h2: Second classical code. Defaults to rep_code(4).

        Returns:
            Detailed HGP info dict.
        """
        if h1 is None:
            h1 = rep_code(3)
        if h2 is None:
            h2 = rep_code(4)

        info = hgp_info(h1, h2)
        h_x, h_z = hgp_code(h1, h2)

        print("=== HGP Construction ===")
        print(f"H1: {info['h1_shape']}, H2: {info['h2_shape']}")
        print(f"H_X: {info['h_x_shape']}, H_Z: {info['h_z_shape']}")
        print(f"Physical qubits:  {info['n_physical_qubits']}")
        print(f"Logical qubits k: {info['quantum_k']}")
        print(f"X stabilizers:    {info['n_x_stabilizers']}")
        print(f"Z stabilizers:    {info['n_z_stabilizers']}")
        print(f"CSS commutation:  {info['commutes']}")
        print(f"H_X density:      {info['sparsity_x']:.4f}")
        print(f"H_Z density:      {info['sparsity_z']:.4f}")
        print()

        return {**info, "h_x": h_x, "h_z": h_z}

    def pennylane_demo(self) -> Optional[float]:
        """Run a tiny PennyLane QNode for context.

        Returns:
            Expectation value <Z>, or None if PennyLane unavailable.
        """
        if not HAS_PENNYLANE:
            print("PennyLane not available — skipping QNode demo.")
            return None

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(theta: float) -> float:
            qml.Hadamard(wires=0)
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        theta = 0.3
        value = float(circuit(theta))
        print("=== PennyLane Tiny Demo ===")
        print(f"QNode <Z> at θ={theta}: {value:.6f}")
        print()
        return value

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete qLDPC pipeline.

        Returns:
            Dict with results from all stages.
        """
        print("=" * 60)
        print("qLDPC Full Pipeline")
        print("=" * 60)
        print()

        classical = self.classical_demo()
        css = self.css_demo()
        hgp = self.build_hgp()
        pl_result = self.pennylane_demo()

        return {
            "classical": classical,
            "css": css,
            "hgp": hgp,
            "pennylane_expectation": pl_result,
        }


# ===================================================================
# 8. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "Hamming(3) rank check",
        "func": "rank_check",
        "h": lambda: hamming_code(3),
        "expected_rank": 3,
    },
    {
        "name": "rep_code(3) rank check",
        "func": "rank_check",
        "h": lambda: rep_code(3),
        "expected_rank": 2,
    },
    {
        "name": "rep_code(3) × rep_code(4) HGP commutes",
        "func": "hgp_commutes",
        "h1": lambda: rep_code(3),
        "h2": lambda: rep_code(4),
    },
    {
        "name": "rep_code(3) CSS with [[1,1,1]] commutes",
        "func": "css_commutes_check",
        "h_x": lambda: rep_code(3),
        "h_z": lambda: np.array([[1, 1, 1]], dtype=int),
    },
    {
        "name": "Hamming(3) × Hamming(3) HGP",
        "func": "hgp_commutes",
        "h1": lambda: hamming_code(3),
        "h2": lambda: hamming_code(3),
    },
    {
        "name": "ring_code(5) rank check",
        "func": "rank_check",
        "h": lambda: ring_code(5),
        "expected_rank": 4,
    },
    {
        "name": "rep_code(5) rank check",
        "func": "rank_check",
        "h": lambda: rep_code(5),
        "expected_rank": 4,
    },
    {
        "name": "rep_code(2) × rep_code(3) code dim > 0",
        "func": "hgp_dimension",
        "h1": lambda: rep_code(2),
        "h2": lambda: rep_code(3),
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case."""
    name = case["name"]
    func = case["func"]

    try:
        if func == "rank_check":
            h = case["h"]()
            rank = binary_matrix_rank(h)
            expected = case["expected_rank"]
            ok = rank == expected
            icon = "ok" if ok else "FAIL"
            print(f"  [{icon}] {name}: rank={rank}, expected={expected}")

        elif func == "hgp_commutes":
            h1, h2 = case["h1"](), case["h2"]()
            h_x, h_z = hgp_code(h1, h2)
            commutes = css_commutes(h_x, h_z)
            ok = commutes
            icon = "ok" if ok else "FAIL"
            info = hgp_info(h1, h2)
            print(f"  [{icon}] {name}: commutes={commutes}, "
                  f"n_phys={info['n_physical_qubits']}, k={info['quantum_k']}")

        elif func == "css_commutes_check":
            h_x, h_z = case["h_x"](), case["h_z"]()
            commutes = css_commutes(h_x, h_z)
            ok = commutes
            icon = "ok" if ok else "FAIL"
            k = code_dimension(h_x, h_z)
            print(f"  [{icon}] {name}: commutes={commutes}, k={k}")

        elif func == "hgp_dimension":
            h1, h2 = case["h1"](), case["h2"]()
            info = hgp_info(h1, h2)
            ok = info["commutes"] and info["quantum_k"] > 0
            icon = "ok" if ok else "FAIL"
            print(f"  [{icon}] {name}: k={info['quantum_k']} (>0 expected), "
                  f"n={info['n_physical_qubits']}")

        else:
            print(f"  [SKIP] {name}: unknown function type")
            return False

        return ok

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False


# ===================================================================
# 9. Main
# ===================================================================

def main() -> None:
    """Run the complete qLDPC tutorial pipeline."""
    solver = QLDPCSolver()
    solver.run_full_pipeline()

    print("=" * 60)
    print("Known Test Cases")
    print("=" * 60)
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
