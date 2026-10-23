"""PennyLane qLDPC tutorial script.

This script demonstrates:
1) Classical LDPC basics (parity-check matrix, Tanner graph, syndrome)
2) CSS commutation checks and code-dimension estimate
3) Hypergraph Product (HGP) construction and commutation validation
4) A tiny PennyLane QNode context example
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib
import networkx as nx
import numpy as np
import pennylane as qml

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hamming_code(rank: int) -> np.ndarray:
    """Generate the binary Hamming parity-check matrix of size (rank, 2^rank - 1)."""
    if rank < 2:
        raise ValueError("rank must be >= 2 for a non-trivial Hamming code")

    n_cols = 2**rank - 1
    cols = []
    for val in range(1, n_cols + 1):
        bits = [(val >> bit) & 1 for bit in range(rank)]
        cols.append(bits)

    return np.array(cols, dtype=int).T % 2


def binary_matrix_rank(binary_matrix: np.ndarray) -> int:
    """Compute rank over Z_2 using Gaussian elimination modulo 2."""
    m = np.array(binary_matrix, dtype=int) % 2
    rows, cols = m.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        pivot = None
        for r in range(pivot_row, rows):
            if m[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != pivot_row:
            m[[pivot_row, pivot]] = m[[pivot, pivot_row]]

        for r in range(rows):
            if r != pivot_row and m[r, col] == 1:
                m[r, :] ^= m[pivot_row, :]

        pivot_row += 1

    return pivot_row


def rep_code(distance: int) -> np.ndarray:
    """Construct repetition-code parity checks of shape (distance-1, distance)."""
    if distance < 2:
        raise ValueError("distance must be >= 2")

    h = np.zeros((distance - 1, distance), dtype=int)
    for i in range(distance - 1):
        h[i, i] = 1
        h[i, i + 1] = 1
    return h


def hgp_code(h1: np.ndarray, h2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct HGP CSS matrices using the standard block form.

    H_X = [H1 ⊗ I_{n2} | I_{r1} ⊗ H2^T]
    H_Z = [I_{n1} ⊗ H2 | H1^T ⊗ I_{r2}]
    where H1 has shape (r1, n1), H2 has shape (r2, n2).
    """
    h1 = np.array(h1, dtype=int) % 2
    h2 = np.array(h2, dtype=int) % 2

    r1, n1 = h1.shape
    r2, n2 = h2.shape

    left_x = np.kron(h1, np.eye(n2, dtype=int))
    right_x = np.kron(np.eye(r1, dtype=int), h2.T)
    h_x = np.hstack([left_x, right_x]) % 2

    left_z = np.kron(np.eye(n1, dtype=int), h2)
    right_z = np.kron(h1.T, np.eye(r2, dtype=int))
    h_z = np.hstack([left_z, right_z]) % 2

    return h_x, h_z


def css_commutes(h_x: np.ndarray, h_z: np.ndarray) -> bool:
    """Check CSS commutation: H_X H_Z^T = 0 mod 2."""
    return np.all((np.array(h_x, dtype=int) @ np.array(h_z, dtype=int).T) % 2 == 0)


def build_tanner_graph(h: np.ndarray) -> nx.Graph:
    """Build a bipartite Tanner graph from parity-check matrix H."""
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


def save_tanner_graph(h: np.ndarray, output_path: Path) -> None:
    """Render and save Tanner graph to an image file."""
    graph = build_tanner_graph(h)

    check_nodes = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 1]
    var_nodes = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]

    pos = {}
    for idx, n in enumerate(check_nodes):
        pos[n] = (0.0, float(-idx))
    for idx, n in enumerate(var_nodes):
        pos[n] = (2.0, float(-idx))

    plt.figure(figsize=(8, 5))
    nx.draw_networkx_nodes(graph, pos, nodelist=check_nodes, node_color="#d95f02", node_size=800)
    nx.draw_networkx_nodes(graph, pos, nodelist=var_nodes, node_color="#1b9e77", node_size=800)
    nx.draw_networkx_edges(graph, pos, width=1.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="white")
    plt.title("Tanner Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def classical_ldpc_demo() -> np.ndarray:
    """Classical LDPC demo: matrix, Tanner graph, syndrome."""
    h = hamming_code(rank=3)
    print("=== Classical LDPC Demo ===")
    print(f"H shape: {h.shape}")
    print(h)

    output_path = Path(__file__).with_name("tanner_graph.png")
    save_tanner_graph(h, output_path)
    print(f"Saved Tanner graph to: {output_path}")

    error = np.array([1, 0, 1, 0, 0, 0, 0], dtype=int)
    syndrome = (h @ error) % 2
    print(f"Error vector e: {error}")
    print(f"Syndrome s = H e mod 2: {syndrome}")
    print()
    return h


def css_demo() -> Tuple[np.ndarray, np.ndarray]:
    """CSS demo with explicit commutation and binary-rank based dimension."""
    h_x = rep_code(3)  # shape (2, 3)
    h_z = np.array([[1, 1, 1]], dtype=int)  # commuting sample by default

    commutes = css_commutes(h_x, h_z)

    n = h_x.shape[1]
    rank_x = binary_matrix_rank(h_x)
    rank_z = binary_matrix_rank(h_z)
    k_estimate = n - rank_x - rank_z

    print("=== CSS Demo ===")
    print(f"H_X shape: {h_x.shape}")
    print(h_x)
    print(f"H_Z shape: {h_z.shape}")
    print(h_z)
    print(f"CSS commutation (H_X H_Z^T == 0 mod 2): {commutes}")
    print(f"rank_Z2(H_X): {rank_x}, rank_Z2(H_Z): {rank_z}")
    print(f"Estimated code dimension k = n - rank(H_X) - rank(H_Z): {k_estimate}")
    print()

    return h_x, h_z


def hgp_demo() -> Tuple[np.ndarray, np.ndarray]:
    """HGP demo with explicit commutation check."""
    h1 = rep_code(3)  # (2, 3)
    h2 = rep_code(4)  # (3, 4)

    h_x, h_z = hgp_code(h1, h2)
    commutes = css_commutes(h_x, h_z)

    print("=== HGP Demo ===")
    print(f"H1 shape: {h1.shape}, H2 shape: {h2.shape}")
    print(f"H_X shape: {h_x.shape}, H_Z shape: {h_z.shape}")
    print(f"HGP commutation (H_X H_Z^T == 0 mod 2): {commutes}")
    print()

    return h_x, h_z


def pennylane_demo() -> None:
    """Small PennyLane context demo with a tiny QNode."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(theta: float) -> float:
        qml.Hadamard(wires=0)
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = 0.3
    value = circuit(theta)
    print("=== PennyLane Tiny Demo ===")
    print(f"QNode expectation <Z> at theta={theta}: {value:.6f}")
    print()


def main() -> None:
    classical_ldpc_demo()

    h_x_css, h_z_css = css_demo()
    print(f"CSS sample commutes (must be True): {css_commutes(h_x_css, h_z_css)}")

    h_x_hgp, h_z_hgp = hgp_demo()
    print(f"HGP sample commutes (must be True): {css_commutes(h_x_hgp, h_z_hgp)}")

    pennylane_demo()


if __name__ == "__main__":
    main()
