"""Manual implementation of Quantum Approximate Optimization Algorithm (QAOA).

QAOA is a hybrid quantum-classical algorithm for combinatorial optimization.
This implementation solves the Max-Cut problem: partition graph vertices into
two sets to maximize the number of cut edges.

Algorithm:
    1. Encode the cost Hamiltonian H_C = Σ_{(u,v)} Z_u Z_v.
    2. Build QAOA circuit: p alternating layers of cost gates (CX-Rz-CX)
       and mixer gates (Rx).
    3. Optimize γ, β parameters via COBYLA to minimize ⟨H_C⟩.
    4. Decode the most probable basis state as the Max-Cut partition.

Circuit structure per layer:
    - Cost layer:  for each edge (u,v): CX(u,v) → Rz(2γ, v) → CX(u,v)
    - Mixer layer: for each qubit j: Rx(2β, j)

Components:
    - build_cost_hamiltonian: Build H_C matrix for Max-Cut
    - build_qaoa_circuit: Build p-layer QAOA circuit
    - qaoa_solve: End-to-end QAOA hybrid loop
    - QAOASolver: Class-based interface
    - plot_qaoa_results: Convergence + Max-Cut graph visualization

Reference:
    SKILL.md — QAOA
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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

from unitarylab.core import Circuit


# ===================================================================
# 1. Cost Hamiltonian construction
# ===================================================================

# Pauli Z and Identity for Hamiltonian building
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)


def build_cost_hamiltonian(edges: List[Tuple[int, int]], n_qubits: int) -> np.ndarray:
    """Build the Max-Cut cost Hamiltonian H_C = Σ_{(u,v)∈E} Z_u Z_v.

    The ground state of H_C corresponds to the Max-Cut solution:
    states where connected vertices have different spin values have
    lower energy (more negative).

    Args:
        edges: List of (u, v) edge tuples. Vertex indices in [0, n_qubits).
        n_qubits: Number of qubits = number of graph vertices.

    Returns:
        Cost Hamiltonian matrix of shape (2^n, 2^n), complex128.

    Raises:
        ValueError: If any vertex index is out of range.
    """
    dim = 1 << n_qubits
    h_c = np.zeros((dim, dim), dtype=np.complex128)

    for u, v in edges:
        if u >= n_qubits or v >= n_qubits or u < 0 or v < 0:
            raise ValueError(
                f"Edge ({u},{v}) contains vertex out of range [0, {n_qubits})"
            )

        # Build Z_u ⊗ Z_v with identity on all other qubits
        ops = [_I2] * n_qubits
        ops[u] = _Z
        ops[v] = _Z
        term = ops[0]
        for k in range(1, n_qubits):
            term = np.kron(term, ops[k])

        h_c += term

    return h_c


def exact_max_cut_energy(edges: List[Tuple[int, int]], n_qubits: int) -> float:
    """Compute the exact minimum energy (ground state) of H_C.

    Args:
        edges: Edge list.
        n_qubits: Number of qubits.

    Returns:
        Minimum eigenvalue of H_C.
    """
    h_c = build_cost_hamiltonian(edges, n_qubits)
    return float(np.linalg.eigvalsh(h_c)[0])


def compute_max_cut_value(
    bitstring: str, edges: List[Tuple[int, int]]
) -> int:
    """Count the number of cut edges for a given bitstring partition.

    An edge (u,v) is cut if bitstring[u] != bitstring[v].

    Args:
        bitstring: Binary string (e.g. "010110") encoding the partition.
        edges: Edge list.

    Returns:
        Number of cut edges.
    """
    return sum(1 for u, v in edges if bitstring[u] != bitstring[v])


# ===================================================================
# 2. QAOA circuit builder
# ===================================================================


def build_qaoa_circuit(
    params: np.ndarray,
    edges: List[Tuple[int, int]],
    n_qubits: int,
) -> Circuit:
    """Build the p-layer QAOA circuit for Max-Cut.

    Circuit:
        1. Initial state: H^{⊗n}|0⟩ = |+⟩^{⊗n} (uniform superposition)
        2. For each layer l = 0..p-1:
           a. Cost layer: for each edge (u,v):
              CX(u,v) → Rz(2·γ_l, v) → CX(u,v)
           b. Mixer layer: for each qubit j:
              Rx(2·β_l, j)

    Parameters: params = [γ_0, ..., γ_{p-1}, β_0, ..., β_{p-1}]

    Args:
        params: Flat parameter array of length 2p.
        edges: Edge list.
        n_qubits: Number of qubits.

    Returns:
        QAOA Circuit object.
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    qc = Circuit(n_qubits, name=f"QAOA_p{p}")

    # Initial state: uniform superposition |+⟩^{⊗n}
    for i in range(n_qubits):
        qc.h(i)

    # QAOA layers
    for layer in range(p):
        # Cost layer: e^{-i·γ_l·H_C} implemented as per-edge CX-Rz-CX
        for u, v in edges:
            u, v = int(u), int(v)
            qc.cx(u, v)
            qc.rz(2 * gammas[layer], v)
            qc.cx(u, v)

        # Mixer layer: e^{-i·β_l·H_mix} implemented as per-qubit Rx
        for j in range(n_qubits):
            qc.rx(2 * betas[layer], j)

    return qc


# ===================================================================
# 3. Energy evaluation
# ===================================================================


def evaluate_energy(
    params: np.ndarray,
    edges: List[Tuple[int, int]],
    n_qubits: int,
    h_cost: np.ndarray,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> float:
    """Evaluate ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩ for given parameters.

    Args:
        params: QAOA parameters [γ_0..γ_{p-1}, β_0..β_{p-1}].
        edges: Edge list.
        n_qubits: Number of qubits.
        h_cost: Precomputed cost Hamiltonian matrix.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Energy expectation value (float).
    """
    qc = build_qaoa_circuit(params, edges, n_qubits)
    # Execute with default |0⟩^n initial state (circuit applies H^{⊗n} internally)
    psi = np.asarray(
        qc.execute(backend=backend, device=device, dtype=dtype).state,
        dtype=complex,
    )
    energy = np.real(psi.conj().T @ h_cost @ psi)
    # Guard against NaN/Inf from extreme COBYLA probe points
    if not np.isfinite(energy):
        return 1e10  # large penalty steers optimizer away from invalid regions
    return float(energy)


# ===================================================================
# 4. End-to-end QAOA solver
# ===================================================================


def qaoa_solve(
    edges: List[Tuple[int, int]],
    n_qubits: int,
    layers: int = 2,
    max_iter: int = 100,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full QAOA hybrid quantum-classical loop for Max-Cut.

    Pipeline:
        1. Build the cost Hamiltonian H_C.
        2. Initialize random parameters γ, β ∈ [0, π).
        3. COBYLA optimization minimizing ⟨H_C⟩.
        4. Decode most probable basis state → bitstring → cut value.
        5. Generate convergence + Max-Cut graph plots.

    Args:
        edges: List of (u, v) edge tuples.
        n_qubits: Number of qubits = number of vertices.
        layers: Number of QAOA layers p. Higher = better approximation.
        max_iter: COBYLA maximum iterations.
        backend: Simulation backend ('torch').
        device: Compute device.
        dtype: Numerical dtype.
        seed: Random seed for parameter initialization.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - Optimal bitstring: Binary partition string
            - Max-Cut Value: Number of cut edges
            - Optimized Energy: Final ⟨H_C⟩ value
            - Exact Ground Energy: Minimum eigenvalue of H_C
            - Quantum Computation Time (s): Optimization wall-clock time
            - energy_history: Per-evaluation energy values
            - circuit: Final optimized QAOA Circuit
            - h_cost: Cost Hamiltonian matrix

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If n_qubits is too small for edges.
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy.optimize is required for QAOA. Install with: pip install scipy"
        )

    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

    np.random.seed(seed)

    total_start = time.perf_counter()

    # --- Stage 1: Hamiltonian + parameters ---
    h_cost = build_cost_hamiltonian(edges, n_qubits)
    exact_energy = float(np.linalg.eigvalsh(h_cost)[0])
    n_params = 2 * layers
    initial_params = np.random.uniform(0, np.pi, n_params)

    if verbose:
        print(f"QAOA Max-Cut Solver")
        print(f"  Qubits (vertices): {n_qubits}")
        print(f"  Edges:              {len(edges)}")
        print(f"  QAOA layers (p):    {layers}")
        print(f"  Parameters:         {n_params} (={layers} γ + {layers} β)")
        print(f"  Exact ground energy: {exact_energy:.6f}")
        print(f"  COBYLA max_iter:    {max_iter}")

    # --- Stage 2: Preview circuit ---
    qc_draw = build_qaoa_circuit(initial_params, edges, n_qubits)
    if verbose:
        print(f"  Preview circuit:     {qc_draw.get_num_qubits()} qubits")

    # --- Stage 3: COBYLA optimization ---
    if verbose:
        print(f"  Running COBYLA optimization...")

    energy_history: List[float] = []

    def objective(params_flat: np.ndarray) -> float:
        energy = evaluate_energy(
            params_flat, edges, n_qubits, h_cost,
            backend=backend, device=device, dtype=dtype,
        )
        energy_history.append(energy)
        return energy

    opt_start = time.perf_counter()
    opt_result = minimize(
        objective,
        x0=initial_params,
        method="COBYLA",
        options={"maxiter": max_iter, "disp": False},
    )
    opt_time = time.perf_counter() - opt_start

    best_params = opt_result.x
    best_energy = float(opt_result.fun)

    if verbose:
        n_evals = len(energy_history)
        print(f"  Optimization complete: {opt_time:.2f}s")
        print(f"  Function evaluations:  {n_evals}")
        print(f"  Optimized energy:      {best_energy:.6f}")
        print(f"  COBYLA success:        {opt_result.success}")

    # --- Stage 4: Solution decoding ---
    qc_final = build_qaoa_circuit(best_params, edges, n_qubits)
    psi_final = np.asarray(
        qc_final.execute(
            backend=backend, device=device, dtype=dtype,
        ).state,
        dtype=complex,
    )
    probs = np.abs(psi_final.flatten()) ** 2
    best_idx = int(np.argmax(probs))
    best_bits = format(best_idx, f"0{n_qubits}b")
    maxcut_val = compute_max_cut_value(best_bits, edges)
    best_prob = float(probs[best_idx])

    total_time = time.perf_counter() - total_start

    if verbose:
        print(f"  Best bitstring:   |{best_bits}⟩ (prob={best_prob:.4f})")
        print(f"  Max-Cut value:    {maxcut_val}/{len(edges)} edges cut")
        print(f"  Approximation ratio: {maxcut_val / len(edges):.2%} of total")
        print(f"  Total time:       {total_time:.2f}s")
        print(f"  Status:           ok")

    return {
        "status": "ok",
        "Optimal bitstring": best_bits,
        "Max-Cut Value": maxcut_val,
        "Optimized Energy": best_energy,
        "Exact Ground Energy": exact_energy,
        "Best Probability": best_prob,
        "Quantum Computation Time (s)": round(opt_time, 2),
        "Total Time (s)": round(total_time, 2),
        "energy_history": energy_history,
        "n_evals": len(energy_history),
        "circuit": qc_final,
        "h_cost": h_cost,
        "layers": layers,
        "n_qubits": n_qubits,
        "n_edges": len(edges),
        "circuit_path": "",
        "plot": [],
    }


# ===================================================================
# 5. Visualization
# ===================================================================


def plot_qaoa_results(
    edges: List[Tuple[int, int]],
    best_bits: str,
    energy_history: List[float],
    n_qubits: int,
    output_dir: str = ".",
) -> Dict[str, str]:
    """Generate QAOA result plots: convergence + Max-Cut graph.

    Args:
        edges: Edge list.
        best_bits: Optimal bitstring.
        energy_history: Per-evaluation energies.
        n_qubits: Number of qubits.
        output_dir: Directory for saved plots.

    Returns:
        Dict mapping plot type to file path.
    """
    if not HAS_MATPLOTLIB:
        return {}

    import os

    paths: Dict[str, str] = {}

    # Convergence plot
    conv_path = os.path.abspath(os.path.join(output_dir, "QAOA_Convergence.svg"))
    plt.figure(figsize=(6, 4))
    plt.plot(energy_history, color="#3498db", lw=2)
    plt.title("QAOA Energy Convergence")
    plt.xlabel("Function Evaluation")
    plt.ylabel("Energy ⟨H_C⟩")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(conv_path, dpi=150)
    plt.close()
    paths["convergence"] = conv_path

    # Max-Cut solution graph
    if HAS_NETWORKX:
        sol_path = os.path.abspath(
            os.path.join(output_dir, "MaxCut_Solution.svg"),
        )
        g = nx.Graph()
        g.add_edges_from(edges)
        colors = [
            "#3498db" if best_bits[i] == "0" else "#e74c3c"
            for i in range(n_qubits)
        ]
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(g, seed=42)
        nx.draw(
            g, pos, node_color=colors, with_labels=True,
            node_size=800, font_color="white", edge_color="#888888",
        )
        plt.title(f"Max-Cut Solution ({compute_max_cut_value(best_bits, edges)} edges cut)")
        plt.tight_layout()
        plt.savefig(sol_path, dpi=150)
        plt.close()
        paths["maxcut_solution"] = sol_path

    return paths


# ===================================================================
# 6. Class-based interface
# ===================================================================


class QAOASolver:
    """Class-based solver for QAOA Max-Cut.

    Usage:
        solver = QAOASolver()
        result = solver.run(
            edges=[(0,1),(1,2),(2,0)], n_qubits=3, layers=2,
        )
        print(result['Optimal bitstring'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        edges: List[Tuple[int, int]],
        n_qubits: int,
        layers: int = 2,
        max_iter: int = 100,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run QAOA. See qaoa_solve() for docs."""
        result = qaoa_solve(
            edges=edges, n_qubits=n_qubits, layers=layers,
            max_iter=max_iter, backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        # Generate plots if directory is available
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            plot_paths = plot_qaoa_results(
                edges, result["Optimal bitstring"],
                result["energy_history"], n_qubits, self.algo_dir,
            )
            result["plot"] = [
                {"format": "svg", "filename": p}
                for p in plot_paths.values()
            ]
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "Optimal bitstring": result.get("Optimal bitstring"),
            "Max-Cut Value": result.get("Max-Cut Value"),
            "Optimized Energy": result.get("Optimized Energy"),
            "Exact Ground Energy": result.get("Exact Ground Energy"),
            "Quantum Computation Time (s)": result.get("Quantum Computation Time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ===================================================================
# 7. Graph generators
# ===================================================================


def make_complete_graph(n_vertices: int) -> List[Tuple[int, int]]:
    """Generate a complete graph K_n (all pairs of vertices).

    Args:
        n_vertices: Number of vertices.

    Returns:
        Edge list for K_n.
    """
    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            edges.append((i, j))
    return edges


def make_cycle_graph(n_vertices: int) -> List[Tuple[int, int]]:
    """Generate a cycle graph C_n.

    Args:
        n_vertices: Number of vertices (>= 3).

    Returns:
        Edge list for C_n.
    """
    edges = []
    for i in range(n_vertices):
        edges.append((i, (i + 1) % n_vertices))
    return edges


def make_random_graph(
    n_vertices: int,
    edge_probability: float = 0.5,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """Generate a random Erdős-Rényi graph G(n, p).

    Args:
        n_vertices: Number of vertices.
        edge_probability: Probability of each edge existing.
        seed: Random seed.

    Returns:
        Random edge list.
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < edge_probability:
                edges.append((i, j))
    return edges


# ===================================================================
# 8. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "triangle_K3_p2",
        "edges": [(0, 1), (1, 2), (2, 0)], "n": 3, "layers": 2,
        "max_iter": 80, "min_cut": 2,
    },
    {
        "name": "square_C4_p2",
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0)], "n": 4, "layers": 2,
        "max_iter": 80, "min_cut": 4,
    },
    {
        "name": "star_5_p5",
        "edges": [(0, 1), (0, 2), (0, 3), (0, 4)], "n": 5, "layers": 5,
        "max_iter": 200, "min_cut": 4,
    },
    {
        "name": "6node_demo_p2",
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)],
        "n": 6, "layers": 2, "max_iter": 100, "min_cut": 5,
    },
    {
        "name": "K4_p3",
        "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        "n": 4, "layers": 3, "max_iter": 120, "min_cut": 4,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case."""
    name = case["name"]
    edges = case["edges"]
    n = case["n"]

    result = qaoa_solve(
        edges=edges, n_qubits=n, layers=case["layers"],
        max_iter=case["max_iter"], verbose=False, seed=42,
    )

    cut_val = result["Max-Cut Value"]
    ok = cut_val >= case["min_cut"]
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {name}: cut={cut_val}/{len(edges)} (min={case['min_cut']}), "
          f"bits=|{result['Optimal bitstring']}⟩, "
          f"energy={result['Optimized Energy']:.4f}, "
          f"evals={result['n_evals']}")
    return ok


# ===================================================================
# 9. Main
# ===================================================================

def main() -> None:
    """Run the complete QAOA demonstration pipeline."""
    if not HAS_SCIPY:
        print("scipy is not installed. Install with: pip install scipy")
        return

    print("=" * 60)
    print("QAOA — Quantum Approximate Optimization Algorithm")
    print("  Max-Cut Solver")
    print("=" * 60)

    solver = QAOASolver()

    # --- Demo 1: Default 6-node graph ---
    print("\n--- Demo 1: 6-node benchmark graph (p=2) ---")
    edges1 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
    result1 = solver.run(edges=edges1, n_qubits=6, layers=2, max_iter=80)
    print(f"  Best partition: |{result1['Optimal bitstring']}⟩")
    print(f"  Cut value:      {result1['Max-Cut Value']}/{len(edges1)}")
    print(f"  Energy:         {result1['Optimized Energy']:.6f}")
    print(f"  Time:           {result1['Quantum Computation Time (s)']:.1f}s")

    # --- Demo 2: Complete graph K4 ---
    print("\n--- Demo 2: Complete graph K4 (p=3) ---")
    edges2 = make_complete_graph(4)
    result2 = solver.run(edges=edges2, n_qubits=4, layers=3, max_iter=150)
    print(f"  Best partition: |{result2['Optimal bitstring']}⟩")
    print(f"  Cut value:      {result2['Max-Cut Value']}/{len(edges2)}")
    print(f"  Energy:         {result2['Optimized Energy']:.6f}")

    # --- Demo 3: Cycle graph C6 ---
    print("\n--- Demo 3: Cycle C6 (p=3) ---")
    edges3 = make_cycle_graph(6)
    result3 = solver.run(edges=edges3, n_qubits=6, layers=3, max_iter=150)
    print(f"  Best partition: |{result3['Optimal bitstring']}⟩")
    print(f"  Cut value:      {result3['Max-Cut Value']}/{len(edges3)}")

    # --- Demo 4: Effect of layers ---
    print("\n--- Demo 4: Effect of QAOA depth p (same graph) ---")
    edges4 = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
    for p_val in [1, 2, 3, 4, 5]:
        result4 = qaoa_solve(
            edges=edges4, n_qubits=6, layers=p_val,
            max_iter=80, verbose=False, seed=42,
        )
        print(f"  p={p_val}: cut={result4['Max-Cut Value']}/{len(edges4)}, "
              f"energy={result4['Optimized Energy']:.4f}, "
              f"time={result4['Quantum Computation Time (s)']:.1f}s, "
              f"bits=|{result4['Optimal bitstring']}⟩")

    # --- Demo 5: Energy convergence ---
    print("\n--- Demo 5: Energy convergence (p=2, K4) ---")
    result5 = qaoa_solve(
        edges=make_complete_graph(4), n_qubits=4, layers=2,
        max_iter=100, verbose=False, seed=42,
    )
    hist = result5["energy_history"]
    if len(hist) > 1:
        print(f"  Initial energy: {hist[0]:.4f}")
        print(f"  Final energy:   {hist[-1]:.4f}")
        print(f"  Improvement:    {hist[0] - hist[-1]:.4f}")
        print(f"  Evaluations:    {len(hist)}")

    # --- Demo 6: Random small graph ---
    print("\n--- Demo 6: Random graph G(5, 0.5) ---")
    edges6 = make_random_graph(5, 0.5, seed=123)
    result6 = solver.run(edges=edges6, n_qubits=5, layers=3, max_iter=100)
    print(f"  Edges:          {edges6}")
    print(f"  Best partition: |{result6['Optimal bitstring']}⟩")
    print(f"  Cut value:      {result6['Max-Cut Value']}/{len(edges6)}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
