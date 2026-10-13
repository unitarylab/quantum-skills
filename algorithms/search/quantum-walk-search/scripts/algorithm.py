"""Small example using only the implementation bundled beside this file."""
import argparse
import json
import numpy as np

from quantum_walk_search_implementation import QuantumWalkSearchAlgorithm, main as diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the eight-vertex example.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if not args.execute:
        diagnostic([])
        return
    algorithm = QuantumWalkSearchAlgorithm(text_mode="plain", algo_dir=args.output_dir)
    result = algorithm.run(
        num_vertices=8, marked="2,5", graph_type="complete",
        method="trotter", target_error=1e-4, simulation_steps=8,
        shots=128, seed=7, backend="numpy", device="cpu", dtype=np.complex128,
    )
    print(json.dumps({
        "status": result["status"],
        "success_probability": result["success_probability"],
        "exact_probability_at_time": result["exact_probability_at_time"],
        "simulation_probability_error": result["simulation_probability_error"],
        "counts": result["counts"],
        "observed_count_total": sum(result["counts"].values()),
        "expected_count_total": 128,
        "circuit_path": result["circuit_path"],
        "plot": result["plot"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
