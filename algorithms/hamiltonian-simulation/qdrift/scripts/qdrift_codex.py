#!/usr/bin/env python3
"""QDrift Hamiltonian simulation example generated from SKILL.md.

This script follows the QDrift skill guide:
1. Build a Hermitian Hamiltonian.
2. Run QDrift with reproducible random seeds.
3. Compare approximate and exact evolution matrices.
4. Report Frobenius-norm errors across step counts.

The implementation intentionally uses the public QDriftAlgorithm interface
described in the skill document.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def add_workspace_root() -> Path:
    """Add the workspace root containing unitarylab_algorithms to sys.path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "unitarylab_algorithms").is_dir():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find workspace root containing unitarylab_algorithms")


WORKSPACE_ROOT = add_workspace_root()
os.environ.setdefault("MPLBACKEND", "Agg")

from unitarylab_algorithms import QDriftAlgorithm  # noqa: E402


def heisenberg_like_hamiltonian() -> np.ndarray:
    """Return the 2-qubit Hamiltonian used in the SKILL.md hands-on example."""
    xx = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=float,
    )
    zz = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return xx + zz


def run_single_case(
    H: np.ndarray,
    t: float,
    steps: int,
    seed: int,
    error: float,
    backend: str,
) -> dict[str, Any]:
    """Run one reproducible QDrift trajectory."""
    np.random.seed(seed)
    algo = QDriftAlgorithm(text_mode="plain")
    result = algo.run(H=H, t=t, error=error, steps=steps, backend=backend)

    approx = np.asarray(result["Approximate evolution matrix"])
    exact = np.asarray(result["Exact evolution matrix"])
    fro_error = float(result["Frobenius norm of error"])
    recomputed_error = float(np.linalg.norm(approx - exact, ord="fro"))

    return {
        "status": result["status"],
        "steps": steps,
        "seed": seed,
        "frobenius_error": fro_error,
        "recomputed_frobenius_error": recomputed_error,
        "matrix_shape": list(approx.shape),
        "circuit_path": result.get("circuit_path"),
        "plot": result.get("plot", []),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate error statistics by step count."""
    summaries: list[dict[str, Any]] = []
    for steps in sorted({int(row["steps"]) for row in rows}):
        errors = np.array(
            [float(row["frobenius_error"]) for row in rows if row["steps"] == steps],
            dtype=float,
        )
        summaries.append(
            {
                "steps": steps,
                "runs": int(errors.size),
                "mean_error": float(errors.mean()),
                "std_error": float(errors.std(ddof=0)),
                "min_error": float(errors.min()),
                "max_error": float(errors.max()),
            }
        )
    return summaries


def print_table(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    """Print trajectory-level and aggregate results."""
    print("QDrift randomized Hamiltonian simulation")
    print(f"workspace: {WORKSPACE_ROOT}")
    print()
    print("Individual trajectories:")
    print("  steps   seed   status   Frobenius error")
    print("  -----   ----   ------   ----------------")
    for row in rows:
        print(
            f"  {row['steps']:>5}   {row['seed']:>4}   "
            f"{row['status']:<6}   {row['frobenius_error']:.6e}"
        )

    print()
    print("Aggregate statistics:")
    print("  steps   runs   mean error     std error      min error      max error")
    print("  -----   ----   ----------     ---------      ---------      ---------")
    for item in summaries:
        print(
            f"  {item['steps']:>5}   {item['runs']:>4}   "
            f"{item['mean_error']:.6e}   {item['std_error']:.6e}   "
            f"{item['min_error']:.6e}   {item['max_error']:.6e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the QDrift examples described in qdrift/SKILL.md."
    )
    parser.add_argument("--t", type=float, default=1.0, help="Total evolution time.")
    parser.add_argument(
        "--error",
        type=float,
        default=1e-8,
        help="Target error parameter passed through to QDriftAlgorithm.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1000, 5000],
        help="Step counts to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for repeated trajectories.",
    )
    parser.add_argument(
        "--backend",
        default="torch",
        help="Backend passed to QDriftAlgorithm.run().",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("qdrift_codex_results.json"),
        help="JSON file for trajectory and summary results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    H = heisenberg_like_hamiltonian()

    rows: list[dict[str, Any]] = []
    for steps in args.steps:
        for seed in args.seeds:
            rows.append(
                run_single_case(
                    H=H,
                    t=args.t,
                    steps=steps,
                    seed=seed,
                    error=args.error,
                    backend=args.backend,
                )
            )

    summaries = summarize(rows)
    print_table(rows, summaries)

    payload = {
        "hamiltonian": H.tolist(),
        "t": args.t,
        "error": args.error,
        "backend": args.backend,
        "trajectories": rows,
        "summary": summaries,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Saved JSON results: {args.output}")


if __name__ == "__main__":
    main()
