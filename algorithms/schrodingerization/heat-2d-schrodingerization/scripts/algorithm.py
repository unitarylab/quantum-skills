#!/usr/bin/env python3
"""Run the 2D heat Schrodingerization skill entry point.

This script is generated from the leaf SKILL.md. It uses the implemented
Heat2dEquationAlgorithm class rather than re-implementing solver internals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_workspace_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "unitarylab_algorithms").is_dir() and (candidate / "quantum-skills").is_dir():
            return candidate
    raise RuntimeError("Could not find workspace root containing unitarylab_algorithms and quantum-skills.")


ROOT = find_workspace_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unitarylab_algorithms.schrodingerization.equation_heat2d.algorithm import (  # noqa: E402
    Heat2dEquationAlgorithm,
)


def main() -> None:
    setup_path = ROOT / "unitarylab_algorithms" / "schrodingerization" / "equation_heat2d" / "setup.json"
    params = json.loads(setup_path.read_text(encoding="utf-8"))
    algo_dir = ROOT / "results" / "schrodingerization" / "equation_heat2d"
    algo_dir.mkdir(parents=True, exist_ok=True)

    result = Heat2dEquationAlgorithm().run(
        params=params,
        algo_dir=str(algo_dir),
        backend="torch",
        device="cpu",
    )

    print("2D Heat Schrodingerization")
    print(f"status: {result['status']}")
    print(f"grid: {result['grid']}")
    print(f"plot: {result['plot']['filename']}")
    print(f"circuit files: {len(result.get('circuit', []))}")
    u = result.get("u", [])
    rows = len(u)
    cols = len(u[0]) if rows and isinstance(u[0], list) else 0
    print(f"solution shape: {rows} x {cols}")


if __name__ == "__main__":
    main()
