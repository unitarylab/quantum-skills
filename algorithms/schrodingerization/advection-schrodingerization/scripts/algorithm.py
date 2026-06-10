#!/usr/bin/env python3
"""Run the 1D advection Schrodingerization skill entry point.

This script is generated from the leaf SKILL.md. It uses the implemented
AdvectionEquationAlgorithm class rather than re-implementing solver internals.
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

from unitarylab_algorithms.schrodingerization.equation_advection.algorithm import (  # noqa: E402
    AdvectionEquationAlgorithm,
)


def main() -> None:
    setup_path = ROOT / "unitarylab_algorithms" / "schrodingerization" / "equation_advection" / "setup.json"
    params = json.loads(setup_path.read_text(encoding="utf-8"))
    algo_dir = ROOT / "results" / "schrodingerization" / "equation_advection"
    algo_dir.mkdir(parents=True, exist_ok=True)

    result = AdvectionEquationAlgorithm().run(
        params=params,
        algo_dir=str(algo_dir),
        backend="torch",
        device="cpu",
    )

    print("Advection Schrodingerization")
    print(f"status: {result['status']}")
    print(f"grid: {result['grid']}")
    print(f"plot: {result['plot']['filename']}")
    print(f"circuit files: {len(result.get('circuit', []))}")
    print(f"solution points: {len(result.get('u', []))}")


if __name__ == "__main__":
    main()
