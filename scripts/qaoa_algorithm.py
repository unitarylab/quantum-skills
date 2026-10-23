"""Verification script generated from the leaf SKILL.md.

This file is generated from the first Python code block under
`Reference Implementation Example`. Regenerate it after editing the skill.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _add_workspace_paths() -> None:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "unitarylab_algorithms").is_dir():
            sys.path.insert(0, str(candidate))
            return
        if (candidate / "quantum-skills-new").is_dir() and (candidate / "unitarylab_algorithms").is_dir():
            sys.path.insert(0, str(candidate))
            return
    workspace = here.parents[4] if len(here.parents) > 4 else here.parent
    sys.path.insert(0, str(workspace))


_add_workspace_paths()

from unitarylab_algorithms.quantum_machine_learning.qaoa.algorithm import QAOAAlgorithm

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
n = 6

algo = QAOAAlgorithm(text_mode="plain")
result = algo.run(
    edges=edges,
    n=n,
    layers=4,
    max_iter=100,
    backend='torch'
)

print(f"Best cut value: {result['Max-Cut Value']}")
print(f"Best partition: {result['Optimal bitstring']}")
print(result['plot'])  # list of {"format": "svg", "filename": "..."} dicts
