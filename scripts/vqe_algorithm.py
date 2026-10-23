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

from unitarylab_algorithms.quantum_machine_learning.vqe.algorithm import VQEAlgorithm

algo = VQEAlgorithm(text_mode="plain")
result = algo.run(
    n=2,
    layers=3,
    max_iter=150,
    seed=7,
    backend='torch'
)

print(f"Ground energy (VQE): {result['VQE Energy']:.4f}")
print(f"Exact ground energy: {result['Exact Energy']:.4f}")
print(f"Absolute error: {result['Absolute Error']:.2e}")
print(result['plot'])  # list of saved file dicts
