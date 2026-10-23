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

import numpy as np
from unitarylab_algorithms import CartanDecompositionAlgorithm

# Define a 2x2 real symmetric Hamiltonian
H = np.array([[2.0, 1.0],
              [1.0, 2.0]])

algo = CartanDecompositionAlgorithm()

result = algo.run(
    H=H,
    t=1.0,
    error=1e-3,
    lr=1e-3,
    max_steps=100000,
    reps=5000,
)

print("status      :", result['status'])
print("circuit_path:", result['circuit_path'])
print("plot        :", result['plot'])          # e.g. [{'format': 'txt', 'filename': '/path/to/result.txt'}]
