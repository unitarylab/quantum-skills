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
from unitarylab_algorithms.hamiltonian_simulation.trotter.algorithm import TrotterAlgorithm

# 2-qubit Heisenberg Hamiltonian as a numpy matrix
H = np.array([[2.0, 1.0],
              [1.0, 3.0]])
t = 1.0
error = 1e-8

algo = TrotterAlgorithm()
result = algo.run(H=H, t=t, error=error, order=2, steps=1000, backend='torch')

print("status:", result['status'])          # 'ok' on success
print("circuit_path:", result['circuit_path'])
print("output files:", result['plot'])      # list of {format, filename} dicts
print("circuit:", result['circuit'])
