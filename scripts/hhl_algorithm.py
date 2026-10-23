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
        if (candidate / "quantum-`skills-new").is_dir() and (candidate / "unitarylab_algorithms").is_dir():
            sys.path.insert(0, str(candidate))
            return
    workspace = here.parents[4] if len(here.parents) > 4 else here.parent
    sys.path.insert(0, str(workspace))


_add_workspace_paths()

from unitarylab import Circuit
from unitarylab_algorithms import HHLAlgorithm
import numpy as np

# 2x2 Hermitian system
A = np.array([[1.5, 0.5], [0.5, 1.5]])
b = np.array([1.0, 0.0])

algo = HHLAlgorithm(text_mode="plain")
result = algo.run(
    A=A,
    b=b,
    d=4,           # Phase register bits
    backend='torch'
)

print(result['Estimated solution (quantum)'])    # Quantum solution vector x_q
print(result['Exact solution (classical)'])      # Classical exact solution x_c
print(result['L2 error'])                        # ||x_q - x_c||
print(result['Post-selection probability'])      # Post-selection success probability
print(result['circuit_path'])                    # SVG circuit diagram
