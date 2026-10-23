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

from unitarylab_algorithms import QSPAlgorithm

algo = QSPAlgorithm(text_mode="legacy")
result = algo.run(
    t=1.0,         # Evolution time t: targets cos(t * x)
    d=10,          # Polynomial degree d
    x=0.5,         # Test signal point x ∈ [-1, 1]
    backend='torch'
)

print(result['Absolute error'])    # Absolute error |QSP(x) - cos(t*x)|
print(result['Estimated value'])   # QSP estimated value at x
print(result['Ideal value'])       # cos(t * x) at test point
print(result['circuit_path'])      # SVG circuit diagram
print(result['plot'])              # List of saved output file dicts
