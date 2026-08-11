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
from unitarylab_algorithms.linear_algebra.qsvt_qlsa.algorithm import QSVTLinearSolverAlgorithm

A = np.array([[0.8, 0.0], [0.0, 0.4]])
b = np.array([1.0, 2.0])

algo = QSVTLinearSolverAlgorithm(text_mode="plain")
result = algo.run(A=A, b=b, epsilon=0.0001, backend='torch', device='cpu', dtype=np.complex128)
print(result)
