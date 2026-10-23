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
from unitarylab_algorithms import QSPHSAlgorithm

# 2x2 Hermitian Hamiltonian
H = np.array([[2, 1],
              [1, 3]], dtype=complex)

algo = QSPHSAlgorithm(text_mode="plain")
result = algo.run(
    H=H,
    t=1.0,
    error=1e-8,
    degree=15,
    beta=0.7,
)

print("status      :", result["status"])
print("circuit_path:", result["circuit_path"])
print("plot        :", result["plot"])
print("frob_error  :", result["Frobenius norm of error"])
