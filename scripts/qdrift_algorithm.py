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
from unitarylab_algorithms import QDriftAlgorithm

# 2x2 Hermitian Hamiltonian matrix
H = np.array([[2, 1],
              [1, 3]], dtype=float)

algo = QDriftAlgorithm()
result = algo.run(
    H=H,
    t=1.0,
    error=1e-8,
    steps=5000,
    backend='torch',
)

print("status:", result['status'])
print("Frobenius norm of error:", result['Frobenius norm of error'])
for f in result['plot']:
    print(f"Saved {f['format']} file: {f['filename']}")
