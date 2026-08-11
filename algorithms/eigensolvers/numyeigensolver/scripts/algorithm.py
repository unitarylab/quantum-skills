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

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver

operator = SparsePauliOp(["ZZ", "XI", "IX"], coeffs=[1.0, 0.5, 0.3])

solver = NumPyEigensolver(k=2)
result = solver.compute_eigenvalues(operator)

print(result.eigenvalues)
print(result.eigenstates)
