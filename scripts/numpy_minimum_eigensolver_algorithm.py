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
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

# H = ZI + IZ + 0.5 * XX
operator = SparsePauliOp.from_list([
    ("ZI", 1.0),
    ("IZ", 1.0),
    ("XX", 0.5),
])

aux_ops = {
    "magnetization": SparsePauliOp.from_list([("ZZ", 1.0)]),
}

solver = NumPyMinimumEigensolver()
result = solver.compute_minimum_eigenvalue(operator, aux_operators=aux_ops)

print("minimum eigenvalue:", result.eigenvalue)
print("eigenstate available:", result.eigenstate is not None)
print("aux values:", result.aux_operators_evaluated)
