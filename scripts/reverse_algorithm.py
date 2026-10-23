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

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit_algorithms.gradients.utils import DerivativeType

theta = ParameterVector("theta", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

values = [0.5, 1.0]

# Gradient
observable = SparsePauliOp("ZZ")
grad = ReverseEstimatorGradient(derivative_type=DerivativeType.REAL)
grad_result = grad.run([qc], [observable], [values]).result()
print(grad_result.gradients)

# QGT
qgt = ReverseQGT(phase_fix=True, derivative_type=DerivativeType.COMPLEX)
qgt_result = qgt.run([qc], [values]).result()
print(qgt_result.qgts)
