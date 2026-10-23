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
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import SPSAEstimatorGradient

theta = ParameterVector("theta", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

estimator = StatevectorEstimator()
obs = SparsePauliOp("ZZ")

grad = SPSAEstimatorGradient(estimator=estimator, epsilon=0.01, batch_size=4, seed=123)
result = grad.run(
    circuits=[qc],
    observables=[obs],
    parameter_values=[[0.3, -0.2]],
    parameters=[[theta[0], theta[1]]],
    precision=None,
).result()

print(result.gradients)
