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
from qiskit_algorithms.gradients import QFI, LinCombQGT

# Build a parameterized circuit
theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

parameter_values = [0.5, 1.0]

# Construct QFI using LinCombQGT as the QGT backend
estimator = StatevectorEstimator()
qgt = LinCombQGT(estimator)
qfi = QFI(qgt)

# Run and retrieve results
job = qfi.run([qc], [parameter_values])
result = job.result()

print(result.qfis)      # [array([[...], [...]])]  shape (2, 2) per circuit
print(result.precision) # precision used by the estimator
