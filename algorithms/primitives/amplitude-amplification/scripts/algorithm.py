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

from unitarylab_algorithms import AmplitudeAmplificationAlgorithm
from unitarylab.core import Circuit

# Prepare state U such that some target qubits land in |0>
# Example: 2-qubit state preparation
U = Circuit(2, name="PrepU")
U.ry(0.6, 0)   # Partially rotates qubit 0
U.cx(0, 1)     # Entangles

algo = AmplitudeAmplificationAlgorithm()
result = algo.run(
    U=U,
    good_zero_qubits=[0, 1],   # Good state: both qubits = |0>
    p=0.1,                      # Initial success probability estimate
    backend='torch'
)

print(result['Amplified Target Probability'])  # Amplified probability of good state
print(result['circuit_path'])                  # SVG circuit diagram path
print(result['status'])                        # 'ok' if amplification succeeded
print(result['plot'])                          # List of saved file dicts [{"format": ..., "filename": ...}]
