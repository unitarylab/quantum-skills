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

from unitarylab_algorithms import AmplitudeEstimationAlgorithm
from unitarylab.core import Circuit

# Build state preparation U (data register only, no ancilla)
U = Circuit(2, name="PrepU")
U.ry(1.1, 0)
U.cx(0, 1)

algo = AmplitudeEstimationAlgorithm()
result = algo.run(
    U=U,
    good_zero_qubits=[0],   # Good state: qubit 0 in |0>
    d=6,                    # Phase register bits (higher d = better precision)
    backend='torch'
)

print(result['Target amplitude'])     # Estimated probability p
print(result['Phase'])                # Estimated phase phi
print(result['Most likely phase (bits)'])  # Best phase register bit-string
print(result['circuit_path'])         # SVG circuit diagram path
print(result['plot'])                 # List of saved output file dicts
