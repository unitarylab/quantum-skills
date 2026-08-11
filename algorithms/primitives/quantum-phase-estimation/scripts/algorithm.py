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

from unitarylab_algorithms import QPEAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Build a 1-qubit unitary with known phase phi = 1/4
# U = diag(1, e^{2pi*i*phi}) so with phi=0.25: U = diag(1, i) = S gate
U = Circuit(1, name="S_gate")
U.s(0)   # S gate has phase e^{i*pi/2} = e^{2pi*i*0.25}

# Prepare eigenstate |1> (eigenstate of S is |1> with eigenvalue i=e^{i*pi/2})
prepare_psi = Circuit(1, name="prep_1")
prepare_psi.x(0)   # |0> -> |1>

algo = QPEAlgorithm()          # algo_dir can be set here; defaults to results/
result = algo.run(
    U=U,
    d=4,                       # 4-bit phase precision (1/16 = 0.0625)
    prepare_target=prepare_psi,
    backend='torch'
)

print(result['Estimated phase'])         # Should be ~0.25
print(result['Best phase probability'])  # Probability of the best state
print(result['circuit_path'])            # SVG circuit diagram path
