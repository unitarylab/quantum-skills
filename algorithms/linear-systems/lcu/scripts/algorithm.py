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

from unitarylab_algorithms import LCUAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Build two unitary operators U0, U1
n_sys = 1
U0 = Circuit(n_sys, name='H')
U0.h(0)         # Hadamard

U1 = Circuit(n_sys, name='X')
U1.x(0)         # Pauli-X

# Apply M = 0.6*H + 0.4*X to system
algo = LCUAlgorithm(text_mode='plain')
result = algo.run(
    alphas=[0.6, 0.4],
    unitaries=[U0, U1],
    n_sys=n_sys,
    initial_state=None,    # Starts in |0>
    backend='torch'
)

print(result['Success probability'])   # Post-selection probability
print(result['status'])                # 'ok' on success
print(result['circuit_path'])          # SVG circuit diagram
print(result['Result state'])          # Post-selected system state
