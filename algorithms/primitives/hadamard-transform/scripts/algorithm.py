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

from unitarylab_algorithms import HadamardTransformAlgorithm

algo = HadamardTransformAlgorithm()

# Mode 1: Generate uniform superposition
result = algo.run(n=3, mode='superposition', backend='torch')
print(result['status'])                    # 'ok' or 'failed'
print(result['State vector'])              # Final state vector
print(result['Probability distribution'])  # Bitstring → probability over 2^3 basis states
print(result['circuit_path'])              # Path to SVG circuit diagram

# Mode 2: Verify H^2 = I
result2 = algo.run(n=3, mode='reflexive_test', backend='torch')
print(result2['status'])         # 'ok' if H^2 recovers original state
