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

from unitarylab_algorithms import DiscreteLogAlgorithm

algo = DiscreteLogAlgorithm()
result = algo.run(
    g=3,   # Base
    y=6,   # Target: 3^x ≡ 6 (mod 7)
    P=7,   # Modulus (prime)
    backend='torch'
)

print(result['Found x'])                  # Found discrete log x
print(result['status'])                   # 'ok' on success, 'failed' otherwise
print(result['circuit_path'])             # SVG circuit diagram path
print(result.get('plot', []))             # List of output file dicts [{"format": ..., "filename": ...}]
print(result.get('Detected period r'))    # Detected group order r
print(result.get('Computation time (s)')) # Simulation time in seconds
