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

from unitarylab_algorithms import SimonAlgorithm

algo = SimonAlgorithm()
result = algo.run(
    s='1010',      # Hidden string to find
    backend='torch'
)

print(result['Computed s'])      # Found hidden string (should match s)
print(result['status'])          # 'ok' if found == s
print(result['circuit_path'])    # SVG circuit diagram path
print(result['plot'])            # List of saved file dicts [{"format": ..., "filename": ...}]
