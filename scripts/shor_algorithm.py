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

from unitarylab_algorithms import ShorAlgorithm

algo = ShorAlgorithm()
result = algo.run(
    N=15,              # Number to factor
    method='matrix',   # 'matrix' or 'operator'
    backend='torch',
    max_retries=15
)

print(result['status'])           # 'ok' on success, 'failed' if exhausted
print(result['factors'])          # List of factors, e.g. [3, 5]; None on failure
print(result['period'])           # Found period r, or None for classical path
print(result['Selected base'])    # Random base a used
print(result['circuit_path'])     # Path to SVG circuit diagram (None for classical path)
print(result['plot'])             # List of saved output files: [{'format': 'svg'/'txt', 'filename': '...'}]
