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

from unitarylab_algorithms import QCBMAlgorithm

algo = QCBMAlgorithm(text_mode="plain")
result = algo.run(
    n=4,
    layers=4,
    epochs=40,
    lr=0.1,
    backend='torch'
)

print(f"Final KL Loss: {result['Final KL Loss']:.4f}")
print(f"Circuit path: {result['circuit_path']}")
for f in result['plot']:
    print(f"Output file: {f['filename']} ({f['format']})")
