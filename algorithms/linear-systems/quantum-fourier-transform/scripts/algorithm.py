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

import numpy as np
from unitarylab_algorithms.linear_algebra.qft.algorithm import QFTAlgorithm

algo = QFTAlgorithm(text_mode="plain")

state = np.zeros(8, dtype=complex)
state[1] = 1.0

result = algo.run(
    n=3,
    state=state,
    inverse=False,
    backend="torch",
    device="cpu",
)

print(result["status"])
print(result["Verification error"])
print(result["Final state"])
print(result["Expected state"])
print(result["circuit_path"])
