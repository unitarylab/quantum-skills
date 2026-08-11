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
from sklearn.datasets import make_moons
from unitarylab_algorithms.quantum_machine_learning.cvqnn.algorithm import CVQNNAlgorithm

X, y = make_moons(n_samples=40, noise=0.1, random_state=42)

algo = CVQNNAlgorithm(text_mode="legacy")
result = algo.run(
    x_train=X,
    y_train=y,
    n_layers=2,
    cutoff=6,
    epochs=40,
    lr=0.05
)

print(f"Status: {result['status']}")
print(f"Final Accuracy: {result['Final Accuracy']:.2%}")
print(f"Final Loss:     {result['Final Loss']:.6f}")
print(f"Circuit saved:  {result['circuit_path']}")
for f in result['plot']:
    print(f"Plot saved:     {f['filename']}  (format: {f['format']})")
