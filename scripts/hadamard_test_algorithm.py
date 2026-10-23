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

from unitarylab_algorithms import HadamardTestAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Example: estimate Re(<+|RZ(0.8)|+>) = cos(0.4)
U = Circuit(1, name="RZ_0.8")
U.rz(0.8, 0)

prepare_psi = Circuit(1, name="|+>")
prepare_psi.h(0)

algo = HadamardTestAlgorithm()

# Estimate real part
result_re = algo.run(mode='expectation', U=U, prepare_psi=prepare_psi,
                     imag=False, shots=20000, backend='torch')
print(result_re['Estimated Value'])   # ≈ cos(0.4) ≈ 0.9211

# Estimate imaginary part
result_im = algo.run(mode='expectation', U=U, prepare_psi=prepare_psi,
                     imag=True, shots=20000, backend='torch')
print(result_im['Estimated Value'])   # ≈ -sin(0.4) ≈ -0.3894
