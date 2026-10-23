"""Installed-package API example generated from the leaf SKILL.md.

This file mirrors the first Python code block under
``Reference Implementation Example``. It intentionally imports the installed
``unitarylab-algorithms`` package and never adds a workspace path to sys.path.
"""

from __future__ import annotations

from unitarylab_algorithms import HiddenShiftAlgorithm


algorithm = HiddenShiftAlgorithm(text_mode="plain")
result = algorithm.run(
    qubits=6,
    shift="101100",
    shots=100,
    seed=41,
    backend="numpy",
    measurement_mode="sample",
)

print(result["status"])
print(result["Secret shift"])
print(result["Most common bitstring"])
print(result["Counts"])
print(result["Matched"])
print(result["circuit_path"])
print(result.get("plot", []))
