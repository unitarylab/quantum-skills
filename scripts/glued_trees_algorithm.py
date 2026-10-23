"""Installed-package API example generated from the leaf SKILL.md.

This file mirrors the first Python code block under
``Reference Implementation Example``. It intentionally imports the installed
``unitarylab-algorithms`` package and never adds a workspace path to sys.path.
"""

from __future__ import annotations

from unitarylab_algorithms import GluedTreesAlgorithm


algorithm = GluedTreesAlgorithm(text_mode="plain")
result = algorithm.run(
    qubits=4,
    seed=0,
    shots=256,
    repetitions=4,
    order=1,
    max_terms=None,
    backend="numpy",
    times="0,2,4",
)

print(result["status"])
print(result["Nodes"], result["Exit state"])
print(result["Pauli terms"])
print(result["Peak exit probability"], result["Peak time"])
print(result["Rows"])
print(result["circuit_path"])
print(result.get("plot", []))
