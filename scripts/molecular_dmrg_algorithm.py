"""Small public-API example for the Molecular DMRG workflow."""

from __future__ import annotations

import math
from pathlib import Path

from unitarylab_algorithms import MolecularDMRGAlgorithm


result = MolecularDMRGAlgorithm(text_mode="plain").run(
    xyz_file="H2",
    basis="sto-3g",
    charge=0,
    multiplicity=1,
    active_occupied_orbitals=1,
    active_virtual_orbitals=1,
    max_iterations=10,
    cvd_restarts=1,
    cvd_init_scale=0.5,
    chi0=10,
    nsweeps=10,
    maxdim=20,
    cutoff=1e-8,
    shots=0,
    seed=2077,
    backend="torch",
    device="cpu",
    dtype="complex128",
)

required_energies = (
    "HF energy",
    "DMRG energy",
    "UnitaryLab exact energy",
    "Exact energy difference",
)

assert result["status"] == "ok"
assert result["circuit"] is not None
assert Path(result["circuit_path"]).is_file()
assert all(math.isfinite(float(result[field])) for field in required_energies)
assert math.isclose(
    float(result["Exact energy difference"]),
    abs(float(result["UnitaryLab exact energy"]) - float(result["DMRG energy"])),
    rel_tol=1e-12,
    abs_tol=1e-12,
)

for field in (
    "status",
    "Active space",
    "HF energy",
    "Active-space FCI energy",
    "DMRG energy",
    "UnitaryLab exact energy",
    "Exact energy difference",
    "Sampled energy",
    "Measurement groups",
    "circuit_path",
):
    print(f"{field}: {result[field]}")

print("Generated files:")
for artifact in result["plot"]:
    print(f"- {artifact['format']}: {artifact['filename']}")
