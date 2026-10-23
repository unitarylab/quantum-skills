# Molecular Chemistry

Follow [SKILL.md](../SKILL.md) for task routing and execution permission; see the [SDK version and APIs](simulator-unitarylab.md#sources-and-version).

Chemistry is an optional UnitaryLab extension for electronic structure, DMRG, circuit variational
decomposition, and measurement/QASM workflows.

## Install and Import

```bash
python -m pip install "unitarylab[chemistry]==1.5.1"
```

```python
from unitarylab.library.chemistry import (
    Molecule,
    MolecularDMRGResult,
    MolecularDMRGWorkflow,
)
```

`unitarylab.library.chemistry` loads `unitarylab_chemistry` lazily. Without the optional package,
accessing an exported symbol raises an `ImportError` containing the install command above. Do not
require chemistry dependencies for ordinary simulator tasks.

## Minimal Workflow

```python
from unitarylab.library.chemistry import Molecule, MolecularDMRGWorkflow

molecule = Molecule(
    geometry=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))],
    basis="sto-3g",
    active_occupied=1,
    active_virtual=1,
)

workflow = MolecularDMRGWorkflow(
    molecule,
    backend="numpy",
    device="cpu",
    shots=10_000,
    seed=2077,
    workdir="chemistry-output",
)
result = workflow.run(target="electronic_structure")
print(result.electronic_structure.hf_energy)
```

`Molecule` accepts geometry, basis, charge, multiplicity, and non-negative active occupied/virtual
orbital counts. `Molecule.from_xyz(...)` reads a standard XYZ file. The requested active space must
contain at least one orbital; electron count and multiplicity must be compatible.

## Stages and Results

`MolecularDMRGWorkflow.run(target=...)` runs through the selected terminal stage:

1. `electronic_structure`
2. `dmrg`
3. `cvd_dmrg`
4. `measurement_QASM` (default)

The returned `MolecularDMRGResult` always contains `.molecule`; `.electronic_structure`, `.dmrg`,
`.cvd`, and `.measurement` become non-`None` as their stages complete. The full workflow writes
cache and QASM artifacts below `workdir`; choose that directory explicitly when side effects matter.

Chemistry may require PySCF, OpenFermion, OpenFermion-PySCF, and tensor-network dependencies.
The electronic-structure and DMRG stages are classical scientific computations; CVD constructs an approximate circuit, and measurement uses a separately selected execution path. None of these stages by itself establishes a quantum computational advantage. For authorized environment validation, H2 and an early terminal stage keep the example small.
