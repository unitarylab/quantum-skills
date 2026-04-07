---
name: hamiltonian-simulation-interface
description: "Use when: you need the unified hamiltonian_simulation interface, method dispatch behavior, parameter constraints, and result-object semantics across trotter, qdrift, taylor, and qsp."
---

# Hamiltonian Simulation Interface Skill Guide

## Overview

`hamiltonian_simulation(...)` is the unified entry point for matrix-based Hamiltonian simulation in this package. It validates the input type, dispatches by method name, and returns a method-specific simulator object under one common result abstraction.

### Key Insight

The function does not return a plain matrix. It returns a `HamiltonianSimulationResult`-compatible object with lazy properties (`circuit`, `evolution_result`, `total_error`), so users can switch methods while keeping one stable usage pattern.

### Real Applications:

1. Rapid A/B testing of multiple simulation algorithms on the same Hamiltonian.
2. Building benchmark scripts with a method-agnostic interface.
3. Integrating Hamiltonian simulation into larger research pipelines with minimal branching logic.
4. Teaching and demos where one API call exposes multiple algorithm families.

## Learning Objectives

After using this skill, you should be able to:
1. Call `hamiltonian_simulation` correctly for trotter, qdrift, taylor, and qsp.
2. Understand method-specific keyword constraints enforced by each `_simulation_*` wrapper.
3. Interpret `HamiltonianSimulationResult` lifecycle, including validation, padding, decomposition mode, and lazy error evaluation.
4. Extract reproducible outputs from the returned object for downstream analysis.
5. Connect interface-level behavior to the four algorithm-specific skill documents.

## Prerequisites

### Essential knowledge:

1. Basic linear algebra (Hermitian matrices, norms, matrix exponential).
2. Basic Python usage with NumPy arrays.
3. High-level understanding of Hamiltonian simulation goals.

### Mathematical comfort:

1. Spectral norm and approximation error interpretation.
2. Time-evolution operator $e^{-iHt}$ and truncated approximations.
3. Power-of-two Hilbert-space sizing and matrix padding intuition.

### Recommended:

1. Read [qitonghu/hamiltonian_simulation/algorithms/trotter/skill.md](qitonghu/hamiltonian_simulation/algorithms/trotter/skill.md).
2. Read [qitonghu/hamiltonian_simulation/algorithms/qdrift/skill.md](qitonghu/hamiltonian_simulation/algorithms/qdrift/skill.md).
3. Read [qitonghu/hamiltonian_simulation/algorithms/taylor/skill.md](qitonghu/hamiltonian_simulation/algorithms/taylor/skill.md).
4. Read [qitonghu/hamiltonian_simulation/algorithms/qsp/skill.md](qitonghu/hamiltonian_simulation/algorithms/qsp/skill.md).

## Using the Provided Implementation

### Quick Start Example

```python
import numpy as np
from core.example import build_symmetric_banded_matrix
from hamiltonian_simulation import hamiltonian_simulation

H = build_symmetric_banded_matrix(2, 2)
t = 1.0

trotter_res = hamiltonian_simulation(
    H,
    t,
    method='trotter',
    target_error=1e-4,
    order=2,
    steps=80,
)

qdrift_res = hamiltonian_simulation(
    H,
    t,
    method='qdrift',
    target_error=1e-4,
    steps=3000,
)

taylor_res = hamiltonian_simulation(
    H,
    t,
    method='taylor',
    target_error=1e-4,
    degree=12,
)

qsp_res = hamiltonian_simulation(
    H,
    t,
    method='qsp',
    target_error=1e-4,
    degree=15,
)

# Alias: qsvt -> qsp
qsvt_res = hamiltonian_simulation(H, t, method='qsvt', target_error=1e-4, degree=15)

for name, res in [
    ('trotter', trotter_res),
    ('qdrift', qdrift_res),
    ('taylor', taylor_res),
    ('qsp', qsp_res),
]:
    print(name, res.target_qubits, res.total_error)
```

### Core Parameters Explained

```python
def hamiltonian_simulation(
    H: np.ndarray,
    t: float,
    method: str = 'trotter',
    target_error: float = 1e-8,
    **kwargs
) -> HamiltonianSimulationResult:
    ...
```

Parameter meaning:

1. `H`: input Hamiltonian, must be a `numpy.ndarray`.
2. `t`: total evolution time in $e^{-iHt}$.
3. `method`: dispatch key; current methods are `trotter`, `qdrift`, `taylor`, `qsp` (plus alias `qsvt`) and `cartan-optimization`.
4. `target_error`: target approximation tolerance passed to concrete simulators.
5. `**kwargs`: method-specific parameters checked by each wrapper.

Method-specific kwargs accepted by current wrappers:

1. `trotter`: `order`, `steps`
2. `qdrift`: `steps`
3. `taylor`: `degree`
4. `qsp` or `qsvt`: `degree`
5. `cartan-optimization`: no extra kwargs exposed in the current wrapper

Return dictionary contains:

`hamiltonian_simulation(...)` returns an object, not a dictionary. For logging and serialization, you can standardize it as:

```python
result_obj = hamiltonian_simulation(H, t, method='trotter', target_error=1e-4, order=2, steps=100)

result = {
    'method': result_obj.method,
    'time': result_obj.t,
    'target_qubits': result_obj.target_qubits,
    'circuit': result_obj.circuit,
    'evolution_result': result_obj.evolution_result,
    'total_error': result_obj.total_error,
}
```

Detailed explanation of `HamiltonianSimulationResult` (from `algorithms/__init__.py`):

```python
@dataclass
class HamiltonianSimulationResult:
    method: str
    H: np.ndarray
    t: float
    target_error: float
    tol: float = 1e-12
    original_H: np.array = field(init=False)
    target_qubits: int = field(init=False)
    _circuit: Optional[GateSequence] = field(init=False, repr=False, default=None)
    _total_error: Optional[float] = field(init=False, repr=False, default=None)
    _evolution_result: Optional[np.ndarray] = field(init=False, repr=False, default=None)
```

How it works in detail:

1. Input validation in `__post_init__`:
- Checks square matrix.
- Checks Hermitian condition within tolerance.

2. Automatic padding to power-of-two dimension:
- If dimension is not power-of-two, pads with zeros to next power-of-two.
- Sets `target_qubits = log2(padded_dim)`.

3. Real/banded optimization metadata:
- Detects near-real Hamiltonians.
- Computes bandwidth and may use banded Pauli basis in `_pauli_decompose`.

4. Lazy execution model:
- `circuit` property builds circuit only when needed.
- `evolution_result` builds/returns cached approximate matrix.
- `total_error` computes exact-vs-approx spectral norm only when accessed.

5. Error computation path:
- Exact unitary: `expm(-1j * original_H * t)`.
- Approximate unitary: relevant top-left block from cached evolution matrix.
- Metric: spectral norm `||U_exact - U_approx||_2`.

## Understanding the Core Components

### 1) Dispatcher and Method Routing

The interface uses a mapping dictionary and dispatches by `method` string:

```python
mapping = {
    'trotter': _simulation_trotter,
    'qdrift': _simulation_qdrift,
    'taylor': _simulation_taylor,
    'qsp': _simulation_qsp,
    'qsvt': _simulation_qsp,
    'cartan-optimization': _simulation_cartan_optimization
}
```

Key behavior:

1. Alias `qsvt` routes to `_simulation_qsp`.
2. Unknown method raises `ValueError`.
3. Type check enforces `H` must be `np.ndarray`.

### 2) `_simulation_trotter`: Deterministic Product Formula Wrapper

Code excerpt (directly from `hamiltonian_simulation.py`):

```python
def _simulation_trotter(
    H: Union[np.ndarray, list],
    t: float,
    target_error: float,
    **kwargs: Any
) -> HamiltonianSimulationResult:
    from algorithms.trotter import Trotter
    allowed_keys = {'order', 'steps'}
    for key in kwargs:
        if key not in allowed_keys:
            raise ValueError(f"Unsupported keyword argument '{key}'. Only 'order' and 'steps' are allowed.")

    order = kwargs.get('order', 1)
    steps = kwargs.get('steps', 1000)

    runable = Trotter(H, t, target_error, order=order, steps=steps)
    return runable
```

Responsibilities:

1. Restrict kwargs to `order`, `steps`.
2. Instantiate `algorithms.trotter.Trotter(H, t, target_error, order, steps)`.
3. Return the simulator object as `HamiltonianSimulationResult` interface.

Tie-in with method skill:

- See [qitonghu/hamiltonian_simulation/algorithms/trotter/skill.md](qitonghu/hamiltonian_simulation/algorithms/trotter/skill.md) for Suzuki recursion, error scaling, and practical parameter tuning.

### 3) `_simulation_qdrift`: Randomized Sampling Wrapper

Code excerpt (directly from `hamiltonian_simulation.py`):

```python
def _simulation_qdrift(
    H: Union[np.ndarray, list],
    t: float,
    target_error: float,
    **kwargs: Any
) -> HamiltonianSimulationResult:
    from algorithms.qdrift import QDrift
    allowed_keys = {'steps'}
    for key in kwargs:
        if key not in allowed_keys:
            raise ValueError(f"Unsupported keyword argument '{key}'. Only 'steps' is allowed.")
    steps = kwargs.get('steps', 5000)
    runable = QDrift(H, t, target_error, steps=steps)
    return runable
```

Responsibilities:

1. Restrict kwargs to `steps`.
2. Instantiate `algorithms.qdrift.QDrift(H, t, target_error, steps)`.
3. Return a stochastic-trajectory simulator result object.

Tie-in with method skill:

- See [qitonghu/hamiltonian_simulation/algorithms/qdrift/skill.md](qitonghu/hamiltonian_simulation/algorithms/qdrift/skill.md) for $\lambda$-weighted sampling, trajectory variance, and reproducibility practice.

### 4) `_simulation_taylor`: Truncated-Series + LCU Wrapper

Code excerpt (directly from `hamiltonian_simulation.py`):

```python
def _simulation_taylor(
    H: Union[np.ndarray, list],
    t: float,
    target_error: float,
    **kwargs: Any
) -> HamiltonianSimulationResult:
    from algorithms.taylor import Taylor
    allowed_keys = {'degree'}
    for key in kwargs:
        if key not in allowed_keys:
            raise ValueError(f"Unsupported keyword argument '{key}'. Only 'degree' is allowed.")
    degree = kwargs.get('degree', 15)
    runable = Taylor(H, t, target_error, degree=degree)
    return runable
```

Responsibilities:

1. Restrict kwargs to `degree`.
2. Instantiate `algorithms.taylor.Taylor(H, t, target_error, degree)`.
3. Return the Taylor-LCU simulator result object.

Tie-in with method skill:

- See [qitonghu/hamiltonian_simulation/algorithms/taylor/skill.md](qitonghu/hamiltonian_simulation/algorithms/taylor/skill.md) for dynamic-programming coefficient build, LCU construction, and complexity discussion.

### 5) `_simulation_qsp`: Block-Encoding + Phase Synthesis Wrapper

Code excerpt (directly from `hamiltonian_simulation.py`):

```python
def _simulation_qsp(
    H: Union[np.ndarray, list],
    t: float,
    target_error: float,
    **kwargs
) -> HamiltonianSimulationResult:
    from algorithms.qsp import QSP
    allowed_keys = {'degree'}
    for key in kwargs:
        if key not in allowed_keys:
            raise ValueError(f"Unsupported keyword argument '{key}'. Only 'degree' is allowed.")
    degree = kwargs.get('degree', 15)
    runable = QSP(H, t, target_error, degree=degree)
    return runable
```

Responsibilities:

1. Restrict kwargs to `degree`.
2. Instantiate `algorithms.qsp.QSP(H, t, target_error, degree)`.
3. Return the QSP simulator result object.

Tie-in with method skill:

- See [qitonghu/hamiltonian_simulation/algorithms/qsp/skill.md](qitonghu/hamiltonian_simulation/algorithms/qsp/skill.md) for parity split, phase synthesis, and block-encoding formulas.

Practical note:

1. Current wrapper only forwards `degree` to QSP. If you want to expose `beta`, update `_simulation_qsp` allowed kwargs and constructor forwarding accordingly.

### 6) `_simulation_cartan_optimization`: Cartan Wrapper (Current Stub)

Code excerpt (directly from `hamiltonian_simulation.py`):

```python
def _simulation_cartan_optimization(
    H: Union[np.ndarray, list],
    t: float,
    target_error: float,
    **kwargs
) -> HamiltonianSimulationResult:
    from algorithms.cartan_lax import CartanLax
    runable = CartanLax()
    return runable
```

Notes:

1. The method key is already present in the dispatcher mapping as `cartan-optimization`.
2. This wrapper is currently minimal and does not yet expose algorithm-specific parameters.
3. There is no dedicated Cartan skill document in the current set of four method skills.

## Summary Checklist

1. [ ] Confirm `H` is a NumPy array and Hermitian.
2. [ ] Pick method according to simulation goal (deterministic vs randomized vs polynomial).
3. [ ] Pass only supported kwargs for the chosen wrapper.
4. [ ] Read `result.total_error` only after confirming method settings.
5. [ ] Record `target_qubits`, effective hyperparameters, and runtime for reproducibility.
6. [ ] Cross-check method-level assumptions using the four algorithm skill documents.

## Real-World Applications

1. Unified benchmark harness for algorithm comparison studies.
2. Educational notebooks that switch algorithms by changing one string.
3. Automated regression tests for approximation error budgets.
4. Pipeline integration in chemistry/materials simulation prototypes.
5. Interface-layer validation for future method extensions.
