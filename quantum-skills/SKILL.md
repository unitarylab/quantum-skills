---
name: quantum-skills
description: A comprehensive collection of quantum computing skills, including a complete skill system from basic quantum knowledge to advanced quantum algorithms. This skill provides structured guidance for quantum computing expertise development, covering programming languages, algorithm types, implementation approaches, and learning paths.
---

# Quantum Computing Skills Framework

## How to Use This Skill

This is the **root index** of the quantum-skills system. It does NOT contain implementation details itself. Instead:

1. **Identify** which category applies to the user's request (simulators vs. algorithms).
2. **Follow the matching reference path** to the next-level SKILL.md.
3. **Continue drilling down** until you reach a leaf-level SKILL.md with concrete code, API details, or examples.
4. **Always read the referenced SKILL.md before writing any code.**

Every `See reference:` line is a callable skill — open it before proceeding.

---

## Environment Setup — Do This First (Only When Running Code)

> **Install the environment only when the user intends to run or execute code.**
> For concept explanation, code review, circuit design, or documentation tasks — skip all installation steps.

### Install Decision

```
Does the task require running / executing code?
│
├─ YES → Use uv workflow first (recommended), then fallback to conda if needed
│       1) uv venv --python 3.11
│       2) uv pip install ./dist/<WHEEL>   ← pick wheel for your OS (see table below)
│       3) uv run python -c "import engine; print('UnitaryLab OK')"
│       4) uv run python <your_script>.py
│
│       If uv is unavailable or fails due to local constraints,
│       follow conda installation as fallback.
│
└─ NO  → Skip all installation. Explain, write, or review directly.
```

### UnitaryLab Installation (default simulator, uv first)

Pre-built wheels for all major platforms are located at:
```
.agents/skills/quantum-skills/simulators/unitarylab/dist/
  unitarylab-0.1.0-cp311-cp311-win_amd64.whl        ← Windows x86-64
  unitarylab-0.1.0-cp311-cp311-macosx_11_0_arm64.whl ← macOS 11+ (Apple Silicon / arm64)
  unitarylab-0.1.0-cp311-cp311-linux_x86_64.whl      ← Linux x86-64
```

**Requirements:** Python 3.11 — select the wheel matching the user's OS and architecture.

| OS | Wheel to use |
|----|-------------|
| Windows x64 | `unitarylab-0.1.0-cp311-cp311-win_amd64.whl` |
| macOS (arm64) | `unitarylab-0.1.0-cp311-cp311-macosx_11_0_arm64.whl` |
| Linux x86-64 | `unitarylab-0.1.0-cp311-cp311-linux_x86_64.whl` |

#### Using `uv` (Recommended, run this first)
```bash
# 1. Create local virtual environment
uv venv --python 3.11

# 2. Install UnitaryLab wheel — replace <WHEEL> with the file matching your OS (see table above)
uv pip install ./dist/<WHEEL>

# 3. Install common scientific dependencies
uv pip install numpy scipy scikit-learn matplotlib

# 4. Verify
uv run python -c "import engine; print('UnitaryLab OK')"

# 5. Run user script
uv run python <your_script>.py
```

#### Using `conda` (Fallback if uv is unavailable)
```bash
# 1. Create environment (skip if unitarylab-env already exists)
conda create -n unitarylab-env python=3.11

# 2. Activate
conda activate unitarylab-env

# 3. Install wheel — replace <WHEEL> with the file matching your OS (see table above)
#    Run from the unitarylab/ skill folder
python -m pip install ./dist/<WHEEL>

# 4. Verify
python -c "from engine import GateSequence; print('UnitaryLab OK')"

# 5. Run user script
python <your_script>.py
```

If step 4 prints `UnitaryLab OK`, the environment is ready.
For full details and troubleshooting: See reference `./simulators/unitarylab/SKILL.md`

---

## Skill Tree

```
quantum-skills/                          ← YOU ARE HERE
├── simulators/SKILL.md                  ← Simulator selection & setup
│   ├── unitarylab/SKILL.md              ← DEFAULT simulator (use first)
│   ├── qiskit/SKILL.md                  ← IBM ecosystem / noise models
│   └── pennylane/SKILL.md               ← Differentiable / VQE / QML
│
└── algorithms/SKILL.md                  ← Algorithm family index
    ├── primitives/SKILL.md              ← Foundational subroutines
    │   ├── amplitude-amplification/SKILL.md
    │   ├── amplitude-estimation/SKILL.md
    │   ├── quantum-phase-estimation/SKILL.md
    │   ├── hadamard-test/SKILL.md
    │   └── hadamard-transform/SKILL.md
    │
    ├── linear-systems/SKILL.md          ← Quantum linear algebra
    │   ├── hhl/SKILL.md
    │   ├── lcu/SKILL.md
    │   ├── qsvt-qlsa/SKILL.md
    │   └── quantum-signal-processing/SKILL.md
    │
    ├── cryptography/SKILL.md            ← Quantum attacks on classical crypto
    │   ├── shor/SKILL.md
    │   ├── discretelog/SKILL.md
    │   └── simon/SKILL.md
    │
    ├── hamiltonian-simulation/SKILL.md  ← Time evolution of quantum systems
    │   ├── trotter/SKILL.md
    │   ├── qdrift/SKILL.md
    │   ├── taylor/SKILL.md
    │   └── qsp/SKILL.md
    │
    └── schrodingerization/SKILL.md      ← Quantum PDE solvers
        ├── schrodingerization-skill/SKILL.md   ← Start here for method overview
        ├── advection-schrodingerization/SKILL.md
        ├── backward-heat-1d-schrodingerization/SKILL.md
        ├── backward-heat-2d-schrodingerization/SKILL.md
        ├── black-scholes-schrodingerization/SKILL.md
        ├── burgers-1d-schrodingerization/SKILL.md
        ├── burgers2d-schrodingerization-agent/SKILL.md
        ├── elastic-wave-schrodingerization/SKILL.md
        ├── heat-1d-schrodingerization/SKILL.md
        ├── heat-2d-schrodingerization/SKILL.md
        ├── heat-var-coeff-1d-schrodingerization/SKILL.md
        ├── helmholtz-1d-schrodingerization/SKILL.md
        ├── maxwell-1d-schrodingerization/SKILL.md
        ├── quantum-general-linear-1d/SKILL.md
        ├── quantum-multiscale-elliptic/SKILL.md
        ├── quantum-multiscale-transport-1d/SKILL.md
        ├── quantum-ou-process-1d/SKILL.md
        ├── quantum-schrodinger-abc-simulation/SKILL.md
        └── traffic-flow-schrodingerization-solver/SKILL.md
```

---

## 1. Quantum Simulators

Quantum simulators provide the execution environment for all algorithm examples. **Choose a simulator before writing any algorithm code.**

**Selection rule (read `./simulators/SKILL.md` for full details):**
- Default: **UnitaryLab** — fast prototyping, PDE/Schrodingerization workflows, educational demos.
- Fallback: **Qiskit** — richer ecosystem, noise models, path to IBM hardware.
- Special case: **PennyLane** — differentiable circuits, VQE, QAOA, QML, PyTorch/TensorFlow integration.

See reference: `./simulators/SKILL.md`

### Simulator Sub-Skills
| Simulator | When to Use | Reference |
|-----------|-------------|-----------|
| UnitaryLab | Default. Prototyping, Schrodingerization, education. | `./simulators/unitarylab/SKILL.md` |
| Qiskit | IBM hardware path, noise models, broader ecosystem. | `./simulators/qiskit/SKILL.md` |
| PennyLane | VQE / QAOA / QML, differentiable circuits, ML-framework integration. | `./simulators/pennylane/SKILL.md` |

---

## 2. Quantum Algorithms

An extensive collection of quantum algorithms. Navigate to the appropriate sub-skill for concrete implementations.

See reference: `./algorithms/SKILL.md`

### 2.1 Quantum Primitives
Foundational subroutines used as building blocks inside larger algorithms.

See reference: `./algorithms/primitives/SKILL.md`

| Primitive | Reference |
|-----------|-----------|
| Amplitude Amplification (Grover-style) | `./algorithms/primitives/amplitude-amplification/SKILL.md` |
| Amplitude Estimation | `./algorithms/primitives/amplitude-estimation/SKILL.md` |
| Quantum Phase Estimation (QPE) | `./algorithms/primitives/quantum-phase-estimation/SKILL.md` |
| Hadamard Test | `./algorithms/primitives/hadamard-test/SKILL.md` |
| Hadamard Transform | `./algorithms/primitives/hadamard-transform/SKILL.md` |

### 2.2 Quantum Linear Systems
Algorithms for solving linear systems $Ax = b$ with quantum speedup.

See reference: `./algorithms/linear-systems/SKILL.md`

| Algorithm | Reference |
|-----------|-----------|
| HHL Algorithm | `./algorithms/linear-systems/hhl/SKILL.md` |
| LCU (Linear Combination of Unitaries) | `./algorithms/linear-systems/lcu/SKILL.md` |
| QSVT-QLSA | `./algorithms/linear-systems/qsvt-qlsa/SKILL.md` |
| Quantum Signal Processing (QSP) | `./algorithms/linear-systems/quantum-signal-processing/SKILL.md` |

### 2.3 Quantum Cryptography
Quantum algorithms that attack or underpin classical cryptographic protocols.

See reference: `./algorithms/cryptography/SKILL.md`

| Algorithm | Reference |
|-----------|-----------|
| Shor's Algorithm (integer factorization / RSA) | `./algorithms/cryptography/shor/SKILL.md` |
| Discrete Logarithm Problem (DLP) Solver | `./algorithms/cryptography/discretelog/SKILL.md` |
| Simon's Algorithm (hidden subgroup) | `./algorithms/cryptography/simon/SKILL.md` |

### 2.4 Hamiltonian Simulation
Algorithms for approximating time evolution $e^{-iHt}$. All share the unified `hamiltonian_simulation(...)` entry point.

See reference: `./algorithms/hamiltonian-simulation/SKILL.md`

| Method | Reference |
|--------|-----------|
| Trotter-Suzuki decomposition | `./algorithms/hamiltonian-simulation/trotter/SKILL.md` |
| qDRIFT (randomized) | `./algorithms/hamiltonian-simulation/qdrift/SKILL.md` |
| Taylor series expansion | `./algorithms/hamiltonian-simulation/taylor/SKILL.md` |
| Quantum Signal Processing (QSP) | `./algorithms/hamiltonian-simulation/qsp/SKILL.md` |

### 2.5 Schrodingerization — Quantum PDE Solvers
Converts non-unitary PDEs into Schrödinger-type equations solvable on a quantum computer. **Read the method overview first, then choose the equation-specific skill.**

See reference: `./algorithms/schrodingerization/SKILL.md`

| PDE / Equation | Reference |
|----------------|-----------|
| **Method overview (start here)** | `./algorithms/schrodingerization/schrodingerization-skill/SKILL.md` |
| 1D Advection | `./algorithms/schrodingerization/advection-schrodingerization/SKILL.md` |
| 1D Backward Heat | `./algorithms/schrodingerization/backward-heat-1d-schrodingerization/SKILL.md` |
| 2D Backward Heat | `./algorithms/schrodingerization/backward-heat-2d-schrodingerization/SKILL.md` |
| Black-Scholes (1D) | `./algorithms/schrodingerization/black-scholes-schrodingerization/SKILL.md` |
| Burgers' Equation (1D) | `./algorithms/schrodingerization/burgers-1d-schrodingerization/SKILL.md` |
| Burgers' Equation (2D) | `./algorithms/schrodingerization/burgers2d-schrodingerization-agent/SKILL.md` |
| Elastic Wave (1D & 2D) | `./algorithms/schrodingerization/elastic-wave-schrodingerization/SKILL.md` |
| Heat Equation (1D) | `./algorithms/schrodingerization/heat-1d-schrodingerization/SKILL.md` |
| Heat Equation (2D) | `./algorithms/schrodingerization/heat-2d-schrodingerization/SKILL.md` |
| Heat — Variable Coefficients (1D) | `./algorithms/schrodingerization/heat-var-coeff-1d-schrodingerization/SKILL.md` |
| Helmholtz (1D) | `./algorithms/schrodingerization/helmholtz-1d-schrodingerization/SKILL.md` |
| Maxwell Equations (1D) | `./algorithms/schrodingerization/maxwell-1d-schrodingerization/SKILL.md` |
| General Linear PDE (1D) | `./algorithms/schrodingerization/quantum-general-linear-1d/SKILL.md` |
| Multiscale Elliptic | `./algorithms/schrodingerization/quantum-multiscale-elliptic/SKILL.md` |
| Multiscale Transport (1D) | `./algorithms/schrodingerization/quantum-multiscale-transport-1d/SKILL.md` |
| Ornstein-Uhlenbeck Process (1D) | `./algorithms/schrodingerization/quantum-ou-process-1d/SKILL.md` |
| Schrödinger + Absorbing BCs | `./algorithms/schrodingerization/quantum-schrodinger-abc-simulation/SKILL.md` |
| Traffic Flow (LWR model) | `./algorithms/schrodingerization/traffic-flow-schrodingerization-solver/SKILL.md` |

---

## Quick-Start Decision Tree

```
User request
│
├─ "Which simulator should I use?" or "Set up environment" or "Run this code/script"
│       └─ Read: ./simulators/SKILL.md
│           └─ For execution: prefer uv setup + uv run first
│
├─ "Solve a PDE / differential equation"
│       └─ Read: ./algorithms/schrodingerization/SKILL.md
│           └─ Then pick the equation-specific SKILL.md from section 2.5 above
│
├─ "Simulate a physical Hamiltonian / time evolution"
│       └─ Read: ./algorithms/hamiltonian-simulation/SKILL.md
│
├─ "Solve a linear system Ax = b"
│       └─ Read: ./algorithms/linear-systems/SKILL.md
│
├─ "Factor a number / break RSA / discrete log"
│       └─ Read: ./algorithms/cryptography/SKILL.md
│
└─ "Build a circuit primitive (QPE, Grover, Hadamard, ...)"
        └─ Read: ./algorithms/primitives/SKILL.md
```
