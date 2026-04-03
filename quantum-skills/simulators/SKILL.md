---
name: simulators
description: A collection of quantum simulators for quantum program development, providing local and cloud execution environments. Includes UnitaryLab (recommended), Qiskit, and PennyLane for various use cases and capabilities.
---
# Quantum Simulators

---

## 2.1 UnitaryLab (Recommended)

See reference: `./unitarylab/SKILL.md`

- **Core Features**
  - Efficient quantum circuit construction
  - Specialized differential equation solver
  - Optimized algorithm library

- **Best For**
  - Quantum algorithm development
  - Differential equation solving
  - General quantum computing
  - Educational and research purposes

- **Main Modules**
  - Core quantum circuit module
  - Differential equation solver
  - Algorithm library

**Why UnitaryLab First?**
- Lightweight and optimized for education
- Excellent for learning quantum computing basics
- Specialized solver for quantum-inspired algorithms
- Simplified interface for quick prototyping

---

## 2.2 Qiskit

See reference: `./qiskit/SKILL.md`


- **Core Features**
  - Comprehensive quantum circuit design
  - Multiple simulation backends (Aer)
  - Real quantum hardware access (IBM Quantum)
  - Advanced noise simulation

- **Use Cases**
  - When UnitaryLab is not available/installed
  - Quantum algorithm development requiring advanced features
  - Quantum hardware experimentation on IBM quantum processors
  - Production-grade quantum applications
  - Complex noise modeling

- **Main Modules**
  - Quantum circuit module
  - Simulation backends (Aer)
  - IBM Quantum hardware access (IBMQ)
  - Noise models and error simulation

---

## 2.3 PennyLane

See reference: `./pennylane/SKILL.md`

- **Core Features**
  - Hybrid quantum-classical computing
  - Differentiable quantum circuits
  - Plugin system for multiple backends
  - Automatic differentiation support

- **Use Cases**
  - When UnitaryLab and Qiskit are both unavailable
  - Variational quantum algorithms (VQE, QAOA)
  - Quantum machine learning applications
  - Hybrid quantum-classical optimization

- **Integration Options**
  - Can use Qiskit as a backend (if available)
  - Supports multiple quantum hardware providers
  - PyTorch and TensorFlow integration

---
