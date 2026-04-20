<div align="center">

<h1>&#9883; Quantum Skill</h1>

<p>
  <strong>A practical skill for learning quantum computing algorithms and programming.</strong><br/>
  <strong>一个用于学习量子算法与编程的实用技能。</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3b82f6?style=flat-square&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/UnitaryLab-0.1.0-7c3aed?style=flat-square" alt="UnitaryLab 0.1.0"/>
  <img src="https://img.shields.io/badge/VS_Code_Copilot-Skill-22c55e?style=flat-square&logo=visualstudiocode&logoColor=white" alt="VS Code Copilot Skill"/>
  <img src="https://img.shields.io/badge/Claude_Code-Skill-7c3aed?style=flat-square&logo=anthropic&logoColor=white" alt="Claude Code Skill"/>
  <img src="https://img.shields.io/badge/Quantum_Algorithms-30%2B-f59e0b?style=flat-square" alt="30+ Quantum Algorithms"/>
</p>

<p>
  <a href="#english">English</a>
  &middot;
  <a href="#chinese">中文</a>
</p>

</div>

---

<a name="english"></a>

## English

### What is this?

**Quantum Skill** is a VS Code Copilot agent skill for quantum computing. It provides structured, progressive guidance to help Copilot explain concepts, write code, and run quantum algorithm demos using [UnitaryLab](https://unitarylab.com), Qiskit, or PennyLane.

---

### &#10024; Key Features

- **Progressive Disclosure** — Root `SKILL.md` is lightweight; algorithm and simulator guides load only when needed.
- **Full Algorithm Coverage** — Primitives, linear systems, cryptography, Hamiltonian simulation, and PDE solvers.
- **Multi-Simulator Support** — UnitaryLab (recommended), Qiskit, and PennyLane, with clear selection rules.
- **Run-Ready Setup** — Pre-built wheels for Windows, macOS, and Linux; `uv`-based one-command install.
- **Education-Friendly** — Suitable for concept explanation, circuit design, code review, and hands-on demos.

---

### &#127775; Algorithms Covered

| Category | Algorithms |
|----------|-----------|
| **Primitives** | QPE, Hadamard Test, Hadamard Transform, Amplitude Amplification, Amplitude Estimation |
| **Linear Systems** | HHL, LCU, QSVT, Quantum Signal Processing (QSP) |
| **Cryptography** | Shor's Algorithm, Discrete Logarithm, Simon's Algorithm |
| **Hamiltonian Simulation** | Trotter, QDrift, Taylor Series, QSP |
| **Schrodingerization** | Advection, Heat (1D/2D), Burgers, Black-Scholes, Maxwell, Helmholtz, Elastic Wave, Multiscale, OU Process, … |

---

### &#128187; Supported Simulators

| Simulator | When to Use | Platform |
|-----------|-------------|----------|
| **UnitaryLab** *(default)* | Learning, algorithm demos, PDE workflows | Win / macOS / Linux |
| **Qiskit** | Noise models, IBM hardware workflows | Win / macOS / Linux |
| **PennyLane** | Differentiable hybrid optimization | Win / macOS / Linux |

---

### &#128193; Repository Structure

```
quantum-skills/
|
+-- SKILL.md                    # Root index — entry point for the agent
+-- README.md
|
+-- algorithms/                 # Quantum algorithm skills
|   +-- primitives/             # QPE, Hadamard test/transform, AA, AE
|   +-- linear-systems/         # HHL, LCU, QSVT, QSP
|   +-- cryptography/           # Shor, discrete logarithm, Simon
|   +-- hamiltonian-simulation/ # Trotter, QDrift, Taylor, QSP
|   +-- schrodingerization/     # PDE solvers (20+ problem types)
|
+-- simulators/                 # Simulator selection & installation guides
    +-- unitarylab/             # Recommended — pre-built wheels included
    |   +-- dist/               # .whl files for Win / macOS / Linux
    +-- qiskit/
    +-- pennylane/
```

---

### &#128640; Installation

**Install into your project using `bunx` (recommended) or `npx`:**

```bash
# Using Bun (recommended)
bunx skills add https://github.com/unitarylab/quantum-skills

# Using npm / npx
npx skills add https://github.com/unitarylab/quantum-skills
```

This places the skill under `.agents/skills/quantum-skills/` in your workspace — Copilot will discover it automatically.

**Or clone manually:**

```bash
# macOS / Linux
git clone https://github.com/unitarylab/quantum-skills \
  .agents/skills/quantum-skills

# Windows (PowerShell)
git clone https://github.com/unitarylab/quantum-skills `
  .agents/skills/quantum-skills
```

> The skill itself requires no Python installation. The UnitaryLab simulator is only needed when you intend to **run** code — setup instructions are inside `simulators/unitarylab/SKILL.md`.

---

### &#128161; Usage

This skill is automatically invoked by Copilot when you ask about quantum computing topics. The agent reads the relevant `SKILL.md` files and drills down to the appropriate leaf-level guide before generating any code.

**Example prompts:**

| Prompt | What happens |
|--------|-------------|
| `Implement Grover's algorithm using UnitaryLab` | Loads primitives guide — amplitude amplification |
| `Explain the HHL algorithm with a 2×2 example` | Loads linear-systems/hhl guide with matrix demo |
| `Simulate 1D heat equation with Schrodingerization` | Loads schrodingerization/heat-1d guide |
| `Compare Trotter and QDrift for Hamiltonian simulation` | Loads hamiltonian-simulation guide |
| `Run Shor's algorithm on n=15` | Loads cryptography/shor guide |

---

## License

This repository is licensed under the Apache License 2.0.  
The `unitarylab` package itself contains proprietary binary components —
see the `LICENSE` and `LICENSE-PROPRIETARY` files bundled inside the wheel.

---

<a name="chinese"></a>

## 中文

### 这是什么？

**Quantum Skill** 是一个面向量子计算的 VS Code Copilot Agent 技能包。它为 Copilot 提供结构化、按需加载的引导，帮助其解释量子概念、编写代码，并基于 [UnitaryLab](https://unitarylab.com)、Qiskit 或 PennyLane 运行量子算法示例。

---

### &#10024; 核心特性

- **渐进式加载** — 根 `SKILL.md` 轻量，算法与模拟器指南仅在需要时才加载。
- **算法全覆盖** — 基元、线性系统、密码学、哈密顿量模拟、PDE 求解器一应俱全。
- **多模拟器支持** — UnitaryLab（推荐）、Qiskit、PennyLane，附明确选型规则。
- **开箱即用** — 提供 Windows / macOS / Linux 预编译 wheel，一条命令完成安装。
- **教学友好** — 适用于概念解释、电路设计、代码审查和动手实验。

---

### &#127775; 算法覆盖范围

| 分类 | 算法 |
|------|------|
| **基础量子算法** | QPE、Hadamard 测试、Hadamard 变换、振幅放大、振幅估计 |
| **线性系统** | HHL、LCU、QSVT、量子信号处理（QSP） |
| **密码学** | Shor 算法、离散对数、Simon 算法 |
| **哈密顿量模拟** | Trotter、QDrift、Taylor 级数、QSP |
| **Schrodingerization** | 对流、热方程（一维/二维）、Burgers、Black-Scholes、Maxwell、Helmholtz、弹性波、多尺度、OU 过程 … |

---

### &#128187; 支持的模拟器

| 模拟器 | 适用场景 | 平台 |
|--------|---------|------|
| **UnitaryLab** *(默认)* | 学习、算法演示、PDE 工作流 | Win / macOS / Linux |
| **Qiskit** | 噪声模型、IBM 硬件工作流 | Win / macOS / Linux |
| **PennyLane** | 可微分混合优化 | Win / macOS / Linux |

---

### &#128193; 仓库结构

```
quantum-skills/
|
+-- SKILL.md                    # 根索引 — Agent 入口
+-- README.md
|
+-- algorithms/                 # 量子算法技能
|   +-- primitives/             # QPE、Hadamard 测试/变换、振幅放大与估计
|   +-- linear-systems/         # HHL、LCU、QSVT、QSP
|   +-- cryptography/           # Shor、离散对数、Simon
|   +-- hamiltonian-simulation/ # Trotter、QDrift、Taylor、QSP
|   +-- schrodingerization/     # PDE 求解器（20+ 问题类型）
|
+-- simulators/                 # 模拟器选型与安装指南
    +-- unitarylab/             # 推荐 — 内含预编译 wheel
    |   +-- dist/               # Win / macOS / Linux .whl 文件
    +-- qiskit/
    +-- pennylane/
```

---

### &#128640; 安装

**使用 `bunx`（推荐）或 `npx` 安装到项目中：**

```bash
# 使用 Bun（推荐）
bunx skills add https://github.com/unitarylab/quantum-skills

# 使用 npm / npx
npx skills add https://github.com/unitarylab/quantum-skills
```

命令会将技能放置到工作区的 `.agents/skills/quantum-skills/` 目录下，Copilot 将自动发现并加载它。

**或手动克隆：**

```bash
# macOS / Linux
git clone https://github.com/unitarylab/quantum-skills \
  .agents/skills/quantum-skills

# Windows (PowerShell)
git clone https://github.com/unitarylab/quantum-skills `
  .agents/skills/quantum-skills
```

> 技能本身无需 Python 环境。UnitaryLab 模拟器仅在需要**执行代码**时才需安装，安装步骤详见 `simulators/unitarylab/SKILL.md`。

---

### &#128161; 使用方法

在 VS Code 中向 Copilot 提问量子计算相关问题时，该技能会自动被调用。Agent 会逐级读取对应的 `SKILL.md` 文件，找到最匹配的叶级指南后再生成代码。

**prompt 示例：**

| Prompt | 触发行为 |
|--------|---------|
| `用 UnitaryLab 实现 Grover 算法` | 加载基元指南 — 振幅放大 |
| `解释 HHL 算法，并给出 2×2 的示例` | 加载线性系统/hhl 指南，含矩阵示例 |
| `用 Schrodingerization 模拟一维热方程` | 加载 schrodingerization/heat-1d 指南 |
| `比较 Trotter 和 QDrift 在哈密顿量模拟中的异同` | 加载哈密顿量模拟指南 |
| `对 n=15 运行 Shor 算法` | 加载密码学/shor 指南 |

---

## License

This repository is licensed under the Apache License 2.0.  
The `unitarylab` package itself contains proprietary binary components —
see the `LICENSE` and `LICENSE-PROPRIETARY` files bundled inside the wheel.