<div align="center">

<h1>&#9883; Quantum Skill</h1>

<p>
  <strong>A practical skill for learning quantum computing algorithms and programming.</strong><br/>
  <strong>一个用于学习量子算法与编程的实用技能。</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Algorithm_Guides-51-f59e0b?style=flat-square" alt="51 Algorithm Guides"/>
  <img src="https://img.shields.io/badge/Categories-12-3b82f6?style=flat-square" alt="12 Categories"/>
  <a href="#installation"><img src="https://img.shields.io/badge/Codex%20%7C%20Claude%20Code%20%7C%20VS%20Code%20Copilot-Install_Skill-111827?style=flat-square" alt="Install Skill for Codex, Claude Code, and VS Code Copilot"/></a>
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

**Quantum Skill** provides agent guidance for explaining quantum computing, writing code, and running examples with [UnitaryLab](https://unitarylab.com), Qiskit, or PennyLane.

- **Progressive loading** — The root [SKILL.md](SKILL.md) routes requests to algorithm or simulator guides as needed.
- **Theory and practice** — Guides cover concepts, circuit construction, implementation references, debugging, and validation.
- **Simulator selection** — UnitaryLab is the default; Qiskit and PennyLane support hardware/noise and differentiable workflows respectively.

### Coverage

**12 categories, 51 algorithm and workflow guides.** Counts exclude routing indexes and simulator guides; the quantum error correction tutorial counts as one guide. These include classical reference methods and supporting routines, not just distinct quantum algorithms.

| Category | Guides | Topics |
|----------|-------:|--------|
| Primitives | 6 | Grover, QPE, Hadamard Test/Transform, Amplitude Amplification/Estimation |
| Linear Systems | 7 | HHL, LCU, AQC, QFT, QSP, QSVT-QLSA, VQLS |
| Cryptography | 3 | Shor, Discrete Logarithm, Simon |
| Search | 4 | Glued Trees, Hidden Shift, Quantum Walk Search, Quantum Backtracking |
| Hamiltonian Simulation | 5 | Trotter, QDrift, Taylor Series, QSP, Cartan |
| Schrodingerization | 3 | Advection, Heat (1D/2D) |
| Quantum Machine Learning | 7 | VQE, VQC, QAOA, QCBM, CVQNN, Fermi-Hubbard VQE, 2D Ising Dynamics |
| State Preparation | 5 | Sparse Superposition, Möttönen, MPS, Multiplexer, Pauli-Word Preparation |
| Quantum Chemistry | 1 | Molecular DMRG |
| Eigensolvers | 3 | NumPy Eigensolver, NumPy Minimum Eigensolver, VQD |
| Gradients | 6 | Parameter-shift, Finite-difference, Linear-combination, SPSA, Reverse-mode, QFI |
| Quantum Error Correction | 1 | LDPC, CSS, Hypergraph Product Codes |

See [algorithm routing](algorithms/SKILL.md) for entry points.

### Simulators

| Simulator | Typical use |
|-----------|-------------|
| [UnitaryLab](simulators/unitarylab/SKILL.md) *(default)* | Circuit programming, algorithm demos, PDE workflows |
| [Qiskit](simulators/qiskit/SKILL.md) | Noise models, IBM hardware workflows |
| [PennyLane](simulators/pennylane/SKILL.md) | Differentiable hybrid optimization |

### Installation

Install as a custom skill for Codex, Claude Code, or another compatible agent. Choose the target agent and project/user scope when prompted by the installer:

```bash
bunx skills add https://github.com/unitarylab/quantum-skills
# Or:
npx skills add https://github.com/unitarylab/quantum-skills
```

For manual installation, clone the **complete repository** into the appropriate directory:

| Agent | Project only | All local projects |
|-------|--------------|-------------------|
| Codex | `.agents/skills/quantum-skills/` | `~/.agents/skills/quantum-skills/` |
| Claude Code | `.claude/skills/quantum-skills/` | `~/.claude/skills/quantum-skills/` |

Project-level examples (run from your project root; choose the agent you use):

```bash
# Codex
git clone https://github.com/unitarylab/quantum-skills .agents/skills/quantum-skills
# Claude Code
git clone https://github.com/unitarylab/quantum-skills .claude/skills/quantum-skills
```

For user-level installation, use the corresponding `~` path from the table. Keep the root `SKILL.md`, `algorithms/`, and `simulators/` together so relative references work. This installs a custom skill, not an officially bundled skill.

In Codex, you can also ask: `$skill-installer install the root skill from https://github.com/unitarylab/quantum-skills as quantum-skills, retaining its algorithms and simulators directories.` If the skill does not appear after installation, restart Codex.

References: [Codex documentation](https://learn.chatgpt.com/docs/build-skills), [Claude Code documentation](https://code.claude.com/docs/en/skills), [skills installer](https://github.com/vercel-labs/skills).

The skill documents require no Python environment. To run examples, follow the selected [simulator guide](simulators/SKILL.md) for Python compatibility and dependencies.

### Usage

Ask your agent a quantum computing question. It follows the skill indexes to the relevant guide before generating code.

To select it explicitly, use `$quantum-skills` in Codex or `/quantum-skills` in Claude Code, followed by your request.

| Example prompt | Guide |
|----------------|-------|
| `Implement Grover's algorithm using UnitaryLab` | [Grover](algorithms/primitives/grover/SKILL.md) |
| `Explain HHL with a 2×2 example` | [HHL](algorithms/linear-systems/hhl/SKILL.md) |
| `Simulate the 1D heat equation with Schrodingerization` | [1D Heat](algorithms/schrodingerization/heat-1d-schrodingerization/SKILL.md) |
| `Search a graph with multiple marked vertices using a quantum walk` | [Quantum Walk Search](algorithms/search/quantum-walk-search/SKILL.md) |
| `Explain quantum backtracking on a small binary CNF problem` | [Quantum Backtracking](algorithms/search/quantum-backtracking/SKILL.md) |

---

<a name="chinese"></a>

## 中文

**Quantum Skill** 为 Agent 提供量子计算指南，帮助其解释概念、编写代码，并基于 [UnitaryLab](https://unitarylab.com)、Qiskit 或 PennyLane 运行示例。

- **渐进式加载** — 根 [SKILL.md](SKILL.md) 按需路由到算claudecode法或模拟器指南。
- **理论与实践** — 涵盖概念、电路构建、实现参考、调试和验证。
- **模拟器选型** — 默认使用 UnitaryLab；Qiskit 和 PennyLane 分别支持硬件与噪声、可微分计算工作流。

### 覆盖范围

**12 个分类，51 篇算法与工作流指南。** 数量不含路由索引和模拟器指南，量子纠错教程计为一篇；其中包含经典参考方法和辅助例程，并非 51 种独立量子算法。

| 分类 | 指南数 | 内容 |
|------|-------:|------|
| 基础量子算法 | 6 | Grover、QPE、Hadamard 测试与变换、振幅放大与估计 |
| 线性系统 | 7 | HHL、LCU、AQC、QFT、QSP、QSVT-QLSA、VQLS |
| 密码学 | 3 | Shor、离散对数、Simon |
| 量子搜索 | 4 | Glued Trees、Hidden Shift、Quantum Walk Search（量子游走搜索）、Quantum Backtracking（量子回溯） |
| 哈密顿量模拟 | 5 | Trotter、QDrift、Taylor 级数、QSP、Cartan |
| Schrodingerization | 3 | 对流、热方程（一维/二维） |
| 量子机器学习 | 7 | VQE、VQC、QAOA、QCBM、CVQNN、Fermi-Hubbard VQE、二维 Ising 动力学 |
| 态制备 | 5 | 稀疏叠加态、Möttönen、MPS、Multiplexer、Pauli 字符串态制备 |
| 量子化学 | 1 | 分子 DMRG |
| 本征求解器 | 3 | NumPy 本征求解器、NumPy 最小本征求解器、VQD |
| 梯度方法 | 6 | 参数位移、有限差分、线性组合、SPSA、反向模式、QFI |
| 量子纠错 | 1 | LDPC、CSS、超图乘积码 |

入口见[算法索引](algorithms/SKILL.md)。

### 模拟器

| 模拟器 | 适用场景 |
|--------|----------|
| [UnitaryLab](simulators/unitarylab/SKILL.md) *（默认）* | 电路编程、算法演示、PDE 工作流 |
| [Qiskit](simulators/qiskit/SKILL.md) | 噪声模型、IBM 硬件工作流 |
| [PennyLane](simulators/pennylane/SKILL.md) | 可微分混合优化 |

### 安装

可安装为 Codex、Claude Code 或其他兼容 Agent 的自定义技能。使用以下安装器时，按提示选择目标 Agent 及项目级或用户级范围：

```bash
bunx skills add https://github.com/unitarylab/quantum-skills
# 或：
npx skills add https://github.com/unitarylab/quantum-skills
```

手动安装时，将**完整仓库**克隆到对应目录：

| Agent | 仅当前项目 | 所有本地项目 |
|-------|------------|--------------|
| Codex | `.agents/skills/quantum-skills/` | `~/.agents/skills/quantum-skills/` |
| Claude Code | `.claude/skills/quantum-skills/` | `~/.claude/skills/quantum-skills/` |

项目级安装示例（在项目根目录运行，选择所用 Agent）：

```bash
# Codex
git clone https://github.com/unitarylab/quantum-skills .agents/skills/quantum-skills
# Claude Code
git clone https://github.com/unitarylab/quantum-skills .claude/skills/quantum-skills
```

用户级安装使用表中的 `~` 路径。请保留根 `SKILL.md`、`algorithms/` 和 `simulators/` 的相对位置，确保引用有效。安装后属于自定义技能，并非官方预装技能。

也可在 Codex 中输入：`$skill-installer 从 https://github.com/unitarylab/quantum-skills 安装仓库根技能，命名为 quantum-skills，并保留 algorithms 和 simulators 目录。` 安装后若未显示，请重启 Codex。

参考：[Codex 官方文档](https://learn.chatgpt.com/docs/build-skills)、[Claude Code 官方文档](https://code.claude.com/docs/en/skills)、[skills 安装器](https://github.com/vercel-labs/skills)。

技能文档本身无需 Python 环境。运行示例时，请按所选[模拟器指南](simulators/SKILL.md)配置 Python 版本和依赖。

### 使用

向 Agent 提出量子计算问题，它会沿技能索引读取对应指南，再生成代码。

显式调用时，在 Codex 中使用 `$quantum-skills`，在 Claude Code 中使用 `/quantum-skills`，后接具体问题。

| 示例问题 | 指南 |
|----------|------|
| `用 UnitaryLab 实现 Grover 算法` | [Grover](algorithms/primitives/grover/SKILL.md) |
| `解释 HHL 算法，并给出 2×2 示例` | [HHL](algorithms/linear-systems/hhl/SKILL.md) |
| `用 Schrodingerization 模拟一维热方程` | [一维热方程](algorithms/schrodingerization/heat-1d-schrodingerization/SKILL.md) |
| `用量子游走搜索图上的多个标记顶点` | [量子游走搜索](algorithms/search/quantum-walk-search/SKILL.md) |
| `用一个小型二元 CNF 问题解释量子回溯` | [量子回溯](algorithms/search/quantum-backtracking/SKILL.md) |
