---
name: cryptography
description: Quantum algorithms that attack classical cryptographic protocols, including Shor's integer factorization, discrete logarithm solving, and Simon's hidden subgroup problem.
---

# Quantum Cryptography Algorithms

## Purpose
This file routes requests for quantum algorithms with cryptographic relevance.

Use this category when the user asks about factoring, discrete logarithms, hidden-period problems, or educational demonstrations of quantum attacks on classical cryptographic assumptions.

## Routing Rules

- If the user asks about integer factoring, RSA-style examples, period finding for modular exponentiation, or Shor's algorithm:
  - Read `./shor/SKILL.md`
- If the user asks about solving a discrete logarithm problem or attacking DLP-based systems:
  - Read `./discretelog/SKILL.md`
- If the user asks about Simon's hidden XOR mask problem, oracle promise problems, or exponential separation examples:
  - Read `./simon/SKILL.md`

## Available Leaf Skills

1. Shor's Algorithm: `./shor/SKILL.md`
2. Discrete Logarithm Problem Solver: `./discretelog/SKILL.md`
3. Simon's Algorithm: `./simon/SKILL.md`

## Response Contract

1. Identify the cryptographic problem family.
2. Read the matching leaf skill before writing code or commands.
3. Keep explanations educational and avoid implying practical unauthorized cryptographic exploitation.
