# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional
from engine import GateSequence, Register
from core.pauli_string_decomposition import pauli_string_decomposition, pauli_string_circuit, pauli_string_multiply, pauli_string_to_matrix, pauli_string_power
from core.linear_combination_unitary import LCU
from algorithms import HamiltonianSimulationResult
import cmath
import math
from collections import defaultdict
import itertools

class Taylor(HamiltonianSimulationResult):
    """
    Taylor series expansion for Hamiltonian simulation.

    This class implements the truncated Taylor series expansion of the time evolution
    operator exp(-iHt) and expresses it as a linear combination of unitaries (LCU).
    The expansion order is automatically determined based on the desired error bound
    and the Hamiltonian norm.

    Attributes:
        alpha : float
            Spectral norm of the Hamiltonian.
        lam : float
            Product alpha * t, used to estimate the required expansion order.
        degree : int
            Truncation order of the Taylor series.
    """

    def __init__(self, H: np.ndarray, t: float, target_error: float, degree: int = 10):
        """
        Initialize the Taylor series simulator.

        Parameters:
            H : numpy.ndarray
                Hamiltonian matrix (square Hermitian).
            t : float
                Total evolution time.
            target_error : float
                Desired approximation error (used to determine the truncation order).
            degree : int, optional
                Initial guess for the truncation order. The actual order is adjusted
                based on lam and target_error, clamped between a computed minimum and 15.
                Default is 10.
        """
        super().__init__('taylor', H, t, target_error)
        self.triangular = False
        if np.linalg.norm(H.imag) < self.tol:
            n = H.shape[0]
            # Build a mask marking positions where |i-j| > 1
            mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) > 1
            # Check whether those positions are close to zero
            if np.allclose(H[mask], 0, atol=self.tol, rtol=self.tol):
                print('Use triangular Trotter.')
                self.triangular = True
        self.alpha = np.linalg.norm(H, 2)
        self.lam = self.alpha * self.t
        self.time_split = 0.5
        self.degree = min(max(degree, int(np.ceil(self.lam * 1.5 + np.log(1/self.target_error) * 1.5))), 15)
        self._run()

    def _make_U_rotation(self, qc: GateSequence, phase_angle: float, target_qubits: int) -> GateSequence:
        """
        Apply a global phase rotation to a copy of the given gate sequence.

        This method creates a copy of the input circuit and appends a global phase
        gate with the specified angle. It is used to incorporate the complex phases
        arising from the Taylor coefficients into the LCU building blocks.

        Parameters:
            qc : GateSequence
                Input quantum circuit (typically a Pauli string circuit).
            phase_angle : float
                Angle (in radians) for the global phase gate.
            target_qubits : int
                Number of qubits in the system (used to create the copy).

        Returns:
            GateSequence
                A new circuit with the same gates as `qc` plus an appended global phase.
        """
        qc_copy = qc.copy()
        # if abs(phase_angle) > 1e-12:
        qc_copy.gp(phase_angle)
        return qc_copy

    def _run(self) -> None:
        """
        Build the quantum circuit implementing the truncated Taylor series evolution.
        ...
        """
        # Split total time into r slices to reduce the required Taylor order
        r = int(self.lam / self.time_split) + 1
        
        
        # Decompose the Hamiltonian for a single slice: (H * t / r) into Pauli strings
        ans_decomposition = self._pauli_decompose(self.H, self.t / r)
        L = len(ans_decomposition)
        
        # term_map[k] will store coefficients for (Ht/r)^k (k-th order term)
        ans_term_map = dict()
        for k in range(self.degree + 1):
            ans_term_map[k] = defaultdict(complex)
        # Zero‑th order term: identity
        ans_term_map[0]["I" * self.target_qubits] = 1.0

        # Build the Taylor series for a single slice using dynamic programming
        for k in range(1, self.degree + 1):
            for str_prev in ans_term_map[k-1]:
                for i in range(L):
                    # Multiply the previous product with one Pauli term
                    ans_str, ans_val = pauli_string_multiply(str_prev, ans_decomposition[i][0])
                    # Accumulate coefficient: previous * Pauli factor * original coeff * (-i/k)
                    ans_term_map[k][ans_str] += (ans_term_map[k-1][str_prev] * ans_val * ans_decomposition[i][1] * -1j / k)

        # Combine all orders to get the single‑slice evolution operator
        ans_term_list = defaultdict(complex)
        for k in range(self.degree + 1):
            for str_ in ans_term_map[k]:
                ans_term_list[str_] += ans_term_map[k][str_]
        ans_term_list = [(key, val) for key, val in ans_term_list.items()]
        
        # Raise the slice operator to the power r: (U_slice)^r ≈ exp(-iHt)
        term_list = pauli_string_power(ans_term_list, r)
        
        # Optional: reconstruct matrix from Pauli terms (for verification)
        # equ_H1 = np.zeros_like(self.H, dtype = complex)
        # for key, val in term_list:
        #     equ_H1 += pauli_string_to_matrix(key) * val

        # Convert each term into an LCU building block: unitary + weight
        LCU_terms = list()
        for key, coef in term_list:
            magnitude = abs(coef)                      # weight for LCU
            phase = cmath.phase(coef)                  # global phase to absorb the complex argument
            # Create the Pauli circuit and add the global phase
            U_rotation = self._make_U_rotation(pauli_string_circuit(key), phase, self.target_qubits)
            LCU_terms.append((U_rotation, magnitude))

        # Optional: verify that the weighted sum of unitaries matches the target matrix
        # equ_H2 = np.zeros_like(self.H, dtype = complex)
        # for key, val in LCU_terms:
        #     equ_H2 += key.get_matrix() * val
        
        # Build the LCU circuit
        circuit = self._circuit = LCU(LCU_terms)
        
        # Extract the system block from the full LCU matrix
        m = len(LCU_terms)                             # number of ancilla states
        lcu_matrix = circuit.get_matrix()              # full (N*m) × (N*m) matrix
        U_approx = np.zeros_like(self.H, dtype=complex)
        for i in range(len(U_approx)):
            for j in range(len(U_approx)):
                # Take the top‑left block where ancilla is in |0⟩
                U_approx[i, j] = lcu_matrix[i*m, j*m]
        s = sum(alpha for _, alpha in LCU_terms)       # normalization factor (sum of weights)
        U_approx = U_approx * s                        # remove the 1/s factor from LCU output
        
        self._evolution_result = U_approx