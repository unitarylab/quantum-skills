# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional
from engine import GateSequence, Register
from core.pauli_string_decomposition import pauli_string_decomposition, pauli_string_evolution, pauli_string_to_matrix
from algorithms import HamiltonianSimulationResult
from scipy.linalg import expm
import itertools


class Trotter(HamiltonianSimulationResult):
    """
    Trotter-Suzuki decomposition for Hamiltonian simulation.

    This class implements the Trotter-Suzuki product formula to approximate
    the time evolution operator exp(-iHt). It supports first‑order and higher‑order
    even‑order expansions, with a fixed number of time steps (repetitions).

    Attributes:
        order : int
            Order of the Trotter-Suzuki expansion.
        steps : int
            Number of time slices (repetitions) used.
    """

    def __init__(self, H: np.ndarray, t: float, target_error: float, order: int = 1, steps: int = 1000) -> None:
        """
        Initialize the Trotter-Suzuki simulator.

        Parameters:
            H : numpy.ndarray
                Hamiltonian matrix (square Hermitian).
            t : float
                Total evolution time.
            target_error : float
                Desired approximation error (used for consistency with parent class;
                not actively used in the current implementation).
            order : int, optional
                Order of the Trotter-Suzuki expansion. Must be 1 or an even integer.
                Higher orders provide better accuracy but require more gates.
                Default is 1.
            steps : int, optional
                Number of time slices (repetitions) into which the total time is divided.
                Larger values increase accuracy but also circuit depth.
                Default is 1000.

        Raises:
            ValueError
                If order > 1 and order is odd.
        """
        super().__init__('trotter', H, t, target_error)
        if order > 1 and order % 2 == 1:
            raise ValueError("Order must be 1 or an even integer.")
        self.order = order
        self.alpha = np.linalg.norm(H, 2)
        self.steps = int(min(max(steps, 5**order * np.power(t * self.target_qubits * self.alpha, 1 + 1.0 / order) * np.power(target_error, -1.0 / order) * 1.5), 1e2))
        print('Simulation step:', self.steps)
        self._run()


    def _recurse(self, order: int, decomposition: List[Tuple[str, complex]]) -> List[Tuple[str, complex]]:
        """
        Recursively construct the higher‑order Trotter-Suzuki decomposition.
    
        This method implements the Suzuki recursive formula for even orders.
        For order 1 it returns the decomposition unchanged; for order 2 it applies
        the symmetric composition; for higher orders it recursively builds the
        nested product.
    
        Parameters:
            order : int
                Current order of the decomposition. Must be 1 or an even integer.
            decomposition : List[Tuple[str, complex]]
                A flat list of Pauli strings and their coefficients (the Hamiltonian
                decomposed into Pauli terms).
    
        Returns:
            List[Tuple[str, complex]]
                A list (with possibly modified coefficients) that, when multiplied
                in order, approximates the time evolution for the given order.
        """
        if order == 1:
            return decomposition
        elif order == 2:
            # Second‑order Suzuki: halves of all but the last term, then full last, then reversed halves.
            halves = [(p, c / 2) for p, c in decomposition[:-1]]
            full = [decomposition[-1]]
            return halves + full + list(reversed(halves))
        else:
            # Higher even orders: recursive composition.
            reduction = 1 / (4 - 4 ** (1 / (order - 1)))
            # Outer terms: apply recursion of order-2 with scaled coefficients.
            outer = 2 * self._recurse(order - 2, [(p, c * reduction) for p, c in decomposition])
            # Inner term: apply recursion with a different scaling.
            inner = self._recurse(order - 2, [(p, c * (1 - 4 * reduction)) for p, c in decomposition])
            return outer + inner + outer
    
    def _expand(self, decomposition: List[Tuple[str, complex]]) -> List[Tuple[str, complex]]:
        """
        Expand the Trotter-Suzuki sequence for a given total time.
    
        The method scales the coefficients by t / steps, builds a single time slice
        using recursion, and then repeats the slice `steps` times.
    
        Parameters:
            decomposition : List[Tuple[str, complex]]
                A flat list of Pauli strings and their coefficients (the Hamiltonian
                decomposed into Pauli terms).
        
        Returns:
            List[Tuple[str, complex]]
                The full Trotter sequence repeated `self.steps` times. Each element is a
                (pauli_string, coefficient) for one slice.
        """
        scaled_decomposition = [(p, c / self.steps) for p, c in decomposition]
        one_slice = self._recurse(self.order, scaled_decomposition)
        return one_slice

    def _run(self) -> None:
        """
        Build a quantum circuit implementing the Trotter-Suzuki decomposed evolution.

        This method overrides the abstract `_run` method of the parent class.
        It decomposes the Hamiltonian into Pauli strings, constructs the Trotter
        sequence, and appends the corresponding quantum gates to a circuit.
        The resulting circuit is stored in `self._circuit` for later access.

        Returns:
            None

        Notes:
            The exact evolution matrix (exp(-i H t)) and the approximate matrix
            obtained from the circuit are computed, and the spectral norm error
            is stored in `self._total_error`.
        """
        
        # Decompose Hamiltonian into Pauli terms (H is multiplied by t to incorporate time)
        decomposition = self._pauli_decompose(self.H, self.t)
        
        # Create a register and an empty circuit.
        reg = Register('K', self.target_qubits)
        qc = GateSequence(reg, name='Sub quantum circuit for Trotter Decomposition')

        # Get the full Trotter sequence.
        sequence = self._expand(decomposition)
        
        # Append each Pauli evolution gate to the circuit.
        for pauli_str, angle in sequence:
            gate = pauli_string_evolution(pauli_str, angle)
            qc.append(gate, range(self.target_qubits))
        
        self._circuit = qc.repeat(self.steps)
        
        U_approx = self._circuit.get_matrix()
        self._evolution_result = U_approx