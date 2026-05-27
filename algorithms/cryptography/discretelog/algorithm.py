import time
import os
import math
import numpy as np
from fractions import Fraction
from typing import Dict, Any

# Import core project components
from unitarylab import Circuit, Register
from unitarylab.library import IQFT

try:
    from ...algo_base import BaseAlgorithm
except ImportError:
    # 单独运行时，将上级目录加入 sys.path，使 base 模块可被找到
    import sys
    _algorithms_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _algorithms_dir not in sys.path:
        sys.path.insert(0, _algorithms_dir)
    from algo_base import BaseAlgorithm

class DiscreteLogAlgorithm(BaseAlgorithm):
    """
    Discrete Logarithm Algorithm Module

    Computes x in g^x ≡ y (mod P).
    Based on quantum phase estimation and classical post-processing
    (continued fraction period extraction and congruence equation solving).
    """
    def __init__(self, text_mode: str = "plain", algo_dir: str = None):
        if algo_dir is None:
            _this = os.path.abspath(__file__)
            _directory = os.path.dirname(_this)
            algo_dir = os.path.join(os.getcwd(), "results", os.path.basename(os.path.dirname(_directory)), os.path.basename(_directory))
        os.makedirs(algo_dir, exist_ok=True)

        super().__init__(name="Discrete Logarithm Algorithm", prefix="DLG", text_mode=text_mode, algo_dir=algo_dir)

    def run(self, g: int, y: int, P: int, backend='torch', device='cpu', dtype=np.complex128) -> Dict[str, Any]:
        """
        Run the discrete logarithm algorithm to solve g^x ≡ y (mod P).

        Args:
            g: Base
            y: Target value
            P: Modulus (typically prime)

        Returns:
            Dictionary containing algorithm results with fields:
            - status: Execution status, 'ok' on success
            - circuit_path: Local path to saved quantum circuit diagram
            - file_path: Local path to saved result text file
        """
        input = {"Base number": g, "Target value": y, "Modulus": P}
        self.update_input(input)

        # Stage 1: Parameter preparation
        self.log(f"Stage 1: Parameter preparation")
        
        if math.gcd(g, P) != 1 or math.gcd(y, P) != 1:
            raise ValueError("g and y must be coprime with P")
            
        n_count = 2*P.bit_length()
        n_work = P.bit_length()
        N_size = 2**n_count
        
        self.log(f"  Count register size: {n_count}")
        self.log(f"  Work register size: {n_work}")
        self.log(f"  Total qubits: {2 * n_count + n_work}")

        # Stage 2: Quantum circuit construction
        self.log(f"Stage 2: Building quantum circuit")
        
        ra = Register("reg_a", n_count)
        rb = Register("reg_b", n_count)
        rw = Register("reg_work", n_work)
        gs = Circuit(ra, rb, rw, name=f'DLP_{g}^{{x}}_{y}')

        def get_p(reg_slice):
            reg, idxs = reg_slice[0]
            if reg.name == "reg_a": offset = 0
            elif reg.name == "reg_b": offset = n_count
            else: offset = 2 * n_count
            return [i + offset for i in idxs]

        gs.h(get_p(ra[:]))
        gs.h(get_p(rb[:]))
        gs.x(get_p(rw[0]))
        for i in range(n_count):
            mult = pow(g, 2**i, P)
            matrix = self._get_modular_matrix(mult, P, n_work)
            gs.unitary(matrix, get_p(rw[:]), get_p(ra[i])[0], '1')

        y_inv = pow(y, -1, P)
        for j in range(n_count):
            mult = pow(y_inv, 2**j, P)
            matrix = self._get_modular_matrix(mult, P, n_work)
            gs.unitary(matrix, get_p(rw[:]), get_p(rb[j])[0], '1')

        gs.append(IQFT(n_count), get_p(ra[:]))
        gs.append(IQFT(n_count), get_p(rb[:]))

        # Stage 3: Quantum simulation
        self.log(f"Stage 3: Running quantum simulation")
        
        start_time = time.time()
        res_vec = gs.execute(backend=backend, device=device, dtype=dtype)
        probs_dict = res_vec.calculate_state(range(2 * n_count))
        end_time = time.time()
        comp_time = end_time - start_time

        self.log(f"  Computation time: {comp_time:.4f} s")

        # Stage 4: Classical post-processing
        self.log(f"Stage 4: Classical post-processing (continued fractions and congruences)")
        
        found_x, found_r, info_msg = self._classical_post_processing(probs_dict, g, y, P, n_count, N_size)
        is_success = found_x is not None
        
        self.log(f"  Detected period r: {found_r if found_r else 'unknown'}")
        self.log(f"  Result x: {found_x if is_success else 'not found'}")
        self.log(f"  Verification: {'success' if is_success else 'failed'}")

        # Stage 5: Export circuit diagram
        self.log(f"Stage 5: Exporting circuit diagram")

        output = {"Computation time (s)": round(comp_time, 4), "Detected period r": found_r, "Found x": found_x}
        self.update_output(output)
        self.status = 'success' if is_success else 'failed'
        self.summary = f"Discrete logarithm x found with x={found_x}" if is_success else "Solution failed"

        # Save results
        circuit_path = self.save_circuit(gs)
        filename = self.save_txt()

        return self._build_return_dict(is_success, circuit_path, filename, gs)

    def _get_modular_matrix(self, a, N, n_qubits):
        """Build modular multiplication unitary: z -> (a * z) mod N."""
        dim = 2**n_qubits
        matrix = np.zeros((dim, dim))
        for z in range(dim):
            if z < N: target = (a * z) % N
            else: target = z
            matrix[target, z] = 1.0
        return matrix

    def _classical_post_processing(self, probs, g, y, P, n, N_size):
        """Classical post-processing: extract period via continued fractions and solve congruence equations."""
        sorted_probs = sorted(probs.items(), key=lambda x: x[1]['prob'], reverse=True)
        
        for bitstring, data in sorted_probs:
            if data['prob'] < 0.02: continue
            
            v_bin, u_bin = bitstring[:n], bitstring[n:]
            u, v = int(u_bin, 2), int(v_bin, 2)
            if u == 0: continue

            frac = Fraction(u, N_size).limit_denominator(P)
            r_base, s_base = frac.denominator, frac.numerator
            
            real_r = None
            for k in range(1, 10):
                if pow(g, r_base * k, P) == 1:
                    real_r = r_base * k
                    real_s = s_base * k
                    break
            
            if real_r is None: continue

            target = int(round((v * real_r) / N_size))
            
            d = math.gcd(real_s, real_r)
            if (-target) % d == 0:
                s_red = real_s // d
                r_red = real_r // d
                t_red = (-target) // d
                

                try:
                    x0 = (t_red * pow(s_red, -1, r_red)) % r_red
                    for i in range(d):
                        x_test = (x0 + i * r_red) % real_r
                        if pow(g, x_test, P) == (y % P):
                            final_x = x_test % real_r 
                            return final_x, real_r, "Success"
                except ValueError: 
                    continue
        return None, None, "No solution found"

def test(g=3, y=6, P=7):
    """
    Run the discrete logarithm algorithm to solve g^x ≡ y (mod P).

    Args:
        g: Base
        y: Target value
        P: Modulus (typically prime)

    Returns:
        Dictionary containing algorithm results with fields:
        - status: Execution status, 'ok' on success
        - circuit_path: Local path to saved quantum circuit diagram (SVG)
        - file_path: Local path to saved text file with results
    """
    if not isinstance(g, int):
        g = int(g)
    if not isinstance(y, int):
        y = int(y)
    if not isinstance(P, int):
        P = int(P)
    
    dlg = DiscreteLogAlgorithm(text_mode="legacy")
    result = dlg.run(g=g, y=y, P=P)
    return result


if __name__ == "__main__":
    g = 3 # [PARAM]
    y = 6 # [PARAM]
    P = 7 # [PARAM]

    test(g=g, y=y, P=P)
