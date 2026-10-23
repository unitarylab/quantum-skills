import numpy as np
from unitarylab_algorithms.state_preparation.pauli.algorithm import PauliAlgorithm

psi = np.array([1, 1j], complex) / np.sqrt(2)
r = PauliAlgorithm().run(Psi=psi, target_qubits=1, target_error=1e-6)
print(r["Pauli words"], r["Weights"], r["Total error"])

assert np.isfinite(r["Total error"])
