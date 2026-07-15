import numpy as np
from unitarylab_algorithms.state_preparation.Superposition.algorithm import SuperpositionAlgorithm

psi = np.array([1, 0, 1j, 0]) / np.sqrt(2)
r = SuperpositionAlgorithm().run(Psi=psi, target_qubits=2, target_error=1e-6)
print(r["Support size"], r["Index register qubits"], r["Total error"])

assert r["status"] == "ok"
assert r["Support size"] == 2
assert r["Total error"] <= 1e-6
