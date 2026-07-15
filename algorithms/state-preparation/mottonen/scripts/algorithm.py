import numpy as np
from unitarylab_algorithms.state_preparation.mottonen.algorithm import MottonenAlgorithm

psi = np.array([1, 1j, 1, -1j], complex) / 2
r = MottonenAlgorithm().run(Psi=psi, target_qubits=2, target_error=1e-6)
print(r["status"], r["Total error"])

assert r["status"] == "ok"
assert r["Total error"] <= 1e-6
