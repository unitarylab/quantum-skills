import numpy as np
from unitarylab_algorithms.state_preparation.mps.algorithm import MPSAlgorithm

psi = np.ones(8, complex) / np.sqrt(8)
r = MPSAlgorithm().run(Psi=psi, target_qubits=3, target_error=1e-6,
                       mps_max_bond_dim=2, rng_seed=42)
print(r["MPS tensors"], r["Work leakage"], r["Total error"])

assert np.isfinite(r["Total error"])
assert r["Work leakage"] >= 0
