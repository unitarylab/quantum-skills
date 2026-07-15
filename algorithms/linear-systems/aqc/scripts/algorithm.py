import numpy as np
from unitarylab_algorithms import AQCAlgorithm

algo = AQCAlgorithm(text_mode="plain")
result = algo.run(
    n=2,       # 2 system qubits, N=4
    T=0,       # automatically choose an even step count
    p=1.4,
    backend="torch",
    device="cpu",
)

print(result["status"])
print(result["Quantum Solution (x)"])
print(result["Residual Norm ||Ax-b||"])
print(result["Error vs Classical (L2)"])
print(result["circuit_path"])

assert result["status"] == "ok"
assert np.isfinite(result["Residual Norm ||Ax-b||"])
