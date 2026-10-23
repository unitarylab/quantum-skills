from unitarylab_algorithms import FermiHubbardVQEAlgorithm

result = FermiHubbardVQEAlgorithm().run(
    L=2,           # 2-site open chain, 4 qubits
    t=1.0,         # hopping coefficient
    U=4.0,         # on-site interaction
    B=1.5,         # Zeeman field
    layers=5,      # Ry-Rz ring-entangling ansatz depth
    max_iter=1000, # COBYLA max iterations
    seed=7,
    measure_shots=10000,
    backend="torch",
    device="cpu",
)

print(result["VQE Energy"], result["Exact Energy"])
print(result["Absolute Error"], result["status"])

assert result["VQE Energy"] >= result["Exact Energy"] - 1e-8
