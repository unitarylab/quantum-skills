from unitarylab_algorithms.quantum_machine_learning.ising.algorithm import IsingAlgorithm

algorithm = IsingAlgorithm(text_mode="plain")
result = algorithm.run(
    rows=2,
    cols=2,
    coupling_J=1.0,
    field_h=3.0,
    time_step=0.025,
    trotter_steps=2,
    trotter_order=2,
    max_bond=32,
    cutoff=1e-12,
    routing="auto",
    seed=7,
    sample_shots=0,
    local_configuration_mode="gif",
)

print(result["status"])
print(result["final_observables"])
