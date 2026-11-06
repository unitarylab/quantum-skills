"""Minimal example of the installed NEGF algorithm API."""

from unitarylab_algorithms.quantum_transport.negf import NEGFAlgorithm


def run_example():
    algorithm = NEGFAlgorithm(text_mode="plain", algo_dir="results/negf-example")
    result = algorithm.run(
        L=2,
        W=2,
        a=1.0,
        t=1.0,
        inverse_tolerance=2e-3,
    )

    print("Transmission:", result["transmission"])
    print("Injection residuals:", result["injection_residuals"])
    print("Circuit:", result["circuit_path"])
    print("QASM:", result["qasm_paths"])
    return result


if __name__ == "__main__":
    run_example()
