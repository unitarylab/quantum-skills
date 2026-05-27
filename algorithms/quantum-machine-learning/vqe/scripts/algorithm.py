"""VQE — Variational Quantum Eigensolver for arbitrary Hermitian Hamiltonians."""

from unitarylab_algorithms import VQEAlgorithm


def main():
    algo = VQEAlgorithm()
    result = algo.run(
        n=2,
        layers=2,
        max_iter=150,
        seed=7,
        backend="torch",
    )

    print("=" * 50)
    print("VQE: Ground-State Energy Estimation")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file    : {f['filename']} ({f['format']})")
    print(f"  Status         : {result['status']}")
    print(f"  VQE energy     : {result['VQE Energy']:.4f}")
    print(f"  Exact energy   : {result['Exact Energy']:.4f}")
    print(f"  Absolute error : {result['Absolute Error']:.2e}")
    print(f"  Optimizer msg  : {result.get('Optimizer Message')}")
    print(f"  Quantum time   : {result.get('Quantum Comp Time'):.3f}s")
    print(f"  Circuit path   : {result.get('circuit_path')}")


if __name__ == "__main__":
    main()
