"""CVQNN — Continuous Variable Quantum Neural Network for binary classification."""

import os
import sys
import numpy as np

# Allow importing CVQNNAlgorithm from the parent algorithm.py when run as a script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_algo_dir = os.path.dirname(_script_dir)
if _algo_dir not in sys.path:
    sys.path.insert(0, _algo_dir)

from unitarylab_algorithms import CVQNNAlgorithm


def main():
    try:
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=40, noise=0.1, random_state=42)
    except ImportError:
        X = np.array([[-1.02933991, -0.00298386], [-0.86996117,  0.54241283],
                      [-0.80316236,  0.61089337], [ 0.37814419, -0.41500468],
                      [ 0.8840248 ,  0.77229777], [-0.43851252,  0.95332979],
                      [ 1.82650142,  0.13439534], [ 0.55363043, -0.26334916],
                      [ 2.03229998,  0.36065831], [ 0.32911473,  0.73277684],
                      [ 0.3399878 , -0.20330403], [ 0.1571653 ,  0.81292617],
                      [-0.49050173,  0.83971655], [ 1.29280481, -0.40348121],
                      [ 1.91135621, -0.12862539], [ 0.25512714,  1.01131048],
                      [-1.04112002,  0.21991241], [ 0.02329181, -0.25089093],
                      [ 1.06363051, -0.09067207], [ 1.59455242, -0.20680035],
                      [ 0.69844027,  0.79542838], [-0.33511901,  0.95820148],
                      [ 1.13325543,  0.05220476], [ 1.01242119, -0.32393285],
                      [ 0.0594272 ,  0.16697667], [ 0.97850176,  0.31658757],
                      [ 1.12935882, -0.42297226], [ 1.80150356, -0.06033703],
                      [ 0.86442037,  0.37939163], [ 1.80418942, -0.12555484],
                      [ 0.38387907, -0.07845648], [ 1.32349064, -0.37337902],
                      [ 0.00469116,  1.22530709], [ 0.21647481,  0.25767384],
                      [-0.95705538,  0.25425763], [-0.0610322 ,  0.46838341],
                      [ 0.253453  ,  0.89288858], [-0.58101744,  0.71475467],
                      [ 0.67711022, -0.50537808], [ 2.05876963,  0.30982895]])
        y = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1])

    algo = CVQNNAlgorithm(text_mode="legacy")
    result = algo.run(
        x_train=X,
        y_train=y,
        n_layers=2,
        cutoff=6,
        epochs=40,
        lr=0.05,
    )

    print("=" * 50)
    print("CVQNN: Binary Classification (2 features, CV quantum)")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file     : {f['filename']} ({f['format']})")
    print(f"  Status          : {result['status']}")
    print(f"  Final Accuracy  : {result['Final Accuracy']:.2%}")
    print(f"  Final Loss      : {result['Final Loss']:.6f}")
    print(f"  Computation Time: {result['Total Computation Time (s)']:.2f}s")
    print(f"  Circuit path    : {result['circuit_path']}")


if __name__ == "__main__":
    main()
