#!/bin/bash
set -e

run() {
    local script="$1"
    local name="$2"
    echo ""
    echo "[RUN] $name"
    python "$script"
    echo "[OK] $name"
}

run "algorithms/cryptography/discretelog/scripts/algorithm.py" "cryptography/discretelog"
run "algorithms/cryptography/shor/scripts/algorithm.py" "cryptography/shor"
run "algorithms/cryptography/simon/scripts/algorithm.py" "cryptography/simon"
run "algorithms/primitives/amplitude-amplification/scripts/algorithm.py" "primitives/amplitude-amplification"
run "algorithms/primitives/amplitude-estimation/scripts/algorithm.py" "primitives/amplitude-estimation"
run "algorithms/primitives/hadamard-test/scripts/algorithm.py" "primitives/hadamard-test"
run "algorithms/primitives/hadamard-transform/scripts/algorithm.py" "primitives/hadamard-transform"
run "algorithms/primitives/quantum-phase-estimation/scripts/algorithm.py" "primitives/quantum-phase-estimation"
run "algorithms/hamiltonian-simulation/cartan/scripts/algorithm.py" "hamiltonian-simulation/cartan"
run "algorithms/hamiltonian-simulation/qdrift/scripts/algorithm.py" "hamiltonian-simulation/qdrift"
run "algorithms/hamiltonian-simulation/qsp/scripts/algorithm.py" "hamiltonian-simulation/qsp"
run "algorithms/hamiltonian-simulation/taylor/scripts/algorithm.py" "hamiltonian-simulation/taylor"
run "algorithms/hamiltonian-simulation/trotter/scripts/algorithm.py" "hamiltonian-simulation/trotter"
# run "algorithms/linear-systems/hhl/scripts/algorithm.py" "linear-systems/hhl"
# run "algorithms/linear-systems/lcu/scripts/algorithm.py" "linear-systems/lcu"
# run "algorithms/linear-systems/quantum-signal-processing/scripts/algorithm.py" "linear-systems/quantum-signal-processing"
# run "algorithms/quantum-machine-learning/qaoa/scripts/algorithm.py" "quantum-machine-learning/qaoa"
# run "algorithms/quantum-machine-learning/qcbm/scripts/algorithm.py" "quantum-machine-learning/qcbm"
# run "algorithms/quantum-machine-learning/qnn/scripts/algorithm.py" "quantum-machine-learning/qnn"
# run "algorithms/quantum-machine-learning/vqc/scripts/algorithm.py" "quantum-machine-learning/vqc"
# run "algorithms/quantum-machine-learning/vqe/scripts/algorithm.py" "quantum-machine-learning/vqe"
run "algorithms/schrodingerization/advection-schrodingerization/scripts/algorithm.py" "schrodingerization/advection-schrodingerization"

run "algorithms/schrodingerization/heat-1d-schrodingerization/scripts/algorithm.py" "schrodingerization/heat-1d-schrodingerization"
# run "algorithms/schrodingerization/heat-2d-schrodingerization/scripts/algorithm.py" "schrodingerization/heat-2d-schrodingerization"

echo ""
echo "[OK] All algorithms completed successfully."
