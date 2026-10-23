"""Independent, source-aligned Boolean Hidden Shift implementation.

The module reconstructs the bent-function phase oracle, both measurement
paths, source bit ordering, seeded shift generation, result validation, and a
small command-line verification workflow without importing the packaged
``search.hidden_shift.algorithm`` module.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import random
from typing import Iterable

from unitarylab import Circuit, ClassicalRegister, Register


def bitstring(bits: Iterable[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def make_oracle_f(qubit_count: int) -> Circuit:
    if qubit_count <= 0 or qubit_count % 2:
        raise ValueError("qubit_count must be a positive even integer")
    oracle = Circuit(qubit_count, name="U_f")
    for qubit in range(0, qubit_count, 2):
        oracle.cz(qubit, qubit + 1)
    return oracle


def make_hs_circuit(shift: Iterable[int], *, measure: bool = False) -> Circuit:
    shift = tuple(shift)
    if not shift or len(shift) % 2 or any(bit not in (0, 1) for bit in shift):
        raise ValueError("shift must be a non-empty even-length binary sequence")
    qubit_count = len(shift)
    oracle_f = make_oracle_f(qubit_count)
    qubits = list(range(qubit_count))
    shifted_qubits = [index for index, value in enumerate(shift) if value]
    quantum = Register("q", qubit_count)
    classical = ClassicalRegister("result", qubit_count)
    circuit = Circuit(quantum, classical, name="Hidden Shift")
    circuit.h(qubits)
    if shifted_qubits:
        circuit.x(shifted_qubits)
    circuit.append(oracle_f, target=qubits)
    if shifted_qubits:
        circuit.x(shifted_qubits)
    circuit.h(qubits)
    circuit.append(oracle_f, target=qubits)
    circuit.h(qubits)
    if measure:
        circuit.measure(quantum[:], classical[:])
    return circuit


def has_measurement(circuit: Circuit) -> bool:
    """Read measurement state across current and older UnitaryLab releases."""
    public_value = getattr(circuit, "has_measurement", None)
    if public_value is not None:
        return bool(public_value)
    return bool(getattr(circuit, "_measure_map", ()))


def sample_circuit(
    circuit: Circuit,
    shots: int = 100,
    *,
    backend: str = "numpy",
    seed: int = 42,
) -> Counter:
    if has_measurement(circuit):
        raise ValueError("sample_circuit requires a circuit without measurements")
    result = circuit.execute(backend=backend, device="cpu", seed=seed)
    return Counter(outcome[::-1] for outcome in result.sample(shots=shots, seed=seed))


@dataclass
class HiddenShiftRun:
    status: str
    secret_shift: str
    most_common_bitstring: str
    counts: dict[str, int]
    matched: bool
    circuit: Circuit
    backend: str
    measurement_mode: str
    shots: int
    seed: int


def resolve_shift(qubits: int, shift: str, seed: int) -> tuple[int, list[int]]:
    shift = "" if shift is None else str(shift).strip()
    if shift:
        if any(value not in "01" for value in shift):
            raise ValueError("shift must be a binary string")
        shift_bits = [int(value) for value in shift]
        qubits = len(shift_bits)
    else:
        rng = random.Random(seed)
        shift_bits = [rng.randint(0, 1) for _ in range(qubits)]
    if qubits <= 0 or qubits % 2:
        raise ValueError("the number of qubits must be positive and even")
    return qubits, shift_bits


def run_hidden_shift(
    qubits: int = 6,
    shift: str = "",
    shots: int = 100,
    seed: int = 41,
    backend: str = "numpy",
    measurement_mode: str = "sample",
) -> HiddenShiftRun:
    qubits, shots, seed = int(qubits), int(shots), int(seed)
    backend, measurement_mode = str(backend), str(measurement_mode)
    if shots <= 0 or seed < 0:
        raise ValueError("shots must be positive and seed must be non-negative")
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    if measurement_mode not in ("sample", "execute"):
        raise ValueError("measurement_mode must be 'sample' or 'execute'")
    qubits, shift_bits = resolve_shift(qubits, shift, seed)
    circuit = make_hs_circuit(shift_bits, measure=measurement_mode == "execute")
    if measurement_mode == "sample":
        frequencies = sample_circuit(
            circuit, shots, backend=backend, seed=seed
        )
    else:
        result = circuit.execute(
            backend=backend,
            device="cpu",
            shots=shots,
            seed=seed,
        )
        frequencies = Counter(
            {outcome[::-1]: count for outcome, count in result.counts.items()}
        )
    answer = frequencies.most_common(1)[0][0]
    expected = bitstring(shift_bits)
    matched = answer == expected
    return HiddenShiftRun(
        status="ok" if matched else "failed",
        secret_shift=expected,
        most_common_bitstring=answer,
        counts=dict(frequencies),
        matched=matched,
        circuit=circuit,
        backend=backend,
        measurement_mode=measurement_mode,
        shots=shots,
        seed=seed,
    )


def build_diagnostic_report() -> dict[str, object]:
    """Validate circuit construction and return observable structural evidence."""
    assert bitstring([1, 0, 1, 1, 0, 0]) == "101100"
    circuit_cases = []
    for shift in ("00", "01", "10", "11"):
        bits = [int(bit) for bit in shift]
        sample_circuit_case = make_hs_circuit(bits)
        execute_circuit_case = make_hs_circuit(bits, measure=True)
        assert not has_measurement(sample_circuit_case)
        assert has_measurement(execute_circuit_case)
        shifted_qubits = [index for index, bit in enumerate(bits) if bit]
        circuit_cases.append(
            {
                "shift": shift,
                "qubits": len(bits),
                "shifted_qubits": shifted_qubits,
                "hadamard_gates": 3 * len(bits),
                "cz_gates": len(bits),
                "x_gates": 2 * len(shifted_qubits),
                "sample_has_measurement": has_measurement(sample_circuit_case),
                "execute_has_measurement": has_measurement(execute_circuit_case),
                "execute_measurement_bits": len(bits),
            }
        )
    first = resolve_shift(6, "", 41)
    second = resolve_shift(6, "", 41)
    assert first == second
    assert resolve_shift(2, "01", 999) == (2, [0, 1])
    rejected_inputs = []
    for invalid in ("1", "010", "0x"):
        try:
            resolve_shift(2, invalid, 0)
        except ValueError as error:
            rejected_inputs.append(
                {"input": invalid, "error": str(error)}
            )
        else:
            raise AssertionError(f"invalid shift accepted: {invalid!r}")
    return {
        "oracle_function": "f(x) = XOR_i x[2i]x[2i+1]",
        "circuit_sequence": "H -> X_s U_f X_s -> H -> U_f -> H",
        "source_bit_order": "q0, q1, ...",
        "backend_normalization": "outcome[::-1]",
        "circuit_cases": circuit_cases,
        "seeded_qubits": first[0],
        "seed": 41,
        "seeded_shift": bitstring(first[1]),
        "explicit_shift_override": bitstring(resolve_shift(2, "01", 999)[1]),
        "rejected_inputs": rejected_inputs,
    }


def run_execution_checks(backend: str = "numpy") -> list[HiddenShiftRun]:
    results = []
    for shift in ("00", "01", "10", "11"):
        for mode in ("sample", "execute"):
            result = run_hidden_shift(
                shift=shift,
                shots=32,
                seed=41,
                backend=backend,
                measurement_mode=mode,
            )
            assert sum(result.counts.values()) == 32
            assert all(len(outcome) == len(shift) for outcome in result.counts)
            assert result.matched
            assert result.most_common_bitstring == shift
            results.append(result)
    return results


def print_diagnostic_report(report: dict[str, object]) -> None:
    """Print oracle, circuit, bit-order, seed, and validation evidence."""
    print("Hidden Shift structural report")
    print(f"  oracle={report['oracle_function']}")
    print(f"  circuit={report['circuit_sequence']}")
    print(f"  source_bit_order={report['source_bit_order']}")
    print(f"  backend_normalization={report['backend_normalization']}")
    print("  two-qubit circuit cases:")
    for case in report["circuit_cases"]:
        print(
            f"    shift={case['shift']}: shifted_qubits={case['shifted_qubits']}, "
            f"H={case['hadamard_gates']}, CZ={case['cz_gates']}, "
            f"X={case['x_gates']}"
        )
        print(
            "      sample_has_measurement="
            f"{case['sample_has_measurement']}, "
            "execute_has_measurement="
            f"{case['execute_has_measurement']}, "
            "execute_measurement_bits="
            f"{case['execute_measurement_bits']}"
        )
    print("  shift resolution:")
    print(
        f"    seeded_random: qubits={report['seeded_qubits']}, "
        f"seed={report['seed']}, shift={report['seeded_shift']}"
    )
    print(
        "    explicit_shift_overrides_qubits="
        f"{report['explicit_shift_override']}"
    )
    print("  rejected inputs:")
    for item in report["rejected_inputs"]:
        print(f"    {item['input']!r}: {item['error']}")


def print_execution_report(results: list[HiddenShiftRun]) -> None:
    """Print recovered strings and complete normalized counts for every case."""
    print("Hidden Shift execution report")
    for result in results:
        print(
            f"  shift={result.secret_shift}, mode={result.measurement_mode}, "
            f"backend={result.backend}, shots={result.shots}, seed={result.seed}"
        )
        print(
            f"    recovered={result.most_common_bitstring}, "
            f"matched={result.matched}, status={result.status}"
        )
        print(
            f"    counts={result.counts}, "
            f"count_total={sum(result.counts.values())}, "
            f"circuit_has_measurement={has_measurement(result.circuit)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    args = parser.parse_args()
    diagnostic = build_diagnostic_report()
    print_diagnostic_report(diagnostic)
    if args.execute:
        results = run_execution_checks(args.backend)
        print_execution_report(results)


if __name__ == "__main__":
    main()
