from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.result import Counts
from qiskit_aer.noise import NoiseModel

from src.twobody.backend_factory import get_shot_backend
from src.twobody.observables import build_measurement_circuit_for_pauli, coerce_observable_specs
from src.twobody.types import ObservableSpec


def estimate_pauli_expectation_from_counts(counts: Mapping[str, int], pauli_str: str) -> float:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("counts must contain at least one shot")

    expectation = 0.0
    for bitstring, raw_count in counts.items():
        count = int(raw_count)
        bits_by_qubit = bitstring[::-1]
        sign = 1.0
        for qubit, pauli in enumerate(pauli_str):
            if pauli == "I":
                continue
            if bits_by_qubit[qubit] == "1":
                sign *= -1.0
        expectation += sign * count / total
    return float(expectation)


def estimate_expectations_from_counts(
    count_map: Mapping[str, Mapping[str, int]],
    observables: Sequence[ObservableSpec],
) -> dict[str, float]:
    expectations: dict[str, float] = {}
    for spec in observables:
        counts = count_map.get(spec.name)
        if counts is None:
            raise KeyError(f"missing counts for observable '{spec.name}'")
        expectations[spec.name] = estimate_pauli_expectation_from_counts(counts, spec.pauli)
    return expectations


def run_shot_experiment(
    qc: QuantumCircuit,
    observables: Sequence[ObservableSpec] | None = None,
    *,
    shots: int = 2048,
    seed: int | None = None,
    noise_model: NoiseModel | None = None,
    optimization_level: int = 0,
) -> dict[str, Any]:
    specs = coerce_observable_specs(observables, qc.num_qubits)
    measurement_circuits = [build_measurement_circuit_for_pauli(qc, spec.pauli) for spec in specs]
    backend = get_shot_backend(seed=seed, shots=shots, noise_model=noise_model)
    compiled = transpile(measurement_circuits, backend, optimization_level=optimization_level)
    result = backend.run(compiled, shots=int(shots)).result()

    count_map: dict[str, Counts] = {}
    for index, spec in enumerate(specs):
        count_map[spec.name] = result.get_counts(index)

    expectations = estimate_expectations_from_counts(count_map, specs)
    metadata = {
        "backend": "AerSimulator[shot]",
        "n_qubits": qc.num_qubits,
        "seed": seed,
        "shots": int(shots),
        "observable_count": len(specs),
        "noise_enabled": noise_model is not None,
    }
    return {"counts": count_map, "expectations": expectations, "metadata": metadata}
