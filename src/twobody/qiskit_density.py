from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix
from qiskit_aer.noise import NoiseModel

from src.twobody.backend_factory import get_density_backend
from src.twobody.observables import coerce_observable_specs, observable_to_pauli_op
from src.twobody.types import DensityExperimentResult, ObservableSpec


def get_density_matrix(
    qc: QuantumCircuit,
    seed: int | None = None,
    noise_model: NoiseModel | None = None,
) -> np.ndarray:
    backend = get_density_backend(seed=seed, noise_model=noise_model)
    sim_qc = qc.copy()
    sim_qc.save_density_matrix()
    compiled = transpile(sim_qc, backend)
    result = backend.run(compiled).result()
    raw = result.data(0)["density_matrix"]
    try:
        density = DensityMatrix(raw).data
    except Exception:
        density = np.asarray(raw, dtype=np.complex128)
    return np.asarray(density, dtype=np.complex128)


def estimate_expectations_from_density(
    density_matrix: np.ndarray,
    observables: Sequence[ObservableSpec] | None = None,
) -> dict[str, float]:
    density = DensityMatrix(density_matrix)
    specs = coerce_observable_specs(observables, density.num_qubits)
    expectations: dict[str, float] = {}
    for spec in specs:
        value = density.expectation_value(observable_to_pauli_op(spec))
        expectations[spec.name] = float(np.real_if_close(value))
    return expectations


def run_density_experiment(
    qc: QuantumCircuit,
    observables: Sequence[ObservableSpec] | None = None,
    seed: int | None = None,
    noise_model: NoiseModel | None = None,
) -> DensityExperimentResult:
    density = get_density_matrix(qc, seed=seed, noise_model=noise_model)
    expectations = estimate_expectations_from_density(density, observables=observables)
    metadata = {
        "backend": "AerSimulator[density_matrix]",
        "n_qubits": qc.num_qubits,
        "seed": seed,
        "observable_count": len(coerce_observable_specs(observables, qc.num_qubits)),
        "noise_enabled": noise_model is not None,
    }
    return DensityExperimentResult(density_matrix=density, expectations=expectations, metadata=metadata)
