from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, depolarizing_error

from src.twobody.noise.common import SINGLE_QUBIT_BASIS_GATES, TWO_QUBIT_BASIS_GATES, _existing_basis_gates


def build_depolarizing_error(probability: float, n_qubits: int):
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if n_qubits not in {1, 2}:
        raise ValueError(f"n_qubits must be 1 or 2, got {n_qubits}")
    return depolarizing_error(float(probability), n_qubits)


def apply_depolarizing_channel(
    qc: QuantumCircuit,
    probability: float,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    out = qc if inplace else qc.copy()
    if probability <= 0.0:
        return out
    instruction = build_depolarizing_error(probability, 1).to_instruction()
    for qubit in range(out.num_qubits):
        out.append(instruction, [qubit])
    return out


def add_depolarizing_noise(model: NoiseModel, probability: float) -> NoiseModel:
    if probability <= 0.0:
        return model
    error_1q = build_depolarizing_error(probability, 1)
    error_2q = build_depolarizing_error(probability, 2)
    for gate in _existing_basis_gates(model, SINGLE_QUBIT_BASIS_GATES):
        model.add_all_qubit_quantum_error(error_1q, gate)
    for gate in _existing_basis_gates(model, TWO_QUBIT_BASIS_GATES):
        model.add_all_qubit_quantum_error(error_2q, gate)
    return model
