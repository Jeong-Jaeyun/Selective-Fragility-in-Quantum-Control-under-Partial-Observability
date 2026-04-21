from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, amplitude_damping_error

from src.twobody.noise.common import SINGLE_QUBIT_BASIS_GATES, TWO_QUBIT_BASIS_GATES, _existing_basis_gates


def build_amplitude_damping_error(eta: float):
    if eta < 0.0 or eta > 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    return amplitude_damping_error(float(eta))


def apply_amplitude_damping_channel(
    qc: QuantumCircuit,
    eta: float,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    out = qc if inplace else qc.copy()
    if eta <= 0.0:
        return out
    instruction = build_amplitude_damping_error(eta).to_instruction()
    for qubit in range(out.num_qubits):
        out.append(instruction, [qubit])
    return out


def add_amplitude_noise(model: NoiseModel, eta: float) -> NoiseModel:
    if eta <= 0.0:
        return model
    error_1q = build_amplitude_damping_error(eta)
    error_2q = error_1q.tensor(error_1q)
    for gate in _existing_basis_gates(model, SINGLE_QUBIT_BASIS_GATES):
        model.add_all_qubit_quantum_error(error_1q, gate)
    for gate in _existing_basis_gates(model, TWO_QUBIT_BASIS_GATES):
        model.add_all_qubit_quantum_error(error_2q, gate)
    return model
