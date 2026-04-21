from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, phase_damping_error

from src.twobody.noise.common import SINGLE_QUBIT_BASIS_GATES, TWO_QUBIT_BASIS_GATES, _existing_basis_gates


def build_dephasing_error(gamma: float):
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    return phase_damping_error(float(gamma))


def apply_dephasing_channel(
    qc: QuantumCircuit,
    gamma: float,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    out = qc if inplace else qc.copy()
    if gamma <= 0.0:
        return out
    instruction = build_dephasing_error(gamma).to_instruction()
    for qubit in range(out.num_qubits):
        out.append(instruction, [qubit])
    return out


def add_dephasing_noise(model: NoiseModel, gamma: float) -> NoiseModel:
    if gamma <= 0.0:
        return model
    error_1q = build_dephasing_error(gamma)
    error_2q = error_1q.tensor(error_1q)
    for gate in _existing_basis_gates(model, SINGLE_QUBIT_BASIS_GATES):
        model.add_all_qubit_quantum_error(error_1q, gate)
    for gate in _existing_basis_gates(model, TWO_QUBIT_BASIS_GATES):
        model.add_all_qubit_quantum_error(error_2q, gate)
    return model
