from __future__ import annotations

from qiskit_aer.noise import NoiseModel, ReadoutError


def build_measurement_noise_model(p_meas: float, n_qubits: int) -> list[ReadoutError]:
    if p_meas < 0.0 or p_meas > 1.0:
        raise ValueError(f"p_meas must be in [0, 1], got {p_meas}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    matrix = [[1.0 - p_meas, p_meas], [p_meas, 1.0 - p_meas]]
    return [ReadoutError(matrix) for _ in range(n_qubits)]


def add_measurement_noise(model: NoiseModel, p_meas: float, n_qubits: int) -> NoiseModel:
    if p_meas <= 0.0:
        return model
    for qubit, error in enumerate(build_measurement_noise_model(p_meas, n_qubits)):
        model.add_readout_error(error, [qubit])
    return model
