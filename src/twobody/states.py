from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from src.twobody.utils import rng_from_seed


def available_state_families() -> tuple[str, ...]:
    return ("zero", "rotated_product", "bell", "bell_i", "random_low_depth")


def build_state_circuit(state_family: str, n_qubits: int, seed: int | None = None) -> QuantumCircuit:
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")

    qc = QuantumCircuit(n_qubits, name=f"state:{state_family}")
    if state_family == "zero":
        return qc

    if state_family == "rotated_product":
        for qubit in range(n_qubits):
            angle_y = np.pi / (5.0 + qubit)
            angle_z = np.pi / (7.0 + 2.0 * qubit)
            qc.ry(angle_y, qubit)
            qc.rz(angle_z, qubit)
        return qc

    if state_family == "bell":
        if n_qubits < 2:
            raise ValueError("bell state requires at least 2 qubits")
        qc.h(0)
        qc.cx(0, 1)
        return qc

    if state_family == "bell_i":
        if n_qubits < 2:
            raise ValueError("bell_i state requires at least 2 qubits")
        qc.h(0)
        qc.cx(0, 1)
        qc.s(0)
        return qc

    if state_family == "random_low_depth":
        rng = rng_from_seed(seed)
        for qubit in range(n_qubits):
            qc.ry(float(rng.uniform(-np.pi, np.pi)), qubit)
            qc.rz(float(rng.uniform(-np.pi, np.pi)), qubit)
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)
        for qubit in range(n_qubits):
            qc.rx(float(rng.uniform(-np.pi, np.pi)), qubit)
            qc.rz(float(rng.uniform(-np.pi, np.pi)), qubit)
        return qc

    raise ValueError(f"unsupported state family: {state_family}")
