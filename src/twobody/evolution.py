from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from scipy.linalg import expm

from src.twobody.hamiltonian import build_pauli_hamiltonian, hamiltonian_matrix
from src.twobody.types import SystemConfig
from src.twobody.utils import coerce_system_config


def build_evolution_circuit(cfg: SystemConfig | dict[str, Any]) -> QuantumCircuit:
    system_cfg = coerce_system_config(cfg)
    qc = QuantumCircuit(system_cfg.n_qubits, name=f"evo:{system_cfg.hamiltonian_type}")
    if abs(system_cfg.evolution_time) == 0.0:
        return qc

    if system_cfg.evolution_method == "exact":
        unitary = expm(-1j * system_cfg.evolution_time * hamiltonian_matrix(system_cfg))
        qc.unitary(np.asarray(unitary, dtype=np.complex128), range(system_cfg.n_qubits), label="U_exact")
        return qc

    if system_cfg.evolution_method == "trotter":
        hamiltonian = build_pauli_hamiltonian(system_cfg)
        reps = max(int(system_cfg.trotter_steps), 1)
        gate = PauliEvolutionGate(
            hamiltonian,
            time=system_cfg.evolution_time,
            synthesis=LieTrotter(reps=reps),
        )
        qc.append(gate, range(system_cfg.n_qubits))
        return qc

    raise ValueError(f"unsupported evolution_method: {system_cfg.evolution_method}")


def compose_state_and_evolution(state_qc: QuantumCircuit, evo_qc: QuantumCircuit) -> QuantumCircuit:
    if state_qc.num_qubits != evo_qc.num_qubits:
        raise ValueError(
            f"qubit-count mismatch: state has {state_qc.num_qubits}, evolution has {evo_qc.num_qubits}"
        )

    qc = QuantumCircuit(state_qc.num_qubits, name="prepared_evolution")
    qc.compose(state_qc, inplace=True)
    qc.compose(evo_qc, inplace=True)
    return qc
