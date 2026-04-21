from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from src.twobody.types import SystemConfig
from src.twobody.utils import coerce_system_config


def _pauli_label(n_qubits: int, axes: dict[int, str]) -> str:
    label = ["I"] * n_qubits
    for qubit, axis in axes.items():
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"qubit index {qubit} is outside n_qubits={n_qubits}")
        label[int(qubit)] = str(axis)
    return "".join(label)


def _nearest_neighbor_terms(n_qubits: int, pair_axis: str, coeff: float) -> list[tuple[str, float]]:
    if abs(coeff) == 0.0:
        return []
    return [
        (_pauli_label(n_qubits, {qubit: pair_axis[0], qubit + 1: pair_axis[1]}), float(coeff))
        for qubit in range(n_qubits - 1)
    ]


def _local_field_terms(n_qubits: int, axis: str, coeff: float) -> list[tuple[str, float]]:
    if abs(coeff) == 0.0:
        return []
    return [(_pauli_label(n_qubits, {qubit: axis}), float(coeff)) for qubit in range(n_qubits)]


def build_pauli_hamiltonian(cfg: SystemConfig | dict[str, Any]) -> SparsePauliOp:
    system_cfg = coerce_system_config(cfg)
    if system_cfg.n_qubits < 2:
        raise ValueError(f"two-body Hamiltonian requires n_qubits >= 2, got {system_cfg.n_qubits}")

    if system_cfg.hamiltonian_type == "xx_zz":
        terms = [
            *_nearest_neighbor_terms(system_cfg.n_qubits, "XX", system_cfg.jx),
            *_nearest_neighbor_terms(system_cfg.n_qubits, "ZZ", system_cfg.jz),
            *_local_field_terms(system_cfg.n_qubits, "Z", system_cfg.hz),
        ]
    elif system_cfg.hamiltonian_type == "xy":
        terms = [
            *_nearest_neighbor_terms(system_cfg.n_qubits, "XX", system_cfg.jx),
            *_nearest_neighbor_terms(system_cfg.n_qubits, "YY", system_cfg.jy),
            *_local_field_terms(system_cfg.n_qubits, "Z", system_cfg.hz),
        ]
    elif system_cfg.hamiltonian_type == "ising_x":
        terms = [
            *_nearest_neighbor_terms(system_cfg.n_qubits, "ZZ", system_cfg.jz),
            *_local_field_terms(system_cfg.n_qubits, "X", system_cfg.hx),
        ]
    else:
        raise ValueError(f"unsupported hamiltonian_type: {system_cfg.hamiltonian_type}")

    nonzero_terms = [(label, coeff) for label, coeff in terms if abs(coeff) > 0.0]
    if not nonzero_terms:
        nonzero_terms = [("I" * system_cfg.n_qubits, 0.0)]
    return SparsePauliOp.from_list(nonzero_terms).simplify()


def describe_hamiltonian(cfg: SystemConfig | dict[str, Any]) -> dict[str, float | int | str]:
    system_cfg = coerce_system_config(cfg)
    hamiltonian = build_pauli_hamiltonian(system_cfg)
    return {
        "n_qubits": system_cfg.n_qubits,
        "hamiltonian_type": system_cfg.hamiltonian_type,
        "evolution_method": system_cfg.evolution_method,
        "evolution_time": float(system_cfg.evolution_time),
        "trotter_steps": int(system_cfg.trotter_steps),
        "n_terms": len(hamiltonian.paulis),
    }


def hamiltonian_matrix(cfg: SystemConfig | dict[str, Any]) -> np.ndarray:
    return np.asarray(build_pauli_hamiltonian(cfg).to_matrix(), dtype=np.complex128)
