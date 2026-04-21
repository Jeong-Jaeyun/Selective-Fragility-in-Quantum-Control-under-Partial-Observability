from __future__ import annotations

from typing import Any, Iterable

from qiskit import QuantumCircuit

from src.twobody.types import NoiseConfig


def apply_local_phase_drift(
    qc: QuantumCircuit,
    qubits: Iterable[int],
    phi: float,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    out = qc if inplace else qc.copy()
    if abs(phi) == 0.0:
        return out
    for qubit in qubits:
        out.rz(float(phi), int(qubit))
    return out


def apply_correlated_phase_drift(
    qc: QuantumCircuit,
    q0: int,
    q1: int,
    phi_corr: float,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    out = qc if inplace else qc.copy()
    if abs(phi_corr) == 0.0:
        return out
    out.rzz(float(phi_corr), int(q0), int(q1))
    return out


def apply_coherent_noise_block(
    qc: QuantumCircuit,
    noise_cfg: NoiseConfig | dict[str, Any] | None,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    out = qc if inplace else qc.copy()
    if noise_cfg is None:
        return out

    if isinstance(noise_cfg, NoiseConfig):
        phi = noise_cfg.phi
        phi_corr = noise_cfg.phi_corr
    else:
        phi = float(noise_cfg.get("phi", 0.0))
        phi_corr = float(noise_cfg.get("phi_corr", 0.0))

    out = apply_local_phase_drift(out, range(out.num_qubits), phi, inplace=True)
    if out.num_qubits >= 2:
        out = apply_correlated_phase_drift(out, 0, 1, phi_corr, inplace=True)
    return out
