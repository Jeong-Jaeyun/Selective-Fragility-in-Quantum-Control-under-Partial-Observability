from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from src.twobody.noise.amplitude import add_amplitude_noise, apply_amplitude_damping_channel
from src.twobody.noise.dephasing import add_dephasing_noise, apply_dephasing_channel
from src.twobody.noise.depolarizing import add_depolarizing_noise, apply_depolarizing_channel
from src.twobody.noise.measurement import add_measurement_noise
from src.twobody.types import NoiseConfig


def _coerce_noise_cfg(noise_cfg: NoiseConfig | dict[str, Any] | None) -> NoiseConfig:
    if noise_cfg is None:
        return NoiseConfig()
    if isinstance(noise_cfg, NoiseConfig):
        return noise_cfg
    return NoiseConfig(**noise_cfg)


def build_noise_model(noise_cfg: NoiseConfig | dict[str, Any] | None, n_qubits: int) -> NoiseModel | None:
    cfg = _coerce_noise_cfg(noise_cfg)
    if (
        cfg.gamma_dephasing <= 0.0
        and cfg.eta_amplitude <= 0.0
        and cfg.p_depolarizing <= 0.0
        and cfg.p_measurement <= 0.0
    ):
        return None

    model = NoiseModel()
    add_dephasing_noise(model, cfg.gamma_dephasing)
    add_amplitude_noise(model, cfg.eta_amplitude)
    add_depolarizing_noise(model, cfg.p_depolarizing)
    add_measurement_noise(model, cfg.p_measurement, n_qubits=n_qubits)
    return model


def apply_stochastic_noise_block(
    qc: QuantumCircuit,
    noise_cfg: NoiseConfig | dict[str, Any] | None,
    *,
    inplace: bool = False,
) -> QuantumCircuit:
    cfg = _coerce_noise_cfg(noise_cfg)
    out = qc if inplace else qc.copy()
    out = apply_dephasing_channel(out, cfg.gamma_dephasing, inplace=True)
    out = apply_amplitude_damping_channel(out, cfg.eta_amplitude, inplace=True)
    out = apply_depolarizing_channel(out, cfg.p_depolarizing, inplace=True)
    return out


def describe_noise_config(noise_cfg: NoiseConfig | dict[str, Any] | None) -> dict[str, float]:
    cfg = _coerce_noise_cfg(noise_cfg)
    return {
        "phi": float(cfg.phi),
        "phi_corr": float(cfg.phi_corr),
        "gamma_dephasing": float(cfg.gamma_dephasing),
        "eta_amplitude": float(cfg.eta_amplitude),
        "p_depolarizing": float(cfg.p_depolarizing),
        "p_measurement": float(cfg.p_measurement),
    }
