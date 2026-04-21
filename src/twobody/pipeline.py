from __future__ import annotations

from typing import Any

from src.twobody.evolution import build_evolution_circuit, compose_state_and_evolution
from src.twobody.features import extract_features
from src.twobody.latent import estimate_latent
from src.twobody.noise import apply_local_phase_drift, apply_stochastic_noise_block, build_noise_model
from src.twobody.observables import get_observable_specs
from src.twobody.qiskit_density import run_density_experiment
from src.twobody.qiskit_shot import run_shot_experiment
from src.twobody.reconstruction import compensate_expectations
from src.twobody.states import build_state_circuit
from src.twobody.types import NoiseConfig, SystemConfig


def build_base_circuit(
    *,
    state_family: str,
    system_cfg: SystemConfig,
    seed: int,
):
    return compose_state_and_evolution(
        build_state_circuit(state_family, system_cfg.n_qubits, seed=int(seed)),
        build_evolution_circuit(system_cfg),
    )


def inject_noise(base_qc, noise_cfg: NoiseConfig):
    with_phase = apply_local_phase_drift(base_qc, [0], float(noise_cfg.phi))
    return apply_stochastic_noise_block(with_phase, noise_cfg)


def measure_expectations(
    qc,
    *,
    backend_type: str,
    shots: int,
    seed: int,
    system_cfg: SystemConfig,
    p_measurement: float,
    include_cross_terms: bool = True,
) -> dict[str, float]:
    observables = get_observable_specs(system_cfg.n_qubits, include_cross_terms=include_cross_terms)
    readout_noise = build_noise_model(
        NoiseConfig(p_measurement=float(p_measurement)),
        n_qubits=system_cfg.n_qubits,
    )
    if backend_type == "density":
        result = run_density_experiment(qc, observables=observables, seed=int(seed), noise_model=readout_noise)
        return result.expectations
    if backend_type == "shot":
        result = run_shot_experiment(
            qc,
            observables=observables,
            shots=int(shots),
            seed=int(seed),
            noise_model=readout_noise,
        )
        return result["expectations"]
    raise ValueError(f"unsupported backend_type: {backend_type}")


def run_latent_pipeline(
    *,
    probe_state_family: str,
    system_cfg: SystemConfig,
    noise_cfg: NoiseConfig,
    backend_type: str,
    shots: int,
    seed: int,
) -> dict[str, Any]:
    base_qc = build_base_circuit(state_family=probe_state_family, system_cfg=system_cfg, seed=int(seed))
    noisy_qc = inject_noise(base_qc, noise_cfg)
    expectations = measure_expectations(
        noisy_qc,
        backend_type=backend_type,
        shots=int(shots),
        seed=int(seed),
        system_cfg=system_cfg,
        p_measurement=float(noise_cfg.p_measurement),
        include_cross_terms=True,
    )
    latent = estimate_latent(expectations)
    return {
        "base_circuit": base_qc,
        "noisy_circuit": noisy_qc,
        "probe_expectations": expectations,
        "latent": latent,
    }


def run_feature_pipeline(
    *,
    target_state_family: str,
    system_cfg: SystemConfig,
    noise_cfg: NoiseConfig,
    backend_type: str,
    shots: int,
    seed: int,
    latent: dict[str, float] | None = None,
    include_full_oracle: bool = False,
) -> dict[str, Any]:
    observables = get_observable_specs(system_cfg.n_qubits, include_cross_terms=True)
    base_qc = build_base_circuit(state_family=target_state_family, system_cfg=system_cfg, seed=int(seed))
    ideal_expectations: dict[str, float] | None = None
    ideal_features: dict[str, float] | None = None
    full_oracle_expectations: dict[str, float] | None = None
    full_oracle_features: dict[str, float] | None = None
    if include_full_oracle:
        ideal_expectations = run_density_experiment(
            base_qc,
            observables=observables,
            seed=int(seed),
            noise_model=None,
        ).expectations
        ideal_features = extract_features(ideal_expectations, latent={"phi_hat": 0.0, "gamma_hat": 0.0})
        full_oracle_expectations = ideal_expectations
        full_oracle_features = ideal_features
    clean_expectations = measure_expectations(
        base_qc,
        backend_type=backend_type,
        shots=int(shots),
        seed=int(seed),
        system_cfg=system_cfg,
        p_measurement=0.0,
        include_cross_terms=True,
    )
    noisy_qc = inject_noise(base_qc, noise_cfg)
    noisy_expectations = measure_expectations(
        noisy_qc,
        backend_type=backend_type,
        shots=int(shots),
        seed=int(seed),
        system_cfg=system_cfg,
        p_measurement=float(noise_cfg.p_measurement),
        include_cross_terms=True,
    )
    latent_estimate = latent if latent is not None else estimate_latent(noisy_expectations)
    compensated_expectations = compensate_expectations(
        noisy_expectations,
        observables,
        phi=float(latent_estimate["phi_hat"]),
        gamma=float(latent_estimate["gamma_hat"]),
    )
    oracle_expectations = compensate_expectations(
        noisy_expectations,
        observables,
        phi=float(noise_cfg.phi),
        gamma=float(noise_cfg.gamma_dephasing),
    )
    return {
        "observables": observables,
        "base_circuit": base_qc,
        "ideal_expectations": ideal_expectations,
        "clean_expectations": clean_expectations,
        "noisy_expectations": noisy_expectations,
        "compensated_expectations": compensated_expectations,
        "oracle_expectations": oracle_expectations,
        "full_oracle_expectations": ideal_expectations,
        "clean_features": extract_features(clean_expectations, latent=None),
        "ideal_features": ideal_features,
        "none_features": extract_features(noisy_expectations, latent=latent_estimate),
        "compensated_features": extract_features(compensated_expectations, latent=latent_estimate),
        "oracle_features": extract_features(oracle_expectations, latent={"phi_hat": noise_cfg.phi, "gamma_hat": noise_cfg.gamma_dephasing}),
        "full_oracle_features": full_oracle_features,
        "latent": latent_estimate,
    }
