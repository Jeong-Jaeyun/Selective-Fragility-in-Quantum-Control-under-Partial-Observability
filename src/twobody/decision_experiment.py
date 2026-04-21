from __future__ import annotations

from typing import Any

import numpy as np

from src.twobody.decision import (
    decision_predictions,
    decision_scores,
    fit_linear_classifier,
    fit_logistic_regression,
    fit_threshold_classifier,
)
from src.twobody.pipeline import run_feature_pipeline, run_latent_pipeline
from src.twobody.types import NoiseConfig, SystemConfig


def _feature_vector(feature_dict: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.asarray([float(feature_dict[name]) for name in feature_names], dtype=float)


def _fit_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    threshold_feature: str,
    classifiers: list[str],
):
    models = {}
    for classifier in classifiers:
        if classifier == "threshold":
            models[classifier] = fit_threshold_classifier(x_train, y_train, feature_names, threshold_feature)
        elif classifier == "linear":
            models[classifier] = fit_linear_classifier(x_train, y_train, feature_names)
        elif classifier == "logistic":
            models[classifier] = fit_logistic_regression(x_train, y_train, feature_names)
        else:
            raise ValueError(f"unsupported classifier: {classifier}")
    return models


def merge_system_config(base: SystemConfig, overrides: dict[str, Any] | None = None) -> SystemConfig:
    data = base.to_dict()
    if overrides:
        data.update(overrides)
    return SystemConfig(**data)


def merge_noise_config(base: NoiseConfig, overrides: dict[str, Any] | None = None) -> NoiseConfig:
    data = base.to_dict()
    if overrides:
        data.update(overrides)
    return NoiseConfig(**data)


def run_decision_condition(
    *,
    train_seeds: list[int],
    test_seeds: list[int],
    backend_types: list[str],
    shot_list: list[int],
    probe_state_family: str,
    target_state_families: list[str],
    feature_names: list[str],
    threshold_feature: str,
    classifiers: list[str],
    phi_values: list[float],
    gamma_values: list[float],
    base_system_cfg: SystemConfig,
    train_system_overrides: dict[str, Any] | None = None,
    test_system_overrides: dict[str, Any] | None = None,
    train_noise_overrides: dict[str, Any] | None = None,
    test_noise_overrides: dict[str, Any] | None = None,
    train_state_families: list[str] | None = None,
    record_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    train_system_cfg = merge_system_config(base_system_cfg, train_system_overrides)
    test_system_cfg = merge_system_config(base_system_cfg, test_system_overrides)
    train_noise_cfg = merge_noise_config(NoiseConfig(), train_noise_overrides)
    base_test_noise_cfg = merge_noise_config(NoiseConfig(), test_noise_overrides)
    train_families = list(train_state_families) if train_state_families is not None else list(target_state_families)

    label_map = {family: idx for idx, family in enumerate(target_state_families)}
    for family in train_families:
        if family not in label_map:
            raise ValueError(f"train_state_family '{family}' must be included in target_state_families")

    metadata = dict(record_metadata or {})
    instance_rows: list[dict[str, Any]] = []

    for backend_type in backend_types:
        backend_shots = [0] if backend_type == "density" else [int(shots) for shots in shot_list]
        for shots in backend_shots:
            x_train_rows: list[np.ndarray] = []
            y_train_rows: list[int] = []
            for seed in train_seeds:
                for family in train_families:
                    train_result = run_feature_pipeline(
                        target_state_family=family,
                        system_cfg=train_system_cfg,
                        noise_cfg=train_noise_cfg,
                        backend_type=backend_type,
                        shots=int(shots),
                        seed=int(seed),
                        latent={
                            "phi_hat": float(train_noise_cfg.phi),
                            "gamma_hat": float(train_noise_cfg.gamma_dephasing),
                        },
                    )
                    x_train_rows.append(_feature_vector(train_result["none_features"], feature_names))
                    y_train_rows.append(label_map[family])

            x_train = np.vstack(x_train_rows)
            y_train = np.asarray(y_train_rows, dtype=int)
            models = _fit_models(x_train, y_train, feature_names, threshold_feature, classifiers)

            for seed in test_seeds:
                for phi in phi_values:
                    for gamma in gamma_values:
                        test_noise_cfg = merge_noise_config(
                            base_test_noise_cfg,
                            {
                                "phi": float(phi),
                                "gamma_dephasing": float(gamma),
                            },
                        )
                        latent_result = run_latent_pipeline(
                            probe_state_family=probe_state_family,
                            system_cfg=test_system_cfg,
                            noise_cfg=test_noise_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                            seed=int(seed),
                        )
                        estimate = latent_result["latent"]

                        for family in target_state_families:
                            feature_result = run_feature_pipeline(
                                target_state_family=family,
                                system_cfg=test_system_cfg,
                                noise_cfg=test_noise_cfg,
                                backend_type=backend_type,
                                shots=int(shots),
                                seed=int(seed),
                                latent=estimate,
                            )
                            feature_map = {
                                "none": feature_result["none_features"],
                                "compensated": feature_result["compensated_features"],
                                "oracle": feature_result["oracle_features"],
                            }

                            for method, features in feature_map.items():
                                x = _feature_vector(features, feature_names)[None, :]
                                for classifier, model in models.items():
                                    score = float(decision_scores(model, x)[0])
                                    pred = int(decision_predictions(model, x)[0])
                                    row = {
                                        "seed": int(seed),
                                        "backend_type": backend_type,
                                        "shots": int(shots),
                                        "classifier": classifier,
                                        "method": method,
                                        "phi_true": float(phi),
                                        "gamma_true": float(gamma),
                                        "state_family": family,
                                        "label": int(label_map[family]),
                                        "score": score,
                                        "prediction": pred,
                                    }
                                    row.update(metadata)
                                    instance_rows.append(row)

    return instance_rows
