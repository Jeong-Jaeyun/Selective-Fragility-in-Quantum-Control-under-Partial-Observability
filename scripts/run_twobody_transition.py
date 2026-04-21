from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.twobody.decision import (
    balanced_accuracy_score,
    decision_predictions,
    fit_linear_classifier,
    fit_logistic_regression,
    fit_threshold_classifier,
)
from src.twobody.pipeline import run_feature_pipeline, run_latent_pipeline
from src.twobody.transition import summarize_transition_records
from src.twobody.types import NoiseConfig, SystemConfig
from src.utils.io import ensure_dir, load_yaml, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transition and failure-boundary analysis for the two-body package.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/twobody/transition.yaml",
        help="Path to the transition YAML config.",
    )
    return parser.parse_args()


def _as_list(value):
    return list(value) if isinstance(value, list) else [value]


def _feature_vector(feature_dict: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.asarray([float(feature_dict[name]) for name in feature_names], dtype=float)


def _single_feature_auc(values_a: list[float], values_b: list[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    greater = np.sum(a[:, None] > b[None, :])
    ties = np.sum(a[:, None] == b[None, :])
    auc = float((greater + 0.5 * ties) / (a.size * b.size))
    return max(auc, 1.0 - auc)


def _cohen_d(values_a: list[float], values_b: list[float], eps: float = 1e-8) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    pooled = np.sqrt(max(((a.size - 1) * var_a + (b.size - 1) * var_b) / max(a.size + b.size - 2, 1), eps))
    return float((np.mean(a) - np.mean(b)) / pooled)


def _fit_classifier(name: str, x: np.ndarray, y: np.ndarray, feature_names: list[str], threshold_feature: str):
    if name == "threshold":
        return fit_threshold_classifier(x, y, feature_names, threshold_feature)
    if name == "linear":
        return fit_linear_classifier(x, y, feature_names)
    if name == "logistic":
        return fit_logistic_regression(x, y, feature_names)
    raise ValueError(f"unsupported classifier: {name}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg.get("experiment", {})
    system_cfg = SystemConfig(**cfg.get("system", {}))
    base_noise_cfg = NoiseConfig(**cfg.get("noise", {}))
    transition_cfg = cfg.get("transition", {})

    exp_id = datetime.utcnow().strftime("twobody_transition_%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(exp_cfg.get("output_dir", "results")) / exp_id)

    train_seeds = [int(seed) for seed in _as_list(exp_cfg.get("train_seeds", [11, 17, 23]))]
    test_seeds = [int(seed) for seed in _as_list(exp_cfg.get("test_seeds", [101, 103, 107]))]
    backend_types = [str(value) for value in _as_list(exp_cfg.get("backend_types", ["shot"]))]
    shot_list = [int(value) for value in _as_list(exp_cfg.get("shot_list", [8192]))]
    probe_state_family = str(exp_cfg.get("probe_state_family", "bell"))
    target_state_families = [str(value) for value in _as_list(exp_cfg.get("target_state_families", ["bell", "bell_i"]))]
    feature_names = [str(value) for value in _as_list(exp_cfg.get("feature_names", ["phase_cos_component", "phase_sin_component"]))]
    threshold_feature = str(exp_cfg.get("threshold_feature", feature_names[0]))
    classifier_name = str(exp_cfg.get("classifier", "threshold"))
    survival_feature = str(exp_cfg.get("survival_feature", "phase_sin_component"))

    noise_axis = str(transition_cfg.get("noise_axis", "gamma_dephasing"))
    noise_values = [float(value) for value in _as_list(transition_cfg.get("noise_values", [0.0, 0.1, 0.2, 0.3]))]
    fixed_phi = float(transition_cfg.get("fixed_phi", 0.4))
    fixed_gamma = float(transition_cfg.get("fixed_gamma", 0.0))

    label_map = {family: idx for idx, family in enumerate(target_state_families)}
    record_rows: list[dict] = []

    for backend_type in backend_types:
        backend_shots = [0] if backend_type == "density" else shot_list
        for shots in backend_shots:
            x_train_rows: list[np.ndarray] = []
            y_train_rows: list[int] = []
            for seed in train_seeds:
                for family in target_state_families:
                    clean_features = run_feature_pipeline(
                        target_state_family=family,
                        system_cfg=system_cfg,
                        noise_cfg=NoiseConfig(),
                        backend_type=backend_type,
                        shots=int(shots),
                        seed=int(seed),
                        latent={"phi_hat": 0.0, "gamma_hat": 0.0},
                    )["clean_features"]
                    x_train_rows.append(_feature_vector(clean_features, feature_names))
                    y_train_rows.append(label_map[family])

            classifier = _fit_classifier(
                classifier_name,
                np.vstack(x_train_rows),
                np.asarray(y_train_rows, dtype=int),
                feature_names,
                threshold_feature,
            )

            for seed in test_seeds:
                for noise_level in noise_values:
                    noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                    noise_cfg.phi = float(fixed_phi)
                    noise_cfg.gamma_dephasing = float(fixed_gamma)
                    setattr(noise_cfg, noise_axis, float(noise_level))

                    latent_result = run_latent_pipeline(
                        probe_state_family=probe_state_family,
                        system_cfg=system_cfg,
                        noise_cfg=noise_cfg,
                        backend_type=backend_type,
                        shots=int(shots),
                        seed=int(seed),
                    )
                    latent = latent_result["latent"]

                    feature_values_by_label: dict[int, float] = {}
                    predictions_by_method: dict[str, list[int]] = {"none": [], "compensated": [], "oracle": []}
                    labels_true: list[int] = []

                    for family in target_state_families:
                        label = int(label_map[family])
                        labels_true.append(label)
                        feature_result = run_feature_pipeline(
                            target_state_family=family,
                            system_cfg=system_cfg,
                            noise_cfg=noise_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                            seed=int(seed),
                            latent=latent,
                        )
                        feature_values_by_label[label] = float(feature_result["none_features"][survival_feature])

                        for method_name, feature_key in (
                            ("none", "none_features"),
                            ("compensated", "compensated_features"),
                            ("oracle", "oracle_features"),
                        ):
                            x = _feature_vector(feature_result[feature_key], feature_names)[None, :]
                            pred = int(decision_predictions(classifier, x)[0])
                            predictions_by_method[method_name].append(pred)

                    label_zero = min(feature_values_by_label)
                    label_one = max(feature_values_by_label)
                    separability_auc = _single_feature_auc(
                        [feature_values_by_label[label_one]],
                        [feature_values_by_label[label_zero]],
                    )
                    separability_d = _cohen_d(
                        [feature_values_by_label[label_one]],
                        [feature_values_by_label[label_zero]],
                    )

                    y_true = np.asarray(labels_true, dtype=int)
                    ba_none = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["none"], dtype=int))
                    ba_compensated = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["compensated"], dtype=int))
                    ba_oracle = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["oracle"], dtype=int))

                    phi_mae = abs(float(latent["phi_hat"]) - float(noise_cfg.phi))
                    gamma_mae = abs(float(latent["gamma_hat"]) - float(noise_cfg.gamma_dephasing))
                    record_rows.append(
                        {
                            "seed": int(seed),
                            "backend_type": backend_type,
                            "shots": int(shots),
                            "noise_axis": noise_axis,
                            "noise_level": float(noise_level),
                            "phi_true": float(noise_cfg.phi),
                            "gamma_true": float(noise_cfg.gamma_dephasing),
                            "phi_mae": float(phi_mae),
                            "gamma_mae": float(gamma_mae),
                            "latent_error": float(0.5 * (phi_mae + gamma_mae)),
                            "separability_auc": float(separability_auc),
                            "separability_cohen_d": float(separability_d),
                            "classification_none": float(ba_none),
                            "classification_compensated": float(ba_compensated),
                            "classification_oracle": float(ba_oracle),
                            "control_gain": float(ba_compensated - ba_none),
                            "oracle_gap": float(ba_oracle - ba_compensated),
                        }
                    )

    summary_rows = summarize_transition_records(record_rows)
    write_csv(out_dir / "transition_records.csv", record_rows)
    write_csv(out_dir / "transition_summary.csv", summary_rows)

    print(f"output_dir={out_dir}")
    print(f"records_csv={out_dir / 'transition_records.csv'}")
    print(f"summary_csv={out_dir / 'transition_summary.csv'}")


if __name__ == "__main__":
    main()
