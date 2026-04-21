from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from src.twobody.decision import (
    balanced_accuracy_score,
    compute_classification_metrics,
    decision_predictions,
    decision_scores,
    fit_linear_classifier,
    fit_logistic_regression,
    fit_threshold_classifier,
)
from src.twobody.fingerprint import fit_centroid_fingerprint_model, predict_fingerprint_node
from src.twobody.pipeline import run_feature_pipeline, run_latent_pipeline
from src.twobody.reconstruction import observable_mae
from src.twobody.transition import summarize_transition_records
from src.twobody.types import NoiseConfig, SystemConfig
from src.utils.io import ensure_dir, load_yaml, write_csv


LATENT_FEATURE_NAMES = [
    "phi_hat",
    "gamma_hat",
    "coherence_amp",
    "phase_cos_component",
    "phase_sin_component",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style two-body figure sweeps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/twobody/paper_figures.yaml",
        help="Path to the paper-figure YAML config.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars even when tqdm is installed.",
    )
    return parser.parse_args()


def _as_list(value):
    return list(value) if isinstance(value, list) else [value]


def _progress_bar(*, total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(total=total, desc=desc, dynamic_ncols=True, file=sys.stdout)
    return None


def _status(message: str, *, progress, enabled: bool) -> None:
    if progress is not None:
        progress.set_description(message)
    elif enabled:
        print(message, flush=True)


def _feature_vector(feature_dict: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.asarray([float(feature_dict[name]) for name in feature_names], dtype=float)


def _latent_vector(row: dict[str, float]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in LATENT_FEATURE_NAMES], dtype=float)


def _to_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.vstack([_latent_vector(row) for row in rows])


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


def _train_classifier(
    *,
    train_seeds: list[int],
    target_state_families: list[str],
    target_system_cfg: SystemConfig,
    backend_type: str,
    shots: int,
    feature_names: list[str],
    threshold_feature: str,
    classifier_name: str,
) -> tuple[Any, dict[str, int]]:
    label_map = {family: idx for idx, family in enumerate(target_state_families)}
    x_train_rows: list[np.ndarray] = []
    y_train_rows: list[int] = []
    for seed in train_seeds:
        for family in target_state_families:
            full_oracle_features = run_feature_pipeline(
                target_state_family=family,
                system_cfg=target_system_cfg,
                noise_cfg=NoiseConfig(),
                backend_type=backend_type,
                shots=int(shots),
                seed=int(seed),
                latent={"phi_hat": 0.0, "gamma_hat": 0.0},
                include_full_oracle=True,
            )["full_oracle_features"]
            x_train_rows.append(_feature_vector(full_oracle_features, feature_names))
            y_train_rows.append(label_map[family])

    classifier = _fit_classifier(
        classifier_name,
        np.vstack(x_train_rows),
        np.asarray(y_train_rows, dtype=int),
        feature_names,
        threshold_feature,
    )
    return classifier, label_map


def _evaluate_noise_condition(
    *,
    probe_system_cfg: SystemConfig,
    target_system_cfg: SystemConfig,
    noise_cfg: NoiseConfig,
    backend_type: str,
    shots: int,
    seed: int,
    probe_state_family: str,
    target_state_families: list[str],
    label_map: dict[str, int],
    classifier,
    feature_names: list[str],
    survival_feature: str,
) -> dict[str, float]:
    latent_result = run_latent_pipeline(
        probe_state_family=probe_state_family,
        system_cfg=probe_system_cfg,
        noise_cfg=noise_cfg,
        backend_type=backend_type,
        shots=int(shots),
        seed=int(seed),
    )
    latent = latent_result["latent"]

    feature_values_by_label: dict[int, float] = {}
    predictions_by_method: dict[str, list[int]] = {
        "none": [],
        "compensated": [],
        "structured_oracle": [],
        "full_oracle": [],
    }
    observable_mae_by_method: dict[str, list[float]] = {
        "none": [],
        "compensated": [],
        "structured_oracle": [],
        "full_oracle": [],
    }
    labels_true: list[int] = []

    for family in target_state_families:
        label = int(label_map[family])
        labels_true.append(label)
        feature_result = run_feature_pipeline(
            target_state_family=family,
            system_cfg=target_system_cfg,
            noise_cfg=noise_cfg,
            backend_type=backend_type,
            shots=int(shots),
            seed=int(seed),
            latent=latent,
            include_full_oracle=True,
        )
        feature_values_by_label[label] = float(feature_result["none_features"][survival_feature])
        ideal_expectations = feature_result["ideal_expectations"]
        if ideal_expectations is None:
            raise ValueError("include_full_oracle=True must provide ideal_expectations")
        observable_mae_by_method["none"].append(
            observable_mae(ideal_expectations, feature_result["noisy_expectations"], feature_result["observables"])
        )
        observable_mae_by_method["compensated"].append(
            observable_mae(ideal_expectations, feature_result["compensated_expectations"], feature_result["observables"])
        )
        observable_mae_by_method["structured_oracle"].append(
            observable_mae(ideal_expectations, feature_result["oracle_expectations"], feature_result["observables"])
        )
        observable_mae_by_method["full_oracle"].append(0.0)
        for method_name, feature_key in (
            ("none", "none_features"),
            ("compensated", "compensated_features"),
            ("structured_oracle", "oracle_features"),
            ("full_oracle", "full_oracle_features"),
        ):
            x = _feature_vector(feature_result[feature_key], feature_names)[None, :]
            predictions_by_method[method_name].append(int(decision_predictions(classifier, x)[0]))

    ordered_labels = sorted(feature_values_by_label)
    if len(ordered_labels) < 2:
        raise ValueError("paper figures require at least two target_state_families")
    separability_auc = _single_feature_auc(
        [feature_values_by_label[ordered_labels[-1]]],
        [feature_values_by_label[ordered_labels[0]]],
    )
    separability_d = _cohen_d(
        [feature_values_by_label[ordered_labels[-1]]],
        [feature_values_by_label[ordered_labels[0]]],
    )

    y_true = np.asarray(labels_true, dtype=int)
    ba_none = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["none"], dtype=int))
    ba_comp = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["compensated"], dtype=int))
    ba_structured_oracle = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["structured_oracle"], dtype=int))
    ba_full_oracle = balanced_accuracy_score(y_true, np.asarray(predictions_by_method["full_oracle"], dtype=int))

    phi_mae = abs(float(latent["phi_hat"]) - float(noise_cfg.phi))
    gamma_mae = abs(float(latent["gamma_hat"]) - float(noise_cfg.gamma_dephasing))
    return {
        "phi_mae": float(phi_mae),
        "gamma_mae": float(gamma_mae),
        "latent_error": float(0.5 * (phi_mae + gamma_mae)),
        "separability_auc": float(separability_auc),
        "separability_cohen_d": float(separability_d),
        "classification_none": float(ba_none),
        "classification_compensated": float(ba_comp),
        "classification_structured_oracle": float(ba_structured_oracle),
        "classification_oracle": float(ba_full_oracle),
        "classification_full_oracle": float(ba_full_oracle),
        "control_gain": float(ba_comp - ba_none),
        "structured_oracle_gap": float(ba_structured_oracle - ba_comp),
        "oracle_gap": float(ba_full_oracle - ba_comp),
        "full_oracle_gap": float(ba_full_oracle - ba_comp),
        "full_oracle_margin": float(ba_full_oracle - ba_none),
        "observable_mae_none": float(np.mean(observable_mae_by_method["none"])),
        "observable_mae_compensated": float(np.mean(observable_mae_by_method["compensated"])),
        "observable_mae_structured_oracle": float(np.mean(observable_mae_by_method["structured_oracle"])),
        "observable_mae_full_oracle": float(np.mean(observable_mae_by_method["full_oracle"])),
    }


def _build_summary_rows(
    records: list[dict[str, Any]],
    *,
    axis_max_map: dict[str, float],
) -> list[dict[str, Any]]:
    summary_rows = summarize_transition_records(records)
    for row in summary_rows:
        axis = str(row["noise_axis"])
        max_value = max(float(axis_max_map.get(axis, 1.0)), 1e-8)
        row["noise_fraction"] = float(row["noise_level"]) / max_value
        row["measured_error"] = 1.0 - float(row["classification_none"])
        row["latent_model_error"] = 1.0 - float(row["classification_compensated"])
        row["oracle_error"] = 1.0 - float(row["classification_oracle"])
        row["full_oracle_error"] = 1.0 - float(row["classification_oracle"])
        if "classification_structured_oracle" in row:
            row["structured_oracle_error"] = 1.0 - float(row["classification_structured_oracle"])
        row["noiseless_limit"] = 1.0
    return summary_rows


def _read_node_map(nodes_cfg: list[dict[str, Any]]) -> dict[str, NoiseConfig]:
    node_map: dict[str, NoiseConfig] = {}
    for node in nodes_cfg:
        node_map[str(node["node_id"])] = NoiseConfig(**dict(node.get("noise", {})))
    return node_map


def _with_disturbance(noise_cfg: NoiseConfig, disturbance: dict[str, float], strength: float) -> NoiseConfig:
    data = noise_cfg.to_dict()
    for key, scale in disturbance.items():
        value = float(data.get(key, 0.0)) + float(strength) * float(scale)
        if key.startswith("p_"):
            value = min(max(value, 0.0), 1.0)
        data[key] = value
    return NoiseConfig(**data)


def _collect_node_samples(
    *,
    node_map: dict[str, NoiseConfig],
    seeds: list[int],
    probe_state_family: str,
    probe_system_cfg: SystemConfig,
    backend_type: str,
    shots: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for node_id, noise_cfg in node_map.items():
            result = run_latent_pipeline(
                probe_state_family=probe_state_family,
                system_cfg=probe_system_cfg,
                noise_cfg=noise_cfg,
                backend_type=backend_type,
                shots=int(shots),
                seed=int(seed),
            )
            latent = result["latent"]
            rows.append(
                {
                    "seed": int(seed),
                    "backend_type": backend_type,
                    "shots": int(shots),
                    "node_id": str(node_id),
                    **{key: float(latent[key]) for key in LATENT_FEATURE_NAMES},
                }
            )
    return rows


def _pairwise_distance_rows(
    rows: list[dict[str, Any]],
    *,
    noise_strength: float,
) -> list[dict[str, Any]]:
    by_node: dict[str, list[np.ndarray]] = defaultdict(list)
    backend_type = str(rows[0]["backend_type"])
    shots = int(rows[0]["shots"])
    for row in rows:
        by_node[str(row["node_id"])].append(_latent_vector(row))

    output_rows: list[dict[str, Any]] = []
    for node_id, vectors in by_node.items():
        for idx in range(len(vectors)):
            for jdx in range(idx + 1, len(vectors)):
                output_rows.append(
                    {
                        "backend_type": backend_type,
                        "shots": shots,
                        "noise_strength": float(noise_strength),
                        "distance_type": "intra",
                        "left_node": node_id,
                        "right_node": node_id,
                        "distance": float(np.linalg.norm(vectors[idx] - vectors[jdx])),
                    }
                )

    node_items = list(by_node.items())
    for left_idx in range(len(node_items)):
        for right_idx in range(left_idx + 1, len(node_items)):
            left_node, left_vectors = node_items[left_idx]
            right_node, right_vectors = node_items[right_idx]
            for left in left_vectors:
                for right in right_vectors:
                    output_rows.append(
                        {
                            "backend_type": backend_type,
                            "shots": shots,
                            "noise_strength": float(noise_strength),
                            "distance_type": "inter",
                            "left_node": left_node,
                            "right_node": right_node,
                            "distance": float(np.linalg.norm(left - right)),
                        }
                    )
                # inner loop intentionally keeps every cross-node pair
    return output_rows


def _summarize_fingerprint_accuracy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (str(row["backend_type"]), int(row["shots"]), float(row["noise_strength"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots, noise_strength), group in sorted(grouped.items()):
        accuracy = float(np.mean([str(row["node_id"]) == str(row["prediction"]) for row in group]))
        mean_margin = float(np.mean([float(row["margin"]) for row in group]))
        summary_rows.append(
            {
                "backend_type": backend_type,
                "shots": shots,
                "noise_strength": float(noise_strength),
                "n_samples": len(group),
                "accuracy": accuracy,
                "mean_margin": mean_margin,
            }
        )
    return summary_rows


def _summarize_distance_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            str(row["backend_type"]),
            int(row["shots"]),
            float(row["noise_strength"]),
            str(row["distance_type"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots, noise_strength, distance_type), group in sorted(grouped.items()):
        values = np.asarray([float(row["distance"]) for row in group], dtype=float)
        summary_rows.append(
            {
                "backend_type": backend_type,
                "shots": shots,
                "noise_strength": float(noise_strength),
                "distance_type": distance_type,
                "n_pairs": len(group),
                "distance_mean": float(np.mean(values)) if values.size else 0.0,
                "distance_std": float(np.std(values)) if values.size else 0.0,
                "distance_p10": float(np.quantile(values, 0.1)) if values.size else 0.0,
                "distance_p50": float(np.quantile(values, 0.5)) if values.size else 0.0,
                "distance_p90": float(np.quantile(values, 0.9)) if values.size else 0.0,
            }
        )
    return summary_rows


def _fit_and_eval_tamper_detection(
    *,
    base_noise_cfg: NoiseConfig,
    perturbation_axis: str,
    perturbation_value: float,
    seeds_train: list[int],
    seeds_test: list[int],
    probe_state_family: str,
    probe_system_cfg: SystemConfig,
    backend_type: str,
    shots: int,
) -> list[dict[str, Any]]:
    base_data = base_noise_cfg.to_dict()
    tampered_data = dict(base_data)
    tampered_data[perturbation_axis] = float(tampered_data.get(perturbation_axis, 0.0)) + float(perturbation_value)
    tampered_noise = NoiseConfig(**tampered_data)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for split_name, seeds, target in (("train", seeds_train, train_rows), ("test", seeds_test, test_rows)):
        for seed in seeds:
            for label, tag, noise_cfg in ((0, "normal", base_noise_cfg), (1, "tampered", tampered_noise)):
                result = run_latent_pipeline(
                    probe_state_family=probe_state_family,
                    system_cfg=probe_system_cfg,
                    noise_cfg=noise_cfg,
                    backend_type=backend_type,
                    shots=int(shots),
                    seed=int(seed),
                )
                latent = result["latent"]
                target.append(
                    {
                        "split": split_name,
                        "seed": int(seed),
                        "backend_type": backend_type,
                        "shots": int(shots),
                        "tag": tag,
                        "label": int(label),
                        "perturbation_axis": perturbation_axis,
                        "perturbation_value": float(perturbation_value),
                        **{key: float(latent[key]) for key in LATENT_FEATURE_NAMES},
                    }
                )

    model = fit_logistic_regression(
        _to_feature_matrix(train_rows),
        np.asarray([int(row["label"]) for row in train_rows], dtype=int),
        LATENT_FEATURE_NAMES,
        steps=1500,
    )

    x_test = _to_feature_matrix(test_rows)
    y_true = np.asarray([int(row["label"]) for row in test_rows], dtype=int)
    y_score = decision_scores(model, x_test)
    y_pred = decision_predictions(model, x_test)
    metrics = compute_classification_metrics(y_true, y_score, y_pred)

    output_rows: list[dict[str, Any]] = []
    for row, score, pred in zip(test_rows, y_score, y_pred):
        output_rows.append(
            {
                **row,
                "score": float(score),
                "prediction": int(pred),
                **metrics,
            }
        )
    return output_rows


def _summarize_tamper_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            str(row["backend_type"]),
            int(row["shots"]),
            str(row["perturbation_axis"]),
            float(row["perturbation_value"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots, perturbation_axis, perturbation_value), group in sorted(grouped.items()):
        first = group[0]
        summary_rows.append(
            {
                "backend_type": backend_type,
                "shots": shots,
                "perturbation_axis": perturbation_axis,
                "perturbation_value": float(perturbation_value),
                "n_samples": len(group),
                "accuracy": float(first["accuracy"]),
                "balanced_accuracy": float(first["balanced_accuracy"]),
                "precision": float(first["precision"]),
                "recall": float(first["recall"]),
                "f1": float(first["f1"]),
                "mcc": float(first["mcc"]),
                "roc_auc": float(first["roc_auc"]),
                "pr_auc": float(first["pr_auc"]),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg.get("experiment", {})
    target_system_cfg = SystemConfig(**cfg.get("system", {}))
    probe_system_cfg = SystemConfig(**cfg.get("probe_system", cfg.get("system", {})))
    base_noise_cfg = NoiseConfig(**cfg.get("noise", {}))
    transition_cfg = cfg.get("transition", {})
    decomposition_cfg = cfg.get("decomposition", {})
    fingerprint_cfg = cfg.get("fingerprint", {})
    node_map = _read_node_map(list(cfg.get("nodes", [])))

    exp_prefix = str(exp_cfg.get("name", "twobody_paper_figures"))
    exp_id = f"{exp_prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(exp_cfg.get("output_dir", "results")) / exp_id)

    train_seeds = [int(seed) for seed in _as_list(exp_cfg.get("train_seeds", [11, 17]))]
    test_seeds = [int(seed) for seed in _as_list(exp_cfg.get("test_seeds", [101, 103, 107]))]
    backend_types = [str(value) for value in _as_list(exp_cfg.get("backend_types", ["shot"]))]
    shot_list = [int(value) for value in _as_list(exp_cfg.get("shot_list", [4096]))]
    probe_state_family = str(exp_cfg.get("probe_state_family", "bell"))
    target_state_families = [str(value) for value in _as_list(exp_cfg.get("target_state_families", ["bell", "bell_i"]))]
    feature_names = [str(value) for value in _as_list(exp_cfg.get("feature_names", ["phase_cos_component", "phase_sin_component"]))]
    threshold_feature = str(exp_cfg.get("threshold_feature", feature_names[0]))
    classifier_name = str(exp_cfg.get("classifier", "threshold"))
    survival_feature = str(exp_cfg.get("survival_feature", "phase_sin_component"))

    transition_axis = str(transition_cfg.get("noise_axis", "gamma_dephasing"))
    transition_values = [float(value) for value in _as_list(transition_cfg.get("noise_values", [0.0, 0.05, 0.1, 0.2, 0.3]))]
    fixed_phi = float(transition_cfg.get("fixed_phi", 0.8))
    fixed_gamma = float(transition_cfg.get("fixed_gamma", 0.0))

    component_axes_cfg = dict(decomposition_cfg.get("component_axes", {}))
    composite_strength_values = [float(value) for value in _as_list(decomposition_cfg.get("composite_strength_values", [0.0, 0.25, 0.5, 0.75, 1.0]))]
    composite_scales = dict(decomposition_cfg.get("composite_scales", {}))

    fingerprint_strength_values = [float(value) for value in _as_list(fingerprint_cfg.get("noise_strength_values", [0.0, 0.5, 1.0]))]
    common_disturbance = {str(key): float(value) for key, value in dict(fingerprint_cfg.get("common_disturbance", {})).items()}
    tamper_cfg = dict(fingerprint_cfg.get("tamper", {}))
    tamper_reference = str(tamper_cfg.get("reference_node", next(iter(node_map.keys()))))
    tamper_axis = str(tamper_cfg.get("perturbation_axis", "phi"))
    tamper_values = [float(value) for value in _as_list(tamper_cfg.get("perturbation_values", [0.01, 0.02, 0.04]))]
    progress_enabled = (not args.no_progress) and (tqdm is not None)
    status_enabled = not args.no_progress

    transition_records: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    composite_records: list[dict[str, Any]] = []
    fingerprint_accuracy_rows: list[dict[str, Any]] = []
    fingerprint_distance_rows: list[dict[str, Any]] = []
    fingerprint_tamper_rows: list[dict[str, Any]] = []

    axis_max_map = {
        transition_axis: max(transition_values) if transition_values else 1.0,
        "composite_strength": max(composite_strength_values) if composite_strength_values else 1.0,
    }
    for axis_name, values in component_axes_cfg.items():
        float_values = [float(value) for value in _as_list(values)]
        axis_max_map[str(axis_name)] = max(float_values) if float_values else 1.0

    component_eval_count = sum(len(_as_list(values)) for values in component_axes_cfg.values())
    total_progress_units = 0
    for backend_type in backend_types:
        backend_shots = [0] if backend_type == "density" else shot_list
        total_progress_units += len(backend_shots) * (
            len(test_seeds) * (len(transition_values) + component_eval_count + len(composite_strength_values))
            + len(fingerprint_strength_values)
            + len(tamper_values)
        )

    progress = _progress_bar(total=total_progress_units, desc="paper sweeps", enabled=progress_enabled)

    try:
        for backend_type in backend_types:
            backend_shots = [0] if backend_type == "density" else shot_list
            for shots in backend_shots:
                if progress is not None:
                    progress.set_description(f"{backend_type}:{int(shots)} train")
                else:
                    _status(f"[stage] {backend_type}:{int(shots)} train classifier", progress=progress, enabled=status_enabled)
                classifier, label_map = _train_classifier(
                    train_seeds=train_seeds,
                    target_state_families=target_state_families,
                    target_system_cfg=target_system_cfg,
                    backend_type=backend_type,
                    shots=int(shots),
                    feature_names=feature_names,
                    threshold_feature=threshold_feature,
                    classifier_name=classifier_name,
                )

                for seed in test_seeds:
                    _status(f"[seed] {backend_type}:{int(shots)} seed={int(seed)}", progress=progress, enabled=status_enabled)
                    for noise_level in transition_values:
                        _status(f"{backend_type}:{int(shots)} transition seed={int(seed)}", progress=progress, enabled=False)
                        noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                        noise_cfg.phi = float(fixed_phi)
                        noise_cfg.gamma_dephasing = float(fixed_gamma)
                        setattr(noise_cfg, transition_axis, float(noise_level))
                        metrics = _evaluate_noise_condition(
                            probe_system_cfg=probe_system_cfg,
                            target_system_cfg=target_system_cfg,
                            noise_cfg=noise_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                            seed=int(seed),
                            probe_state_family=probe_state_family,
                            target_state_families=target_state_families,
                            label_map=label_map,
                            classifier=classifier,
                            feature_names=feature_names,
                            survival_feature=survival_feature,
                        )
                        transition_records.append(
                            {
                                "seed": int(seed),
                                "backend_type": backend_type,
                                "shots": int(shots),
                                "noise_axis": transition_axis,
                                "noise_level": float(noise_level),
                                "phi_true": float(noise_cfg.phi),
                                "gamma_true": float(noise_cfg.gamma_dephasing),
                                **metrics,
                            }
                        )
                        if progress is not None:
                            progress.update(1)

                    for component_axis, values in component_axes_cfg.items():
                        for noise_level in [float(value) for value in _as_list(values)]:
                            _status(f"{backend_type}:{int(shots)} {component_axis} seed={int(seed)}", progress=progress, enabled=False)
                            noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                            setattr(noise_cfg, str(component_axis), float(noise_level))
                            metrics = _evaluate_noise_condition(
                                probe_system_cfg=probe_system_cfg,
                                target_system_cfg=target_system_cfg,
                                noise_cfg=noise_cfg,
                                backend_type=backend_type,
                                shots=int(shots),
                                seed=int(seed),
                                probe_state_family=probe_state_family,
                                target_state_families=target_state_families,
                                label_map=label_map,
                                classifier=classifier,
                                feature_names=feature_names,
                                survival_feature=survival_feature,
                            )
                            component_records.append(
                                {
                                    "seed": int(seed),
                                    "backend_type": backend_type,
                                    "shots": int(shots),
                                    "noise_axis": str(component_axis),
                                    "noise_level": float(noise_level),
                                    "phi_true": float(noise_cfg.phi),
                                    "gamma_true": float(noise_cfg.gamma_dephasing),
                                    **metrics,
                                }
                            )
                            if progress is not None:
                                progress.update(1)

                    for strength in composite_strength_values:
                        _status(f"{backend_type}:{int(shots)} composite seed={int(seed)}", progress=progress, enabled=False)
                        noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                        noise_cfg.phi = float(composite_scales.get("phi", 0.0)) * float(strength)
                        noise_cfg.gamma_dephasing = float(composite_scales.get("gamma_dephasing", 0.0)) * float(strength)
                        noise_cfg.eta_amplitude = float(composite_scales.get("eta_amplitude", 0.0)) * float(strength)
                        noise_cfg.p_measurement = float(composite_scales.get("p_measurement", 0.0)) * float(strength)
                        metrics = _evaluate_noise_condition(
                            probe_system_cfg=probe_system_cfg,
                            target_system_cfg=target_system_cfg,
                            noise_cfg=noise_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                            seed=int(seed),
                            probe_state_family=probe_state_family,
                            target_state_families=target_state_families,
                            label_map=label_map,
                            classifier=classifier,
                            feature_names=feature_names,
                            survival_feature=survival_feature,
                        )
                        composite_records.append(
                            {
                                "seed": int(seed),
                                "backend_type": backend_type,
                                "shots": int(shots),
                                "noise_axis": "composite_strength",
                                "noise_level": float(strength),
                                "phi_true": float(noise_cfg.phi),
                                "gamma_true": float(noise_cfg.gamma_dephasing),
                                "eta_true": float(noise_cfg.eta_amplitude),
                                "p_measurement_true": float(noise_cfg.p_measurement),
                                **metrics,
                            }
                        )
                        if progress is not None:
                            progress.update(1)

                _status(f"[stage] {backend_type}:{int(shots)} fingerprint train", progress=progress, enabled=status_enabled)
                train_samples = _collect_node_samples(
                    node_map=node_map,
                    seeds=train_seeds,
                    probe_state_family=probe_state_family,
                    probe_system_cfg=probe_system_cfg,
                    backend_type=backend_type,
                    shots=int(shots),
                )
                fingerprint_model = fit_centroid_fingerprint_model(
                    _to_feature_matrix(train_samples),
                    [str(row["node_id"]) for row in train_samples],
                    LATENT_FEATURE_NAMES,
                )

                for strength in fingerprint_strength_values:
                    _status(
                        f"[fingerprint] {backend_type}:{int(shots)} strength={float(strength):.2f}",
                        progress=progress,
                        enabled=status_enabled and progress is None,
                    )
                    perturbed_node_map = {
                        node_id: _with_disturbance(noise_cfg, common_disturbance, float(strength))
                        for node_id, noise_cfg in node_map.items()
                    }
                    test_rows = _collect_node_samples(
                        node_map=perturbed_node_map,
                        seeds=test_seeds,
                        probe_state_family=probe_state_family,
                        probe_system_cfg=probe_system_cfg,
                        backend_type=backend_type,
                        shots=int(shots),
                    )
                    predictions, distances = predict_fingerprint_node(fingerprint_model, _to_feature_matrix(test_rows))
                    for row, prediction, distance_row in zip(test_rows, predictions, distances):
                        sorted_distances = np.sort(distance_row)
                        margin = float(sorted_distances[1] - sorted_distances[0]) if sorted_distances.size > 1 else 0.0
                        fingerprint_accuracy_rows.append(
                            {
                                **row,
                                "noise_strength": float(strength),
                                "prediction": str(prediction),
                                "margin": margin,
                            }
                        )
                    fingerprint_distance_rows.extend(_pairwise_distance_rows(test_rows, noise_strength=float(strength)))
                    if progress is not None:
                        progress.update(1)

                for perturbation_value in tamper_values:
                    _status(
                        f"[tamper] {backend_type}:{int(shots)} dphi={float(perturbation_value):.4f}",
                        progress=progress,
                        enabled=status_enabled and progress is None,
                    )
                    fingerprint_tamper_rows.extend(
                        _fit_and_eval_tamper_detection(
                            base_noise_cfg=node_map[tamper_reference],
                            perturbation_axis=tamper_axis,
                            perturbation_value=float(perturbation_value),
                            seeds_train=train_seeds,
                            seeds_test=test_seeds,
                            probe_state_family=probe_state_family,
                            probe_system_cfg=probe_system_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                        )
                    )
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    transition_summary = _build_summary_rows(transition_records, axis_max_map=axis_max_map)
    component_summary = _build_summary_rows(component_records, axis_max_map=axis_max_map)
    composite_summary = _build_summary_rows(composite_records, axis_max_map=axis_max_map)
    fingerprint_accuracy_summary = _summarize_fingerprint_accuracy(fingerprint_accuracy_rows)
    fingerprint_distance_summary = _summarize_distance_rows(fingerprint_distance_rows)
    fingerprint_tamper_summary = _summarize_tamper_rows(fingerprint_tamper_rows)

    write_csv(out_dir / "transition_records.csv", transition_records)
    write_csv(out_dir / "transition_summary.csv", transition_summary)
    write_csv(out_dir / "component_sweep_records.csv", component_records)
    write_csv(out_dir / "component_sweep_summary.csv", component_summary)
    write_csv(out_dir / "composite_sweep_records.csv", composite_records)
    write_csv(out_dir / "composite_sweep_summary.csv", composite_summary)
    write_csv(out_dir / "fingerprint_noise_sweep.csv", fingerprint_accuracy_rows)
    write_csv(out_dir / "fingerprint_noise_sweep_summary.csv", fingerprint_accuracy_summary)
    write_csv(out_dir / "fingerprint_distance_distribution.csv", fingerprint_distance_rows)
    write_csv(out_dir / "fingerprint_distance_summary.csv", fingerprint_distance_summary)
    write_csv(out_dir / "fingerprint_tamper_sweep.csv", fingerprint_tamper_rows)
    write_csv(out_dir / "fingerprint_tamper_sweep_summary.csv", fingerprint_tamper_summary)

    print(f"output_dir={out_dir}")
    print(f"transition_summary_csv={out_dir / 'transition_summary.csv'}")
    print(f"component_summary_csv={out_dir / 'component_sweep_summary.csv'}")
    print(f"composite_summary_csv={out_dir / 'composite_sweep_summary.csv'}")
    print(f"fingerprint_summary_csv={out_dir / 'fingerprint_noise_sweep_summary.csv'}")
    print(f"tamper_summary_csv={out_dir / 'fingerprint_tamper_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
