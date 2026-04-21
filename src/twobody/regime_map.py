from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from src.twobody.actionability import summarize_actionability_surface


def _matches_label_pair(row: dict[str, Any], label_pair: tuple[str, str] | None) -> bool:
    if label_pair is None:
        return True
    expected = {str(label_pair[0]), str(label_pair[1])}
    actual = {str(row["label_left"]), str(row["label_right"])}
    return actual == expected


def _mean_by_key(rows: Sequence[dict[str, Any]], value_key: str) -> dict[tuple[float, float], float]:
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (float(row["phi_true"]), float(row["gamma_true"]))
        grouped[key].append(float(row[value_key]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def classify_regime(
    row: dict[str, Any],
    *,
    phi_mae_threshold: float = 0.05,
    gamma_mae_threshold: float = 0.05,
    reconstruction_gain_threshold: float = 0.02,
    feature_gain_threshold: float = 0.05,
    decision_gain_threshold: float = 0.05,
    compensated_decision_threshold: float = 0.9,
) -> str:
    identifiable = (
        float(row["phi_mae"]) <= float(phi_mae_threshold)
        and float(row["gamma_mae"]) <= float(gamma_mae_threshold)
    )
    recoverable = (
        float(row["reconstruction_gain"]) >= float(reconstruction_gain_threshold)
        or float(row["feature_gain"]) >= float(feature_gain_threshold)
    )
    actionable = (
        float(row["decision_gain"]) >= float(decision_gain_threshold)
        and float(row["decision_compensated"]) >= float(compensated_decision_threshold)
    )

    if identifiable and actionable:
        return "actionable"
    if identifiable and recoverable:
        return "recoverable_not_actionable"
    if identifiable:
        return "identifiable"
    return "collapse"


def summarize_regime_map(
    identifiability_rows: Sequence[dict[str, Any]],
    reconstruction_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    decision_rows: Sequence[dict[str, Any]],
    *,
    backend_type: str,
    shots: int,
    classifier: str,
    feature_name: str,
    feature_metric: str = "abs_mean_gap",
    decision_metric: str = "balanced_accuracy",
    feature_label_pair: tuple[str, str] | None = None,
    phi_mae_threshold: float = 0.05,
    gamma_mae_threshold: float = 0.05,
    reconstruction_gain_threshold: float = 0.02,
    feature_gain_threshold: float = 0.05,
    decision_gain_threshold: float = 0.05,
    compensated_decision_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    ident_filtered = [
        row
        for row in identifiability_rows
        if str(row["backend_type"]) == backend_type and int(row["shots"]) == int(shots)
    ]
    phi_mae_map = _mean_by_key(
        [
            {
                "phi_true": row["phi_true"],
                "gamma_true": row["gamma_true"],
                "value": abs(float(row["phi_hat"]) - float(row["phi_true"])),
            }
            for row in ident_filtered
        ],
        "value",
    )
    gamma_mae_map = _mean_by_key(
        [
            {
                "phi_true": row["phi_true"],
                "gamma_true": row["gamma_true"],
                "value": abs(float(row["gamma_hat"]) - float(row["gamma_true"])),
            }
            for row in ident_filtered
        ],
        "value",
    )

    recon_filtered = [
        row
        for row in reconstruction_rows
        if str(row["backend_type"]) == backend_type and int(row["shots"]) == int(shots)
    ]
    reconstruction_by_method: dict[str, dict[tuple[float, float], float]] = {}
    for method in ("none", "compensated", "oracle"):
        reconstruction_by_method[method] = _mean_by_key(
            [row for row in recon_filtered if str(row["method"]) == method],
            "all_mae",
        )

    feature_filtered = [
        row
        for row in feature_rows
        if str(row["backend_type"]) == backend_type
        and int(row["shots"]) == int(shots)
        and str(row["feature_name"]) == feature_name
        and _matches_label_pair(row, feature_label_pair)
    ]
    feature_by_method: dict[str, dict[tuple[float, float], float]] = {}
    for method in ("clean", "none", "compensated", "oracle"):
        feature_by_method[method] = _mean_by_key(
            [row for row in feature_filtered if str(row["method"]) == method],
            feature_metric,
        )

    actionability_rows = summarize_actionability_surface(
        decision_rows,
        classifier=classifier,
        backend_type=backend_type,
        shots=shots,
        metric_name=decision_metric,
    )
    actionability_map = {
        (float(row["phi_true"]), float(row["gamma_true"])): row for row in actionability_rows
    }

    all_keys = set(phi_mae_map) | set(gamma_mae_map) | set(actionability_map)
    all_keys |= set(reconstruction_by_method["none"]) | set(reconstruction_by_method["compensated"])
    all_keys |= set(feature_by_method["none"]) | set(feature_by_method["compensated"])

    summary_rows: list[dict[str, Any]] = []
    for phi_true, gamma_true in sorted(all_keys):
        decision_row = actionability_map.get((phi_true, gamma_true), {})
        none_reconstruction = reconstruction_by_method["none"].get((phi_true, gamma_true), float("nan"))
        compensated_reconstruction = reconstruction_by_method["compensated"].get((phi_true, gamma_true), float("nan"))
        oracle_reconstruction = reconstruction_by_method["oracle"].get((phi_true, gamma_true), float("nan"))
        clean_feature = feature_by_method["clean"].get((phi_true, gamma_true), float("nan"))
        none_feature = feature_by_method["none"].get((phi_true, gamma_true), float("nan"))
        compensated_feature = feature_by_method["compensated"].get((phi_true, gamma_true), float("nan"))
        oracle_feature = feature_by_method["oracle"].get((phi_true, gamma_true), float("nan"))

        row = {
            "backend_type": backend_type,
            "shots": int(shots),
            "classifier": classifier,
            "feature_name": feature_name,
            "feature_metric": feature_metric,
            "decision_metric": decision_metric,
            "phi_true": float(phi_true),
            "gamma_true": float(gamma_true),
            "phi_mae": float(phi_mae_map.get((phi_true, gamma_true), float("nan"))),
            "gamma_mae": float(gamma_mae_map.get((phi_true, gamma_true), float("nan"))),
            "reconstruction_none": float(none_reconstruction),
            "reconstruction_compensated": float(compensated_reconstruction),
            "reconstruction_oracle": float(oracle_reconstruction),
            "reconstruction_gain": float(none_reconstruction - compensated_reconstruction),
            "reconstruction_oracle_gap": float(compensated_reconstruction - oracle_reconstruction),
            "feature_clean": float(clean_feature),
            "feature_none": float(none_feature),
            "feature_compensated": float(compensated_feature),
            "feature_oracle": float(oracle_feature),
            "feature_gain": float(compensated_feature - none_feature),
            "feature_oracle_gap": float(oracle_feature - compensated_feature),
            "decision_none": float(decision_row.get("none_value", float("nan"))),
            "decision_compensated": float(decision_row.get("compensated_value", float("nan"))),
            "decision_oracle": float(decision_row.get("oracle_value", float("nan"))),
            "decision_gain": float(decision_row.get("comp_gain", float("nan"))),
            "decision_oracle_gap": float(decision_row.get("oracle_gap", float("nan"))),
            "decision_oracle_vs_none_gain": float(decision_row.get("oracle_vs_none_gain", float("nan"))),
        }
        row["regime_label"] = classify_regime(
            row,
            phi_mae_threshold=phi_mae_threshold,
            gamma_mae_threshold=gamma_mae_threshold,
            reconstruction_gain_threshold=reconstruction_gain_threshold,
            feature_gain_threshold=feature_gain_threshold,
            decision_gain_threshold=decision_gain_threshold,
            compensated_decision_threshold=compensated_decision_threshold,
        )
        summary_rows.append(row)

    return summary_rows
