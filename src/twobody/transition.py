from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def _available_metric_keys(group: Sequence[dict[str, Any]]) -> list[str]:
    base_metric_keys = [
        "latent_error",
        "phi_mae",
        "gamma_mae",
        "separability_auc",
        "separability_cohen_d",
        "classification_none",
        "classification_compensated",
        "classification_oracle",
        "control_gain",
        "oracle_gap",
    ]
    optional_metric_keys = [
        "classification_structured_oracle",
        "structured_oracle_gap",
        "classification_full_oracle",
        "full_oracle_gap",
        "full_oracle_margin",
        "observable_mae_none",
        "observable_mae_compensated",
        "observable_mae_structured_oracle",
        "observable_mae_full_oracle",
    ]
    return [metric_key for metric_key in [*base_metric_keys, *optional_metric_keys] if metric_key in group[0]]


def classify_transition_state(
    row: dict[str, Any],
    *,
    latent_error_threshold: float = 0.05,
    separability_threshold: float = 0.7,
    compensated_score_threshold: float = 0.9,
    control_gain_threshold: float = 0.05,
) -> str:
    latent_ok = float(row["latent_error"]) <= float(latent_error_threshold)
    separable = float(row["separability_auc"]) >= float(separability_threshold)
    actionable = (
        float(row["classification_compensated"]) >= float(compensated_score_threshold)
        and float(row["control_gain"]) >= float(control_gain_threshold)
    )
    if actionable and latent_ok:
        return "actionable"
    if separable and latent_ok:
        return "transition"
    return "collapse"


def summarize_transition_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            str(row["backend_type"]),
            int(row["shots"]),
            str(row["noise_axis"]),
            float(row["noise_level"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots, noise_axis, noise_level), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "backend_type": backend_type,
            "shots": shots,
            "noise_axis": noise_axis,
            "noise_level": noise_level,
            "n_records": len(group),
        }
        for metric_key in _available_metric_keys(group):
            row[metric_key] = float(np.mean([float(item[metric_key]) for item in group]))
            row[f"{metric_key}_std"] = float(np.std([float(item[metric_key]) for item in group]))
        row["transition_state"] = classify_transition_state(row)
        summary_rows.append(row)
    return summary_rows


def summarize_transition_surface_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (
            str(row["backend_type"]),
            int(row["shots"]),
            str(row["map_name"]),
            float(row["x_value"]),
            float(row["y_value"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots, map_name, x_value, y_value), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "backend_type": backend_type,
            "shots": shots,
            "map_name": map_name,
            "x_value": x_value,
            "y_value": y_value,
            "n_records": len(group),
        }
        for metric_key in _available_metric_keys(group):
            values = [float(item[metric_key]) for item in group]
            row[metric_key] = float(np.mean(values))
            row[f"{metric_key}_std"] = float(np.std(values))
        gains = [float(item["control_gain"]) for item in group]
        row["positive_gain_ratio"] = float(np.mean([gain > 0.0 for gain in gains]))
        row["transition_state"] = classify_transition_state(row)
        summary_rows.append(row)
    return summary_rows
