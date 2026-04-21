from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def classify_paper_regime(
    row: dict[str, Any],
    *,
    identifiable_classification_threshold: float = 0.95,
    recoverable_separability_threshold: float = 0.85,
    actionable_compensation_threshold: float = 0.9,
    actionable_gain_threshold: float = 0.05,
    latent_error_threshold: float = 0.05,
) -> str:
    latent_error = float(row["latent_error"])
    separability_auc = float(row["separability_auc"])
    classification_none = float(row["classification_none"])
    classification_compensated = float(row["classification_compensated"])
    control_gain = float(row["control_gain"])

    if latent_error <= latent_error_threshold and classification_none >= identifiable_classification_threshold:
        return "identifiable"
    if latent_error <= latent_error_threshold and separability_auc >= recoverable_separability_threshold and control_gain < actionable_gain_threshold:
        return "recoverable"
    if classification_compensated >= actionable_compensation_threshold and control_gain >= actionable_gain_threshold:
        return "actionable"
    return "collapse"


def contiguous_region_spans(
    x_values: Sequence[float],
    labels: Sequence[str],
) -> list[dict[str, float | str]]:
    if len(x_values) != len(labels):
        raise ValueError("x_values and labels must have the same length")
    if not x_values:
        return []

    spans: list[dict[str, float | str]] = []
    start_idx = 0
    for idx in range(1, len(x_values) + 1):
        if idx == len(x_values) or labels[idx] != labels[start_idx]:
            left = float(x_values[start_idx])
            right = float(x_values[idx - 1])
            if idx - 1 > start_idx:
                step = float(x_values[start_idx + 1] - x_values[start_idx])
            elif len(x_values) > 1:
                step = float(x_values[1] - x_values[0])
            else:
                step = 1.0
            spans.append(
                {
                    "label": str(labels[start_idx]),
                    "x_min": left - 0.5 * step,
                    "x_max": right + 0.5 * step,
                }
            )
            start_idx = idx
    return spans
