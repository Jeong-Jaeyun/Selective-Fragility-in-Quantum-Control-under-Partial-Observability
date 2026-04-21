from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def summarize_actionability_surface(
    rows: Sequence[dict[str, Any]],
    *,
    classifier: str,
    backend_type: str,
    shots: int,
    metric_name: str = "balanced_accuracy",
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if str(row["classifier"]) == classifier
        and str(row["backend_type"]) == backend_type
        and int(row["shots"]) == int(shots)
    ]

    grouped: dict[tuple[float, float], dict[str, float]] = {}
    for row in filtered:
        key = (float(row["phi_true"]), float(row["gamma_true"]))
        grouped.setdefault(key, {})
        grouped[key][str(row["method"])] = float(row[metric_name])

    summary_rows: list[dict[str, Any]] = []
    for (phi_true, gamma_true), method_map in sorted(grouped.items()):
        none_value = float(method_map.get("none", float("nan")))
        compensated_value = float(method_map.get("compensated", float("nan")))
        oracle_value = float(method_map.get("oracle", float("nan")))
        summary_rows.append(
            {
                "classifier": classifier,
                "backend_type": backend_type,
                "shots": int(shots),
                "metric_name": metric_name,
                "phi_true": phi_true,
                "gamma_true": gamma_true,
                "none_value": none_value,
                "compensated_value": compensated_value,
                "oracle_value": oracle_value,
                "comp_gain": compensated_value - none_value,
                "oracle_gap": oracle_value - compensated_value,
                "oracle_vs_none_gain": oracle_value - none_value,
            }
        )
    return summary_rows
