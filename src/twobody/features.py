from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import re
from typing import Any

import numpy as np


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _mean_abs_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values, dtype=float)))) if values else 0.0


def _local_values(expectations: dict[str, float], axis: str) -> list[float]:
    pattern = re.compile(rf"^{re.escape(axis)}\d+$")
    return [float(value) for name, value in sorted(expectations.items()) if pattern.match(name)]


def _pair_values(expectations: dict[str, float], left_axis: str, right_axis: str) -> list[float]:
    pattern = re.compile(rf"^{re.escape(left_axis)}\d+{re.escape(right_axis)}\d+$")
    return [float(value) for name, value in sorted(expectations.items()) if pattern.match(name)]


def extract_features(
    expectations: dict[str, float],
    latent: dict[str, float] | None = None,
) -> dict[str, float]:
    z_values = _local_values(expectations, "Z")
    zz_values = _pair_values(expectations, "Z", "Z")
    xx_values = _pair_values(expectations, "X", "X")
    yy_values = _pair_values(expectations, "Y", "Y")
    xy_values = _pair_values(expectations, "X", "Y")
    yx_values = _pair_values(expectations, "Y", "X")

    if not z_values:
        raise KeyError("extract_features requires at least one local Z expectation")
    if not zz_values or not xx_values or not yy_values:
        raise KeyError("extract_features requires nearest-neighbor ZZ, XX, and YY expectations")

    z_mean = _mean_or_zero(z_values)
    z_imbalance = float(max(z_values) - min(z_values)) if len(z_values) > 1 else 0.0
    zz = _mean_or_zero(zz_values)
    xx = _mean_or_zero(xx_values)
    yy = _mean_or_zero(yy_values)
    xy = _mean_or_zero(xy_values)
    yx = _mean_or_zero(yx_values)

    feature_dict = {
        "z_mean": z_mean,
        "local_imbalance": z_imbalance,
        "parity": zz,
        "coherence_proxy": 0.5 * (_mean_abs_or_zero(xx_values) + _mean_abs_or_zero(yy_values)),
        "phase_cross": 0.5 * (xy + yx),
        "phase_cos_component": 0.5 * (xx - yy),
        "phase_sin_component": 0.5 * (xy + yx),
        "transverse_norm": float(np.sqrt(xx * xx + yy * yy + xy * xy + yx * yx)),
        "correlation_norm": float(np.sqrt(xx * xx + yy * yy + zz * zz)),
    }
    if latent is not None:
        gamma_hat = float(latent.get("gamma_hat", 0.0))
        coherence_amp = max(1.0 - gamma_hat, 1e-8)
        feature_dict["latent_phi_hat"] = float(latent.get("phi_hat", 0.0))
        feature_dict["latent_gamma_hat"] = gamma_hat
        feature_dict["phase_cos_normalized"] = float(feature_dict["phase_cos_component"] / coherence_amp)
        feature_dict["phase_sin_normalized"] = float(feature_dict["phase_sin_component"] / coherence_amp)
    return feature_dict


def _cohen_d(values_a: Sequence[float], values_b: Sequence[float], eps: float = 1e-8) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("cohen_d requires two non-empty groups")
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    pooled = np.sqrt(max(((a.size - 1) * var_a + (b.size - 1) * var_b) / max(a.size + b.size - 2, 1), eps))
    return float((np.mean(a) - np.mean(b)) / pooled)


def _single_feature_auc(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("single_feature_auc requires two non-empty groups")
    greater = np.sum(a[:, None] > b[None, :])
    ties = np.sum(a[:, None] == b[None, :])
    auc = float((greater + 0.5 * ties) / (a.size * b.size))
    return max(auc, 1.0 - auc)


def _overlap_area(values_a: Sequence[float], values_b: Sequence[float], bins: int = 32) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("overlap_area requires two non-empty groups")
    lower = float(min(np.min(a), np.min(b)))
    upper = float(max(np.max(a), np.max(b)))
    if abs(upper - lower) < 1e-12:
        return 1.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lower, upper), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(lower, upper), density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(hist_a, hist_b) * widths))


def summarize_feature_survival_records(
    records: Sequence[dict[str, Any]],
    *,
    extra_group_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    extra_keys = [str(key) for key in (extra_group_keys or ())]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key_parts: list[Any] = [
            str(row["backend_type"]),
            str(row["method"]),
            int(row["shots"]),
            float(row["phi_true"]),
            float(row["gamma_true"]),
            str(row["feature_name"]),
        ]
        key_parts.extend(row.get(key_name) for key_name in extra_keys)
        key = tuple(key_parts)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        backend_type, method, shots, phi_true, gamma_true, feature_name = key[:6]
        by_label: dict[str, list[float]] = defaultdict(list)
        for row in group:
            by_label[str(row["label"])].append(float(row["feature_value"]))
        if len(by_label) != 2:
            continue
        labels = sorted(by_label)
        left, right = labels[0], labels[1]
        values_left = by_label[left]
        values_right = by_label[right]
        summary_row: dict[str, Any] = {
            "backend_type": backend_type,
            "method": method,
            "shots": shots,
            "phi_true": phi_true,
            "gamma_true": gamma_true,
            "feature_name": feature_name,
            "label_left": left,
            "label_right": right,
            "mean_left": float(np.mean(values_left)),
            "mean_right": float(np.mean(values_right)),
            "cohen_d": _cohen_d(values_left, values_right),
            "single_feature_auc": _single_feature_auc(values_left, values_right),
            "overlap_area": _overlap_area(values_left, values_right),
            "abs_mean_gap": float(abs(np.mean(values_left) - np.mean(values_right))),
            "n_left": len(values_left),
            "n_right": len(values_right),
        }
        for idx, key_name in enumerate(extra_keys, start=6):
            summary_row[key_name] = key[idx]
        summary_rows.append(summary_row)
    return summary_rows
