from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import atan2, sqrt
from typing import Any

import numpy as np


def estimate_bell_probe_latents(expectations: dict[str, float]) -> dict[str, float]:
    xx = float(expectations["X1X2"])
    yy = float(expectations["Y1Y2"])
    xy = float(expectations["X1Y2"])
    yx = float(expectations["Y1X2"])

    cos_component = 0.5 * (xx - yy)
    sin_component = 0.5 * (xy + yx)
    coherence_amp = float(sqrt(max(cos_component * cos_component + sin_component * sin_component, 0.0)))

    phi_hat = float(atan2(sin_component, cos_component))
    gamma_hat = float(1.0 - max(min(coherence_amp, 1.0), 0.0))
    gamma_hat = max(0.0, min(1.0, gamma_hat))

    return {
        "phi_hat": phi_hat,
        "gamma_hat": gamma_hat,
        "coherence_amp": coherence_amp,
        "phase_cos_component": float(cos_component),
        "phase_sin_component": float(sin_component),
    }


def estimate_latent(expectations: dict[str, float]) -> dict[str, float]:
    return estimate_bell_probe_latents(expectations)


def _float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _linear_calibration_fit(x_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if x_true.size <= 1 or float(np.std(x_true)) < 1e-12:
        return 1.0, float(np.mean(y_pred) - np.mean(x_true))
    slope, intercept = np.polyfit(x_true, y_pred, deg=1)
    return float(slope), float(intercept)


def _calibration_summary(x_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    slope, intercept = _linear_calibration_fit(x_true, y_pred)
    fitted = slope * x_true + intercept
    residual = y_pred - fitted
    error = y_pred - x_true
    return {
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_bias": float(np.mean(error)),
        f"{prefix}_error_std": float(np.std(error)),
        f"{prefix}_var": float(np.var(y_pred)),
        f"{prefix}_corr": float(np.corrcoef(x_true, y_pred)[0, 1]) if x_true.size > 1 else 1.0,
        f"{prefix}_calibration_slope": slope,
        f"{prefix}_calibration_intercept": intercept,
        f"{prefix}_calibration_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def evaluate_latent_calibration(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    phi_true = _float_array(row["phi_true"] for row in records)
    phi_hat = _float_array(row["phi_hat"] for row in records)
    gamma_true = _float_array(row["gamma_true"] for row in records)
    gamma_hat = _float_array(row["gamma_hat"] for row in records)
    return {
        **_calibration_summary(phi_true, phi_hat, "phi"),
        **_calibration_summary(gamma_true, gamma_hat, "gamma"),
    }


def summarize_identifiability_records(
    records: Sequence[dict[str, Any]],
    *,
    extra_group_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    extra_keys = [str(key) for key in (extra_group_keys or ())]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key_parts: list[Any] = [str(row["backend_type"]), int(row["shots"])]
        key_parts.extend(row.get(key_name) for key_name in extra_keys)
        key = tuple(key_parts)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        backend_type, shots = key[:2]
        summary_row: dict[str, Any] = {
            "backend_type": backend_type,
            "shots": shots,
            "n_records": len(group),
            **evaluate_latent_calibration(group),
        }
        for idx, key_name in enumerate(extra_keys, start=2):
            summary_row[key_name] = key[idx]
        summary_rows.append(summary_row)
    return summary_rows
