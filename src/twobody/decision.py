from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np


@dataclass(slots=True)
class DecisionModel:
    model_type: str
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold_index: int | None = None
    positive_label: int = 1


def _standardize_fit(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def _sigmoid(z: np.ndarray) -> np.ndarray:
    clipped = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_threshold_classifier(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold_feature: str,
) -> DecisionModel:
    x_std, mean, scale = _standardize_fit(x)
    feature_idx = feature_names.index(threshold_feature)
    xj = x_std[:, feature_idx]
    pos_mean = float(np.mean(xj[y == 1]))
    neg_mean = float(np.mean(xj[y == 0]))
    sign = 1.0 if pos_mean >= neg_mean else -1.0
    threshold = 0.5 * (pos_mean + neg_mean)
    weights = np.zeros(x.shape[1], dtype=float)
    weights[feature_idx] = sign
    bias = -sign * threshold
    return DecisionModel(
        model_type="threshold",
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=float(bias),
        threshold_index=feature_idx,
    )


def fit_linear_classifier(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> DecisionModel:
    x_std, mean, scale = _standardize_fit(x)
    x_pos = x_std[y == 1]
    x_neg = x_std[y == 0]
    mu_pos = np.mean(x_pos, axis=0)
    mu_neg = np.mean(x_neg, axis=0)
    weights = mu_pos - mu_neg
    midpoint = 0.5 * (mu_pos + mu_neg)
    bias = -float(weights @ midpoint)
    return DecisionModel(
        model_type="linear",
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=weights.astype(float),
        bias=bias,
    )


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    lr: float = 0.1,
    steps: int = 2000,
    l2: float = 1e-3,
) -> DecisionModel:
    x_std, mean, scale = _standardize_fit(x)
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    y_float = y.astype(float)

    for _ in range(int(steps)):
        logits = x_std @ weights + bias
        probs = _sigmoid(logits)
        error = probs - y_float
        grad_w = (x_std.T @ error) / x_std.shape[0] + l2 * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b

    return DecisionModel(
        model_type="logistic",
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=float(bias),
    )


def decision_scores(model: DecisionModel, x: np.ndarray) -> np.ndarray:
    x_std = _standardize_apply(x, model.mean, model.scale)
    linear = x_std @ model.weights + model.bias
    if model.model_type == "logistic":
        return _sigmoid(linear)
    return linear


def decision_predictions(model: DecisionModel, x: np.ndarray) -> np.ndarray:
    scores = decision_scores(model, x)
    if model.model_type == "logistic":
        return (scores >= 0.5).astype(int)
    return (scores >= 0.0).astype(int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    tpr = float(np.mean(y_pred[pos] == 1))
    tnr = float(np.mean(y_pred[neg] == 0))
    return 0.5 * (tpr + tnr)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")
    greater = np.sum(pos_scores[:, None] > neg_scores[None, :])
    ties = np.sum(pos_scores[:, None] == neg_scores[None, :])
    return float((greater + 0.5 * ties) / (pos_scores.size * neg_scores.size))


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(np.sum(y_true == 1), 1)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": pr_auc_score(y_true, y_score),
    }


def summarize_decision_records(
    records: list[dict[str, Any]],
    *,
    extra_group_keys: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    extra_keys = [str(key) for key in (extra_group_keys or ())]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key_parts: list[Any] = [
            str(row["backend_type"]),
            str(row["method"]),
            str(row["classifier"]),
            int(row["shots"]),
            float(row["phi_true"]),
            float(row["gamma_true"]),
        ]
        key_parts.extend(row.get(key_name) for key_name in extra_keys)
        key = tuple(key_parts)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        backend_type, method, classifier, shots, phi_true, gamma_true = key[:6]
        y_true = np.asarray([int(row["label"]) for row in group], dtype=int)
        y_score = np.asarray([float(row["score"]) for row in group], dtype=float)
        y_pred = np.asarray([int(row["prediction"]) for row in group], dtype=int)
        metrics = compute_classification_metrics(y_true, y_score, y_pred)
        summary_row: dict[str, Any] = {
            "backend_type": backend_type,
            "method": method,
            "classifier": classifier,
            "shots": shots,
            "phi_true": phi_true,
            "gamma_true": gamma_true,
            "n_instances": len(group),
            **metrics,
        }
        for idx, key_name in enumerate(extra_keys, start=6):
            summary_row[key_name] = key[idx]
        summary_rows.append(
            summary_row
        )
    return summary_rows
