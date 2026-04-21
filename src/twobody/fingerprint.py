from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.twobody.decision import compute_classification_metrics


@dataclass(slots=True)
class CentroidFingerprintModel:
    feature_names: list[str]
    labels: list[str]
    centroids: np.ndarray


def fit_centroid_fingerprint_model(
    x: np.ndarray,
    labels: Sequence[str],
    feature_names: list[str],
) -> CentroidFingerprintModel:
    unique_labels = sorted({str(label) for label in labels})
    centroids = np.vstack([np.mean(x[np.asarray(labels) == label], axis=0) for label in unique_labels])
    return CentroidFingerprintModel(
        feature_names=list(feature_names),
        labels=unique_labels,
        centroids=centroids,
    )


def centroid_distances(model: CentroidFingerprintModel, x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - model.centroids[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def predict_fingerprint_node(model: CentroidFingerprintModel, x: np.ndarray) -> tuple[list[str], np.ndarray]:
    distances = centroid_distances(model, x)
    indices = np.argmin(distances, axis=1)
    predictions = [model.labels[int(idx)] for idx in indices]
    return predictions, distances


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def summarize_fingerprint_classification(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (str(row["backend_type"]), int(row["shots"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots), group in sorted(grouped.items()):
        accuracy = float(np.mean([str(row["node_id"]) == str(row["prediction"]) for row in group]))
        mean_margin = float(np.mean([float(row["margin"]) for row in group]))
        summary_rows.append(
            {
                "backend_type": backend_type,
                "shots": shots,
                "n_samples": len(group),
                "accuracy": accuracy,
                "mean_margin": mean_margin,
            }
        )
    return summary_rows


def summarize_tamper_detection(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (str(row["backend_type"]), int(row["shots"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots), group in sorted(grouped.items()):
        y_true = np.asarray([int(row["label"]) for row in group], dtype=int)
        y_score = np.asarray([float(row["score"]) for row in group], dtype=float)
        y_pred = np.asarray([int(row["prediction"]) for row in group], dtype=int)
        summary_rows.append(
            {
                "backend_type": backend_type,
                "shots": shots,
                "n_samples": len(group),
                **compute_classification_metrics(y_true, y_score, y_pred),
            }
        )
    return summary_rows


def summarize_fingerprint_stability(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (str(row["backend_type"]), int(row["shots"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, shots), group in sorted(grouped.items()):
        by_node: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in group:
            vector = np.asarray(
                [
                    float(row["phi_hat"]),
                    float(row["gamma_hat"]),
                    float(row["coherence_amp"]),
                    float(row["phase_cos_component"]),
                    float(row["phase_sin_component"]),
                ],
                dtype=float,
            )
            by_node[str(row["node_id"])].append(vector)

        intra_distances: list[float] = []
        intra_cosines: list[float] = []
        for vectors in by_node.values():
            for idx in range(len(vectors)):
                for jdx in range(idx + 1, len(vectors)):
                    intra_distances.append(float(np.linalg.norm(vectors[idx] - vectors[jdx])))
                    intra_cosines.append(cosine_similarity(vectors[idx], vectors[jdx]))

        inter_distances: list[float] = []
        inter_cosines: list[float] = []
        node_items = list(by_node.items())
        for left_idx in range(len(node_items)):
            for right_idx in range(left_idx + 1, len(node_items)):
                _, left_vectors = node_items[left_idx]
                _, right_vectors = node_items[right_idx]
                for left in left_vectors:
                    for right in right_vectors:
                        inter_distances.append(float(np.linalg.norm(left - right)))
                        inter_cosines.append(cosine_similarity(left, right))

        summary_rows.append(
            {
                "backend_type": backend_type,
                "shots": shots,
                "n_nodes": len(by_node),
                "intra_distance_mean": float(np.mean(intra_distances)) if intra_distances else 0.0,
                "inter_distance_mean": float(np.mean(inter_distances)) if inter_distances else 0.0,
                "intra_cosine_mean": float(np.mean(intra_cosines)) if intra_cosines else 1.0,
                "inter_cosine_mean": float(np.mean(inter_cosines)) if inter_cosines else 1.0,
                "distance_separation": float(np.mean(inter_distances) - np.mean(intra_distances)) if intra_distances and inter_distances else 0.0,
            }
        )
    return summary_rows
