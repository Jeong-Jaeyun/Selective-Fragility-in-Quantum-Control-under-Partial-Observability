from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.twobody.paper_figures import classify_paper_regime, contiguous_region_spans


SERIES_STYLE = {
    "none": {"color": "#c84b55", "linestyle": "--", "marker": "o", "label": "None"},
    "compensated": {"color": "#2f6690", "linestyle": "-", "marker": "s", "label": "Compensated"},
    "oracle": {"color": "#3f9b5f", "linestyle": ":", "marker": "D", "label": "Oracle"},
    "residual": {"color": "#d97b2d", "linestyle": "-.", "marker": "^", "label": "Residual gap"},
    "inter": {"color": "#5c4b8a", "linestyle": "-", "marker": "o", "label": "Inter-cluster"},
    "intra": {"color": "#8b9aad", "linestyle": "-", "marker": "s", "label": "Intra-cluster"},
}

COMPONENT_META = {
    "phi": {"title": "Coherent $\\phi$", "color": "#5f6f7f", "note": "Benign drift"},
    "gamma_dephasing": {"title": "Dephasing $\\gamma$", "color": "#7c6f9e", "note": "Robust regime"},
    "eta_amplitude": {"title": "Amplitude", "color": "#2a9d8f", "note": "Transition collapse"},
    "p_measurement": {"title": "Measurement", "color": "#e76f51", "note": "Early collapse"},
}

REGION_COLORS = {
    "identifiable": "#eaf4cf",
    "recoverable": "#dcebfb",
    "actionable": "#d9f0ef",
    "collapse": "#f8d8db",
}

REGION_LABELS = {
    "identifiable": "Identifiable",
    "recoverable": "Transition",
    "actionable": "Actionable",
    "collapse": "Collapse",
}

TITLE_SIZE = 15
SUBTITLE_SIZE = 10.4
PANEL_TITLE_SIZE = 13.8
NOTE_SIZE = 10.2
ANNOTATION_SIZE = 9.4
LEGEND_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style figures from a paper-figure results directory.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory created by run_twobody_paper_figures.py")
    parser.add_argument("--backend-type", type=str, default="shot", help="Backend type to visualize")
    parser.add_argument("--shots", type=int, default=None, help="Shot slice to visualize")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_optional_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _read_csv(path)


def _resolve_shots(rows: list[dict[str, str]], backend_type: str, requested: int | None) -> int:
    candidates = sorted({int(row["shots"]) for row in rows if row["backend_type"] == backend_type})
    if not candidates:
        raise ValueError(f"no rows found for backend_type={backend_type}")
    if requested is not None:
        if requested not in candidates:
            raise ValueError(f"shots={requested} not found for backend_type={backend_type}")
        return int(requested)
    return int(candidates[0])


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _select_rows(rows: list[dict[str, str]], *, backend_type: str, shots: int) -> list[dict[str, str]]:
    return [row for row in rows if row["backend_type"] == backend_type and int(row["shots"]) == shots]


def _oracle_key(rows: list[dict[str, str]], accuracy: bool) -> str:
    preferred = "classification_full_oracle" if accuracy else "observable_mae_full_oracle"
    fallback = "classification_oracle" if accuracy else "observable_mae_full_oracle"
    if rows and preferred in rows[0]:
        return preferred
    return fallback


def _first_accuracy_drop_onset(rows: list[dict[str, str]], metric_key: str, threshold: float = 0.95) -> float:
    return next((float(row["noise_level"]) for row in rows if float(row[metric_key]) < threshold), math.nan)


def _first_transition_collapse_onset(rows: list[dict[str, str]]) -> float:
    return next((float(row["noise_level"]) for row in rows if str(row.get("transition_state", "")) == "collapse"), math.nan)


def _first_onset(rows: list[dict[str, str]], metric_key: str, threshold: float = 0.95) -> float:
    threshold_onset = _first_accuracy_drop_onset(rows, metric_key, threshold=threshold)
    if not math.isnan(threshold_onset):
        return threshold_onset
    return _first_transition_collapse_onset(rows)


def _series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _band_bounds(center: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(center - spread, 0.0), center + spread


def _distance_separation_stats(distance_rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[float, dict[str, list[float]]] = {}
    for row in distance_rows:
        noise_strength = float(row["noise_strength"])
        distance = float(row["distance"])
        grouped.setdefault(noise_strength, {})
        if row["distance_type"] == "intra":
            pair_key = str(row["left_node"])
        else:
            ordered_pair = tuple(sorted((str(row["left_node"]), str(row["right_node"]))))
            pair_key = f"{ordered_pair[0]}::{ordered_pair[1]}"
        grouped[noise_strength].setdefault(pair_key, []).append(distance)

    x_values: list[float] = []
    inter_mean: list[float] = []
    inter_std: list[float] = []
    intra_mean: list[float] = []
    intra_std: list[float] = []
    for noise_strength in sorted(grouped):
        by_pair = grouped[noise_strength]
        intra_pair_means = [
            float(np.mean(values)) for key, values in by_pair.items() if "::" not in key
        ]
        inter_pair_means = [
            float(np.mean(values)) for key, values in by_pair.items() if "::" in key
        ]
        x_values.append(noise_strength)
        intra_mean.append(float(np.mean(intra_pair_means)))
        intra_std.append(float(np.std(intra_pair_means)))
        inter_mean.append(float(np.mean(inter_pair_means)))
        inter_std.append(float(np.std(inter_pair_means)))

    return {
        "x": {"values": np.asarray(x_values, dtype=float)},
        "inter": {
            "mean": np.asarray(inter_mean, dtype=float),
            "std": np.asarray(inter_std, dtype=float),
        },
        "intra": {
            "mean": np.asarray(intra_mean, dtype=float),
            "std": np.asarray(intra_std, dtype=float),
        },
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "regular",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": LEGEND_SIZE,
            "lines.linewidth": 2.4,
            "lines.markersize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#c9d2db", linewidth=0.9, alpha=0.45)
    ax.set_axisbelow(True)


def _plot_transition_regime(rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["noise_level"]))
    x = _series(rows, "noise_level")
    none = _series(rows, "classification_none")
    compensated = _series(rows, "classification_compensated")
    oracle = _series(rows, _oracle_key(rows, accuracy=True))
    regime_labels = [classify_paper_regime(row) for row in rows]
    spans = contiguous_region_spans(x.tolist(), regime_labels)
    none_onset = _first_onset(rows, "classification_none")
    comp_onset = _first_onset(rows, "classification_compensated")

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for span in spans:
        label = str(span["label"])
        ax.axvspan(float(span["x_min"]), float(span["x_max"]), color=REGION_COLORS[label], alpha=0.35, linewidth=0)

    for key, values in (("none", none), ("compensated", compensated), ("oracle", oracle)):
        style = SERIES_STYLE[key]
        ax.plot(
            x,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=style["label"],
        )

    if not math.isnan(none_onset):
        ax.axvline(none_onset, color=SERIES_STYLE["none"]["color"], linestyle=":", linewidth=1.8)
    if not math.isnan(comp_onset):
        ax.axvline(comp_onset, color=SERIES_STYLE["compensated"]["color"], linestyle=":", linewidth=1.8)
    if not math.isnan(none_onset) and not math.isnan(comp_onset) and comp_onset > none_onset:
        ax.axvspan(none_onset, comp_onset, color=SERIES_STYLE["compensated"]["color"], alpha=0.08)
        delay = comp_onset - none_onset
        ax.text(
            0.56,
            0.56,
            f"Onsets: none {none_onset:.2f}, compensated {comp_onset:.2f}\nCollapse delay $\\Delta={delay:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=ANNOTATION_SIZE,
            color=SERIES_STYLE["compensated"]["color"],
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.0},
        )

    ax.set_title("Transition Regime", loc="left", pad=24, fontsize=TITLE_SIZE)
    ax.text(
        0.0,
        1.01,
        "Compensation delays collapse onset",
        transform=ax.transAxes,
        clip_on=False,
        fontsize=SUBTITLE_SIZE,
        fontweight="bold",
        color=SERIES_STYLE["compensated"]["color"],
        va="bottom",
    )
    ax.set_xlabel("Composite noise strength")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.08)
    _style_axis(ax)
    ax.legend(frameon=False, loc="lower left", ncol=3, fontsize=LEGEND_SIZE)

    _save_figure(fig, out_dir / "paper_transition_regime")


def _plot_noise_decomposition(component_rows: list[dict[str, str]], out_dir: Path) -> None:
    grouped = {
        axis_name: sorted(
            [row for row in component_rows if row["noise_axis"] == axis_name],
            key=lambda row: float(row["noise_level"]),
        )
        for axis_name in COMPONENT_META
    }

    ymax = max(
        float(row["latent_error"]) + float(row["latent_error_std"])
        for axis_rows in grouped.values()
        for row in axis_rows
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.8), sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.08, right=0.98, hspace=0.32, wspace=0.07)
    for ax, axis_name in zip(axes.flat, COMPONENT_META):
        meta = COMPONENT_META[axis_name]
        axis_rows = grouped[axis_name]
        x = _series(axis_rows, "noise_level")
        y = _series(axis_rows, "latent_error")
        y_std = _series(axis_rows, "latent_error_std")
        lower, upper = _band_bounds(y, y_std)
        onset = _first_transition_collapse_onset(axis_rows)

        ax.fill_between(x, lower, upper, color=meta["color"], alpha=0.16)
        ax.plot(x, y, color=meta["color"], marker="o")
        if not math.isnan(onset):
            ax.axvspan(onset, float(np.max(x)), color=REGION_COLORS["collapse"], alpha=0.25)
            ax.axvline(onset, color=meta["color"], linestyle=":", linewidth=1.8)
            ax.text(
                onset,
                ymax * 1.03,
                f"{onset:.2f}",
                color=meta["color"],
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            ax.text(
                0.98,
                0.08,
                "collapse onset",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=meta["color"],
            )
        ax.text(
            0.03,
            0.9,
            meta["note"],
            transform=ax.transAxes,
            fontsize=NOTE_SIZE,
            fontweight="bold",
            color=meta["color"],
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2.0},
        )
        ax.set_title(meta["title"], loc="left", fontsize=PANEL_TITLE_SIZE, pad=6)
        ax.set_xlabel("Noise strength")
        ax.set_ylim(0.0, ymax * 1.14)
        _style_axis(ax)

    axes[0, 0].set_ylabel("Latent error")
    axes[1, 0].set_ylabel("Latent error")

    fig.suptitle(
        "Control Subspace Mismatch Across Noise Channels",
        x=0.085,
        y=0.985,
        ha="left",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.text(
        0.085,
        0.945,
        "Amplitude and measurement define the earliest collapse boundaries.",
        ha="left",
        va="top",
        fontsize=SUBTITLE_SIZE,
        color="#4a5560",
    )
    fig.text(0.82, 0.02, "Shaded band = ±1σ across seeds", fontsize=9.5, color="#5c6670")

    _save_figure(fig, out_dir / "paper_noise_decomposition")


def _plot_model_vs_measurement(rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["noise_level"]))
    x = _series(rows, "noise_level")
    none = _series(rows, "classification_none")
    compensated = _series(rows, "classification_compensated")
    oracle = _series(rows, _oracle_key(rows, accuracy=True))
    residual_gap = np.maximum(compensated - none, 0.0)
    none_onset = _first_onset(rows, "classification_none")
    comp_onset = _first_onset(rows, "classification_compensated")

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    if not math.isnan(none_onset):
        ax.axvspan(none_onset, float(np.max(x)), color=REGION_COLORS["collapse"], alpha=0.32)
    if not math.isnan(none_onset) and not math.isnan(comp_onset) and comp_onset > none_onset:
        ax.axvspan(none_onset, comp_onset, color=SERIES_STYLE["compensated"]["color"], alpha=0.08)

    for key, values in (("none", none), ("compensated", compensated), ("oracle", oracle), ("residual", residual_gap)):
        style = SERIES_STYLE[key]
        ax.plot(
            x,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"] if key != "residual" else None,
            alpha=1.0 if key != "residual" else 0.9,
            linewidth=2.4 if key != "residual" else 2.0,
            label=style["label"],
        )

    if not math.isnan(none_onset):
        ax.axvline(none_onset, color=SERIES_STYLE["none"]["color"], linestyle=":", linewidth=1.8)
    if not math.isnan(comp_onset):
        ax.axvline(comp_onset, color=SERIES_STYLE["compensated"]["color"], linestyle=":", linewidth=1.8)

    ax.set_title("Model vs Measurement", loc="left", pad=24, fontsize=TITLE_SIZE)
    ax.text(
        0.0,
        1.01,
        "Measurement noise causes early collapse",
        transform=ax.transAxes,
        clip_on=False,
        fontsize=SUBTITLE_SIZE,
        fontweight="bold",
        color=SERIES_STYLE["none"]["color"],
        va="bottom",
    )
    if not math.isnan(none_onset) and not math.isnan(comp_onset):
        ax.text(
            0.54,
            0.39,
            f"Onsets: none {none_onset:.2f}, compensated {comp_onset:.2f}",
            transform=ax.transAxes,
            fontsize=ANNOTATION_SIZE,
            color="#38424b",
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.2},
        )

    ax.set_xlabel("Measurement noise strength")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.08)
    _style_axis(ax)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=LEGEND_SIZE)

    _save_figure(fig, out_dir / "paper_model_vs_measurement")


def _plot_fingerprint_stability(
    fingerprint_summary_rows: list[dict[str, str]],
    accuracy_rows: list[dict[str, str]],
    distance_rows: list[dict[str, str]],
    tamper_rows: list[dict[str, str]],
    out_dir: Path,
) -> None:
    accuracy_rows = sorted(accuracy_rows, key=lambda row: float(row["noise_strength"]))
    distance_rows = sorted(distance_rows, key=lambda row: (float(row["noise_strength"]), row["distance_type"]))
    tamper_rows = sorted(tamper_rows, key=lambda row: float(row["perturbation_value"]))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.15), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.992, top=0.92, bottom=0.16, wspace=0.18)

    distance_stats = _distance_separation_stats(distance_rows)
    x_distance = distance_stats["x"]["values"]
    inter_mean = distance_stats["inter"]["mean"]
    inter_std = distance_stats["inter"]["std"]
    intra_mean = distance_stats["intra"]["mean"]
    intra_std = distance_stats["intra"]["std"]
    inter_low, inter_high = _band_bounds(inter_mean, inter_std)
    intra_low, intra_high = _band_bounds(intra_mean, intra_std)

    for key, x_vals, mean_vals, low_vals, high_vals in (
        ("inter", x_distance, inter_mean, inter_low, inter_high),
        ("intra", x_distance, intra_mean, intra_low, intra_high),
    ):
        style = SERIES_STYLE[key]
        axes[0].fill_between(x_vals, low_vals, high_vals, color=style["color"], alpha=0.08)
        axes[0].plot(
            x_vals,
            mean_vals,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.8,
            label=style["label"],
        )
    axes[0].text(
        0.04,
        0.915,
        "Inter > Intra across all noise levels",
        transform=axes[0].transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=SERIES_STYLE["inter"]["color"],
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.4},
    )
    axes[0].set_title("Embedding Separation", loc="left", fontsize=PANEL_TITLE_SIZE)
    axes[0].set_xlabel("Common disturbance strength")
    axes[0].set_ylabel("Embedding Distance")
    _style_axis(axes[0])
    axes[0].legend(frameon=False, loc="lower left", fontsize=LEGEND_SIZE)

    if fingerprint_summary_rows:
        x_acc = np.asarray([float(row["noise_strength"]) for row in fingerprint_summary_rows], dtype=float)
        acc_mean = np.asarray([float(row["fingerprint_accuracy_mean"]) for row in fingerprint_summary_rows], dtype=float)
        acc_std = np.asarray([float(row["fingerprint_accuracy_std"]) for row in fingerprint_summary_rows], dtype=float)
    else:
        x_acc = np.asarray([float(row["noise_strength"]) for row in accuracy_rows], dtype=float)
        acc_mean = np.asarray([float(row["accuracy"]) for row in accuracy_rows], dtype=float)
        acc_std = np.zeros_like(acc_mean)
    acc_low, acc_high = _band_bounds(acc_mean, acc_std)
    axes[1].fill_between(x_acc, acc_low, acc_high, color=SERIES_STYLE["compensated"]["color"], alpha=0.12)
    axes[1].plot(
        x_acc,
        acc_mean,
        color=SERIES_STYLE["compensated"]["color"],
        marker="o",
        linewidth=2.8,
        label="Classification accuracy",
    )
    axes[1].text(
        0.12,
        0.925,
        "Identifiable under moderate stress",
        transform=axes[1].transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=SERIES_STYLE["compensated"]["color"],
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.2},
    )
    axes[1].set_title("Fingerprint Stability", loc="left", fontsize=PANEL_TITLE_SIZE)
    axes[1].set_xlabel("Common disturbance strength")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.08)
    _style_axis(axes[1])
    axes[1].legend(frameon=False, loc="lower left", fontsize=LEGEND_SIZE)

    perturb = np.asarray([float(row["perturbation_value"]) for row in tamper_rows], dtype=float)
    roc_auc = np.asarray([float(row["roc_auc"]) for row in tamper_rows], dtype=float)
    bal_acc = np.asarray([float(row["balanced_accuracy"]) for row in tamper_rows], dtype=float)
    axes[2].plot(
        perturb,
        roc_auc,
        color=SERIES_STYLE["inter"]["color"],
        marker="o",
        linewidth=2.6,
        label="ROC-AUC",
    )
    axes[2].plot(
        perturb,
        bal_acc,
        color=SERIES_STYLE["residual"]["color"],
        marker="s",
        linewidth=2.6,
        label="Balanced accuracy",
    )
    if perturb.size:
        turning_idx = int(np.argmax(roc_auc >= 0.95)) if np.any(roc_auc >= 0.95) else -1
        if turning_idx >= 0:
            axes[2].axvline(perturb[turning_idx], color="#6b7280", linestyle=":", linewidth=1.6)
            axes[2].text(
                perturb[turning_idx],
                0.14,
                "near-perfect detection",
                rotation=90,
                ha="right",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
                color="#5c6670",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1.8},
            )
    axes[2].set_title("Tamper Sensitivity", loc="left", fontsize=PANEL_TITLE_SIZE)
    axes[2].set_xlabel("Tamper perturbation")
    axes[2].set_ylabel("Detection score")
    axes[2].set_ylim(0.0, 1.08)
    _style_axis(axes[2])
    axes[2].legend(frameon=False, loc="lower right", fontsize=LEGEND_SIZE)

    _save_figure(fig, out_dir / "paper_fingerprint_stability")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    _configure_matplotlib()

    transition_rows_all = _read_csv(results_dir / "transition_summary.csv")
    shots = _resolve_shots(transition_rows_all, args.backend_type, args.shots)
    component_rows = _select_rows(
        _read_csv(results_dir / "component_sweep_summary.csv"),
        backend_type=args.backend_type,
        shots=shots,
    )
    composite_rows = _select_rows(
        _read_csv(results_dir / "composite_sweep_summary.csv"),
        backend_type=args.backend_type,
        shots=shots,
    )
    measurement_rows = [row for row in component_rows if row["noise_axis"] == "p_measurement"]
    fingerprint_accuracy_rows = _select_rows(
        _read_csv(results_dir / "fingerprint_noise_sweep_summary.csv"),
        backend_type=args.backend_type,
        shots=shots,
    )
    fingerprint_distance_rows = _select_rows(
        _read_csv(results_dir / "fingerprint_distance_distribution.csv"),
        backend_type=args.backend_type,
        shots=shots,
    )
    fingerprint_tamper_rows = _select_rows(
        _read_csv(results_dir / "fingerprint_tamper_sweep_summary.csv"),
        backend_type=args.backend_type,
        shots=shots,
    )
    fingerprint_summary_rows = _read_optional_csv(results_dir / "fingerprint_summary_table.csv")

    _plot_transition_regime(composite_rows, results_dir)
    _plot_noise_decomposition(component_rows, results_dir)
    _plot_model_vs_measurement(measurement_rows, results_dir)
    _plot_fingerprint_stability(
        fingerprint_summary_rows,
        fingerprint_accuracy_rows,
        fingerprint_distance_rows,
        fingerprint_tamper_rows,
        results_dir,
    )

    print(f"paper_transition_png={results_dir / 'paper_transition_regime.png'}")
    print(f"paper_noise_decomposition_png={results_dir / 'paper_noise_decomposition.png'}")
    print(f"paper_model_vs_measurement_png={results_dir / 'paper_model_vs_measurement.png'}")
    print(f"paper_fingerprint_stability_png={results_dir / 'paper_fingerprint_stability.png'}")


if __name__ == "__main__":
    main()
