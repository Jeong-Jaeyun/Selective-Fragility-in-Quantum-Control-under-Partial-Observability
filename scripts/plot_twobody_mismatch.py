from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


METHOD_STYLE = {
    "none": {"color": "#c84b55", "label": "None"},
    "compensated": {"color": "#2f6690", "label": "Compensated"},
    "oracle": {"color": "#3f9b5f", "label": "Oracle"},
}

TITLE_SIZE = 15
SUBTITLE_SIZE = 10.4
ANNOTATION_SIZE = 9.4
LEGEND_SIZE = 10

SCENARIO_LABELS = {
    "channel_mismatch_dephasing_to_amp": "Channel\nmismatch",
    "probe_transfer_bell_i": "Probe transfer\nBell-i",
    "probe_transfer_rotated": "Probe transfer\nrotated",
    "strength_mismatch_low_to_high": "Strength\nmismatch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mismatch and generalization results for the two-body package.")
    parser.add_argument("--results-dir", type=str, required=True, help="Result directory containing mismatch_overview.csv")
    parser.add_argument("--classifier", type=str, default="threshold", help="Classifier slice to visualize")
    parser.add_argument("--shots", type=int, default=2048, help="Shot slice to visualize")
    parser.add_argument("--metric", type=str, default="balanced_accuracy_mean", help="Overview metric to visualize")
    return parser.parse_args()


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#c9d2db", linewidth=0.9, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    overview_path = results_dir / "mismatch_overview.csv"
    with overview_path.open("r", encoding="utf-8", newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if row["classifier"] == args.classifier and int(row["shots"]) == args.shots
        ]

    scenario_names = sorted({row["scenario_name"] for row in rows})
    methods = ["none", "compensated", "oracle"]
    grouped = {(row["scenario_name"], row["method"]): float(row[args.metric]) for row in rows}

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": LEGEND_SIZE,
        }
    )

    x = np.arange(len(scenario_names), dtype=float)
    width = 0.22
    offsets = {"none": -width, "compensated": 0.0, "oracle": width}

    fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    highlight_index = next((idx for idx, name in enumerate(scenario_names) if name == "probe_transfer_bell_i"), None)
    if highlight_index is not None:
        ax.add_patch(
            Rectangle(
                (highlight_index - 0.47, 0.0),
                0.94,
                1.02,
                fill=False,
                edgecolor="#bf8b30",
                linewidth=1.6,
                linestyle=(0, (4, 3)),
                zorder=1,
            )
        )

    for method in methods:
        ys = np.asarray([grouped.get((name, method), np.nan) for name in scenario_names], dtype=float)
        style = METHOD_STYLE[method]
        ax.bar(
            x + offsets[method],
            ys,
            width=width,
            label=style["label"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
        )

    if highlight_index is not None:
        none_score = grouped.get(("probe_transfer_bell_i", "none"), float("nan"))
        comp_score = grouped.get(("probe_transfer_bell_i", "compensated"), float("nan"))
        if np.isfinite(none_score) and np.isfinite(comp_score):
            delta = comp_score - none_score
            label_y = max(none_score, comp_score) + 0.05
            ax.text(
                highlight_index,
                min(label_y, 0.88),
                f"$\\Delta_{{comp-none}}={delta:+.2f}$",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
                fontweight="bold",
                color=METHOD_STYLE["compensated"]["color"] if delta >= 0 else METHOD_STYLE["none"]["color"],
                bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.0},
            )

    ax.set_title("Mismatch Overview", loc="left", pad=26, fontsize=TITLE_SIZE)
    ax.text(
        0.0,
        1.01,
        "Compensation fails at the probe-transfer boundary",
        transform=ax.transAxes,
        clip_on=False,
        fontsize=SUBTITLE_SIZE,
        fontweight="bold",
        color=METHOD_STYLE["none"]["color"],
        va="bottom",
    )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(name, name) for name in scenario_names])
    if highlight_index is not None:
        tick_labels = ax.get_xticklabels()
        tick_labels[highlight_index].set_fontweight("bold")
        tick_labels[highlight_index].set_color("#7a4b00")
    _style_axis(ax)
    ax.legend(frameon=False, loc="lower right", ncol=3, fontsize=LEGEND_SIZE)

    out_path = results_dir / f"mismatch_{args.classifier}_{args.shots}.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"plot_png={out_path}")


if __name__ == "__main__":
    main()
