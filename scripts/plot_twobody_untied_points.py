from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = ["G1_in_subspace", "G2_amplitude", "G3_readout", "G4_mixed_out"]
GROUP_LABELS = {
    "G1_in_subspace": "In-subspace",
    "G2_amplitude": "Amplitude-stressed",
    "G3_readout": "Readout-stressed",
    "G4_mixed_out": "Mixed out-of-subspace",
}
METHODS = [
    ("classification_none", "None", "#c84b55"),
    ("classification_compensated", "Compensated", "#2f6690"),
    ("classification_full_oracle", "Full oracle", "#3f9b5f"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Layer C representative untied-point diagnostics.")
    parser.add_argument("--result-dir", required=True, type=str)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    rows = _read_csv(result_dir / "layerC_untied_point_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=0.84, wspace=0.18)
    fig.suptitle("Representative untied points: 4Q fails most sharply once out-of-subspace burden dominates", fontsize=14)

    group_positions = np.arange(len(GROUP_ORDER))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(METHODS))

    for ax, n_qubits in zip(axes, (2, 3, 4)):
        subset = [row for row in rows if int(row["n_qubits"]) == n_qubits]
        ax.set_title(f"{n_qubits}Q", fontsize=11)
        for offset, (metric, label, color) in zip(offsets, METHODS):
            values = []
            for group in GROUP_ORDER:
                group_rows = [row for row in subset if row["point_group"] == group]
                arr = np.asarray([_float(row, metric) for row in group_rows], dtype=float)
                values.append(float(np.mean(arr)) if arr.size else float("nan"))
            ax.bar(group_positions + offset, values, width=width, color=color, alpha=0.88, label=label)

        ax.axhline(0.9, color="#9ca3af", linestyle=":", linewidth=1.0)
        ax.axhline(0.5, color="#d1d5db", linestyle="--", linewidth=0.8)
        ax.set_xticks(group_positions)
        ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER], rotation=18, ha="right")
        ax.set_ylim(0.43, 1.03)
        ax.grid(True, axis="y", color="#e5e7eb", alpha=0.7, linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Balanced accuracy", fontsize=10)
    axes[1].set_xlabel("Representative point group", fontsize=10)
    axes[2].legend(frameon=False, fontsize=9, loc="upper right")

    out_png = result_dir / "layerC_untied_point_diagnostics.png"
    out_pdf = result_dir / "layerC_untied_point_diagnostics.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"figure_png={out_png}")
    print(f"figure_pdf={out_pdf}")


if __name__ == "__main__":
    main()
