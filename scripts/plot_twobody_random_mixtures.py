from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Q_COLORS = {
    2: "#4b5563",
    3: "#2563eb",
    4: "#d97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Layer B random-mixture diagnostics.")
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
    rows = _read_csv(result_dir / "layerB_random_mixture_table.csv")

    by_q = {n: [row for row in rows if int(row["n_qubits"]) == n] for n in (2, 3, 4)}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.84, wspace=0.28)
    fig.suptitle("Random mixture ensemble: 4Q delay suppression tracks out-of-subspace burden", fontsize=14)

    ax = axes[0]
    positions = np.arange(1, 4)
    data = []
    for n in (2, 3, 4):
        values = [_float(row, "delta_tau") for row in by_q[n] if np.isfinite(_float(row, "delta_tau"))]
        data.append(values if values else [np.nan])
    box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=True)
    for patch, n in zip(box["boxes"], (2, 3, 4)):
        patch.set_facecolor(Q_COLORS[n])
        patch.set_alpha(0.28)
        patch.set_edgecolor(Q_COLORS[n])
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.6)
    ax.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(["2Q", "3Q", "4Q"])
    ax.set_ylabel("Collapse delay Δτ", fontsize=10)
    ax.set_title("A. Delay distribution across random mixtures", fontsize=11)
    ax.grid(True, axis="y", color="#e5e7eb", alpha=0.7, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    rows4 = [row for row in rows if int(row["n_qubits"]) == 4]
    x = np.asarray([_float(row, "out_of_subspace_weight") for row in rows4], dtype=float)
    y = np.asarray([_float(row, "delta_tau") for row in rows4], dtype=float)
    c = np.asarray([_float(row, "full_gap_s015") for row in rows4], dtype=float)
    scatter = ax.scatter(x, y, c=c, cmap="viridis", s=56, edgecolors="white", linewidths=0.6)
    ax.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Out-of-subspace burden weight", fontsize=10)
    ax.set_ylabel("4Q collapse delay Δτ", fontsize=10)
    ax.set_title("B. 4Q delay vs out-of-subspace burden", fontsize=11)
    ax.grid(True, color="#e5e7eb", alpha=0.7, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("4Q full-oracle gap at s=0.15", fontsize=9)

    out_png = result_dir / "layerB_random_mixture_diagnostics.png"
    out_pdf = result_dir / "layerB_random_mixture_diagnostics.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"figure_png={out_png}")
    print(f"figure_pdf={out_pdf}")


if __name__ == "__main__":
    main()
