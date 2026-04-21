from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RULE_ORDER = [
    "R0_base",
    "R1_phase_heavy",
    "R2_balanced_alt",
    "R3_amplitude_heavy",
    "R4_readout_heavy",
]

RULE_LABELS = {
    "R0_base": "R0 Base",
    "R1_phase_heavy": "R1 Phase-heavy",
    "R2_balanced_alt": "R2 Balanced-alt",
    "R3_amplitude_heavy": "R3 Amplitude-heavy",
    "R4_readout_heavy": "R4 Readout-heavy",
}

Q_COLORS = {
    2: "#4b5563",
    3: "#2563eb",
    4: "#d97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Layer A deterministic tying-rule curves.")
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
    record_rows = _read_csv(result_dir / "layerA_tying_rule_records.csv")
    summary_rows = _read_csv(result_dir / "layerA_tying_rule_summary.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax in axes_flat[-1:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.88, hspace=0.34, wspace=0.22)
    fig.suptitle("Deterministic tying-rule variants: 4Q fragility persists across composite schedules", fontsize=15)

    for idx, rule_name in enumerate(RULE_ORDER):
        ax = axes_flat[idx]
        ax.set_title(RULE_LABELS[rule_name], fontsize=11)

        for n_qubits in (2, 3, 4):
            color = Q_COLORS[n_qubits]
            seed_groups: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in record_rows:
                if int(row["n_qubits"]) == n_qubits and row["rule_name"] == rule_name:
                    seed_groups[int(row["seed"])].append(row)
            for seed, rows in sorted(seed_groups.items()):
                ordered = sorted(rows, key=lambda item: _float(item, "noise_level"))
                ax.plot(
                    [_float(row, "noise_level") for row in ordered],
                    [_float(row, "classification_compensated") for row in ordered],
                    color=color,
                    alpha=0.12,
                    linewidth=0.9,
                )

            ordered_summary = sorted(
                [row for row in summary_rows if int(row["n_qubits"]) == n_qubits and row["rule_name"] == rule_name],
                key=lambda item: _float(item, "noise_level"),
            )
            x = [_float(row, "noise_level") for row in ordered_summary]
            y_none = [_float(row, "classification_none") for row in ordered_summary]
            y_comp = [_float(row, "classification_compensated") for row in ordered_summary]
            ax.plot(x, y_none, "--", color=color, linewidth=1.4, alpha=0.9)
            ax.plot(x, y_comp, "-o", color=color, linewidth=2.2, markersize=4, label=f"{n_qubits}Q")

        ax.axhline(0.9, color="#9ca3af", linestyle=":", linewidth=1.0)
        ax.axhline(0.5, color="#d1d5db", linestyle="--", linewidth=0.8)
        ax.set_xlim(-0.01, 0.61)
        ax.set_ylim(0.43, 1.03)
        ax.grid(True, color="#e5e7eb", alpha=0.6, linewidth=0.7)
        ax.tick_params(labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[1, 0].set_xlabel("Composite strength s", fontsize=10)
    axes[1, 1].set_xlabel("Composite strength s", fontsize=10)
    axes[1, 0].set_ylabel("Balanced accuracy", fontsize=10)
    axes[0, 0].set_ylabel("Balanced accuracy", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right", title="Qubit count")

    fig.text(
        0.07,
        0.92,
        "Thin lines show seed-level compensated curves; dashed lines show none; bold lines show seed means.",
        fontsize=10,
    )

    out_png = result_dir / "layerA_tying_rule_curves.png"
    out_pdf = result_dir / "layerA_tying_rule_curves.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"figure_png={out_png}")
    print(f"figure_pdf={out_pdf}")


if __name__ == "__main__":
    main()
