from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, write_csv


COLORS = {
    2: "#4b5563",
    3: "#2563eb",
    4: "#f97316",
    5: "#8b5cf6",
    6: "#dc2626",
    8: "#0f766e",
}
ORACLE_COLOR = "#16a34a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full-scale scalability comparison across result directories.")
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        help="Mapping in the form N=path, e.g. --result-dir 3=results/run_3q",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _parse_result_dirs(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--result-dir must have the form N=path, got: {value}")
        left, right = value.split("=", 1)
        parsed[int(left)] = Path(right)
    return dict(sorted(parsed.items()))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rows(path: Path, filename: str) -> list[dict[str, Any]]:
    return _read_csv(path / filename)


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _sorted(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _float(row, key))


def _first_below(rows: list[dict[str, Any]], metric: str, threshold: float) -> float:
    for row in rows:
        if _float(row, metric) < threshold:
            return _float(row, "noise_level")
    return float("nan")


def _overview_rows(result_dirs: dict[int, Path]) -> list[dict[str, Any]]:
    overview: list[dict[str, Any]] = []
    for n_qubits, result_dir in result_dirs.items():
        transition = _sorted(_rows(result_dir, "transition_summary.csv"), "noise_level")
        composite = _sorted(_rows(result_dir, "composite_sweep_summary.csv"), "noise_level")
        fingerprint = _sorted(_rows(result_dir, "fingerprint_noise_sweep_summary.csv"), "noise_strength")
        tamper = _sorted(_rows(result_dir, "fingerprint_tamper_sweep_summary.csv"), "perturbation_value")

        overview.append(
            {
                "n_qubits": int(n_qubits),
                "transition_gamma0_none": _float(transition[0], "classification_none"),
                "transition_gamma0_compensated": _float(transition[0], "classification_compensated"),
                "transition_gamma_max_none": _float(transition[-1], "classification_none"),
                "transition_gamma_max_compensated": _float(transition[-1], "classification_compensated"),
                "transition_gamma_max_oracle": _float(transition[-1], "classification_oracle"),
                "composite_none_first_below_0_9": _first_below(composite, "classification_none", 0.9),
                "composite_compensated_first_below_0_9": _first_below(composite, "classification_compensated", 0.9),
                "composite_stress_compensated": _float(composite[-1], "classification_compensated"),
                "composite_stress_oracle": _float(composite[-1], "classification_oracle"),
                "composite_stress_oracle_gap": _float(composite[-1], "oracle_gap"),
                "fingerprint_accuracy0": _float(fingerprint[0], "accuracy"),
                "fingerprint_first_drop": next(
                    (_float(row, "noise_strength") for row in fingerprint if _float(row, "accuracy") < 0.99),
                    float("nan"),
                ),
                "fingerprint_accuracy1": _float(fingerprint[-1], "accuracy"),
                "tamper_ba_0_01": next(
                    (_float(row, "balanced_accuracy") for row in tamper if abs(_float(row, "perturbation_value") - 0.01) < 1e-12),
                    float("nan"),
                ),
                "tamper_ba_0_02": next(
                    (_float(row, "balanced_accuracy") for row in tamper if abs(_float(row, "perturbation_value") - 0.02) < 1e-12),
                    float("nan"),
                ),
            }
        )
    return overview


def _configure_accuracy_axis(ax) -> None:
    ax.set_ylim(0.43, 1.04)
    ax.grid(True, color="#d1d5db", alpha=0.45, linewidth=0.7)
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    args = parse_args()
    result_dirs = _parse_result_dirs(args.result_dir)
    output_dir = ensure_dir(args.output_dir or Path("results") / f"twobody_scalability_fullscale_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.10, top=0.89, hspace=0.42, wspace=0.28)
    fig.suptitle("Full-scale scalability: same seed/sweep budget across qubit counts", fontsize=14)

    ax = axes[0, 0]
    for n_qubits, result_dir in result_dirs.items():
        rows = _sorted(_rows(result_dir, "transition_summary.csv"), "noise_level")
        x = [_float(row, "noise_level") for row in rows]
        none = [_float(row, "classification_none") for row in rows]
        comp = [_float(row, "classification_compensated") for row in rows]
        color = COLORS.get(n_qubits, "#111827")
        ax.plot(x, none, linestyle="--", linewidth=1.3, color=color, alpha=0.45)
        ax.plot(x, comp, marker="o", linewidth=2.0, color=color, label=f"{n_qubits}q compensated")
    ax.set_title(r"Latent axis: $\gamma$ sweep", fontsize=12)
    ax.set_xlabel(r"Dephasing $\gamma$", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _configure_accuracy_axis(ax)

    ax = axes[0, 1]
    for n_qubits, result_dir in result_dirs.items():
        rows = _sorted(_rows(result_dir, "composite_sweep_summary.csv"), "noise_level")
        x = [_float(row, "noise_level") for row in rows]
        none = [_float(row, "classification_none") for row in rows]
        comp = [_float(row, "classification_compensated") for row in rows]
        color = COLORS.get(n_qubits, "#111827")
        ax.plot(x, none, linestyle="--", linewidth=1.3, color=color, alpha=0.45)
        ax.plot(x, comp, marker="o", linewidth=2.0, color=color, label=f"{n_qubits}q compensated")
    ax.axhline(1.0, color=ORACLE_COLOR, linestyle=":", linewidth=1.8, label="full oracle")
    ax.set_title("Composite stress", fontsize=12)
    ax.set_xlabel("Composite noise strength", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _configure_accuracy_axis(ax)

    ax = axes[1, 0]
    for n_qubits, result_dir in result_dirs.items():
        rows = _sorted(_rows(result_dir, "fingerprint_noise_sweep_summary.csv"), "noise_strength")
        ax.plot(
            [_float(row, "noise_strength") for row in rows],
            [_float(row, "accuracy") for row in rows],
            marker="o",
            linewidth=2.0,
            color=COLORS.get(n_qubits, "#111827"),
            label=f"{n_qubits}q",
        )
    ax.set_title("Fingerprint stability", fontsize=12)
    ax.set_xlabel("Common disturbance strength", fontsize=10)
    ax.set_ylabel("Node accuracy", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _configure_accuracy_axis(ax)

    ax = axes[1, 1]
    for n_qubits, result_dir in result_dirs.items():
        rows = _sorted(_rows(result_dir, "fingerprint_tamper_sweep_summary.csv"), "perturbation_value")
        ax.plot(
            [_float(row, "perturbation_value") for row in rows],
            [_float(row, "balanced_accuracy") for row in rows],
            marker="o",
            linewidth=2.0,
            color=COLORS.get(n_qubits, "#111827"),
            label=f"{n_qubits}q",
        )
    ax.set_title("Tamper sensitivity", fontsize=12)
    ax.set_xlabel(r"Probe perturbation $\Delta\phi$", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _configure_accuracy_axis(ax)

    fig.text(
        0.5,
        0.025,
        "Dashed lines show uncorrected classifiers; solid markers show latent-compensated classifiers.",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )

    overview = _overview_rows(result_dirs)
    write_csv(output_dir / "scalability_fullscale_overview.csv", overview)
    output_path = output_dir / "paper_scalability_fullscale.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"output_dir={output_dir}")
    print(f"overview_csv={output_dir / 'scalability_fullscale_overview.csv'}")
    print(f"figure_png={output_path}")


if __name__ == "__main__":
    main()
