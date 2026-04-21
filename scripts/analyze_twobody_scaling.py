from __future__ import annotations

import argparse
import csv
import math
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
NONE_COLOR = "#c84b55"
COMP_COLOR = "#2f6690"
ORACLE_COLOR = "#3f9b5f"
AMPLITUDE_COLOR = "#2a9d8f"
MEASUREMENT_COLOR = "#e76f51"
COMPOSITE_COLOR = "#7c6f9e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-scale multi-qubit scaling tables and figures.")
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        help="Mapping in the form N=path, e.g. --result-dir 4=results/run_4q",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--fingerprint-threshold", type=float, default=0.99)
    return parser.parse_args()


def _parse_result_dirs(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--result-dir expects N=path, got: {value}")
        left, right = value.split("=", 1)
        parsed[int(left)] = Path(right)
    return dict(sorted(parsed.items()))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rows(result_dir: Path, filename: str) -> list[dict[str, str]]:
    path = result_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_csv(path)


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def _sorted(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _float(row, key))


def _first_below(rows: list[dict[str, str]], metric: str, threshold: float, x_key: str = "noise_level") -> float:
    for row in _sorted(rows, x_key):
        if _float(row, metric) < threshold:
            return _float(row, x_key)
    return float("nan")


def _first_state(rows: list[dict[str, str]], state: str, x_key: str = "noise_level") -> float:
    for row in _sorted(rows, x_key):
        if str(row.get("transition_state", "")) == state:
            return _float(row, x_key)
    return float("nan")


def _value_at(rows: list[dict[str, str]], x_key: str, x_value: float, metric: str) -> float:
    for row in rows:
        if abs(_float(row, x_key) - float(x_value)) < 1e-12:
            return _float(row, metric)
    return float("nan")


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr)) if arr.size else float("nan")


def _component_axis_rows(result_dir: Path, axis: str) -> list[dict[str, str]]:
    return [row for row in _rows(result_dir, "component_sweep_summary.csv") if str(row["noise_axis"]) == axis]


def _fingerprint_first_drop(result_dir: Path, threshold: float) -> float:
    return _first_below(_rows(result_dir, "fingerprint_noise_sweep_summary.csv"), "accuracy", threshold, "noise_strength")


def build_table_scaling_overview(
    result_dirs: dict[int, Path],
    *,
    fingerprint_threshold: float,
) -> list[dict[str, float | int]]:
    table: list[dict[str, float | int]] = []
    for n_qubits, result_dir in result_dirs.items():
        transition = _sorted(_rows(result_dir, "transition_summary.csv"), "noise_level")
        fingerprint = _sorted(_rows(result_dir, "fingerprint_noise_sweep_summary.csv"), "noise_strength")
        tamper = _rows(result_dir, "fingerprint_tamper_sweep_summary.csv")
        table.append(
            {
                "n_qubits": int(n_qubits),
                "transition_gamma0_none": _float(transition[0], "classification_none"),
                "transition_gamma0_compensated": _float(transition[0], "classification_compensated"),
                "transition_gamma_max_none": _float(transition[-1], "classification_none"),
                "transition_gamma_max_compensated": _float(transition[-1], "classification_compensated"),
                "fingerprint_accuracy0": _float(fingerprint[0], "accuracy"),
                "fingerprint_accuracy1": _float(fingerprint[-1], "accuracy"),
                "fingerprint_first_drop": _fingerprint_first_drop(result_dir, fingerprint_threshold),
                "tamper_ba_0_01": _value_at(tamper, "perturbation_value", 0.01, "balanced_accuracy"),
                "tamper_ba_0_02": _value_at(tamper, "perturbation_value", 0.02, "balanced_accuracy"),
            }
        )
    return table


def build_table_boundary_onsets(result_dirs: dict[int, Path], threshold: float) -> list[dict[str, float | int]]:
    table: list[dict[str, float | int]] = []
    for n_qubits, result_dir in result_dirs.items():
        amplitude = _component_axis_rows(result_dir, "eta_amplitude")
        measurement = _component_axis_rows(result_dir, "p_measurement")
        composite = _rows(result_dir, "composite_sweep_summary.csv")
        table.append(
            {
                "n_qubits": int(n_qubits),
                "amplitude_first_collapse": _first_state(amplitude, "collapse"),
                "measurement_first_collapse": _first_state(measurement, "collapse"),
                "composite_none_first_below_0_9": _first_below(composite, "classification_none", threshold),
                "composite_compensated_first_below_0_9": _first_below(
                    composite,
                    "classification_compensated",
                    threshold,
                ),
            }
        )
    return table


def build_table_improvement_stability(result_dirs: dict[int, Path]) -> list[dict[str, float | int]]:
    table: list[dict[str, float | int]] = []
    for n_qubits, result_dir in result_dirs.items():
        transition = _sorted(_rows(result_dir, "transition_summary.csv"), "noise_level")
        none_values = [_float(row, "observable_mae_none") for row in transition]
        comp_values = [_float(row, "observable_mae_compensated") for row in transition]
        none_mean = _mean(none_values)
        comp_mean = _mean(comp_values)
        table.append(
            {
                "n_qubits": int(n_qubits),
                "observable_mae_none_mean": none_mean,
                "observable_mae_comp_mean": comp_mean,
                "relative_gain": (none_mean - comp_mean) / none_mean if abs(none_mean) > 1e-12 else float("nan"),
                "stability_std_none": _std(none_values),
                "stability_std_comp": _std(comp_values),
                "latent_gamma_mae_mean": _mean([_float(row, "gamma_mae") for row in transition]),
                "latent_phi_mae_mean": _mean([_float(row, "phi_mae") for row in transition]),
            }
        )
    return table


def build_boundary_consistency(boundary_table: list[dict[str, float | int]]) -> list[dict[str, float | str]]:
    metric_map = {
        "amplitude": "amplitude_first_collapse",
        "measurement": "measurement_first_collapse",
    }
    rows: list[dict[str, float | str]] = []
    for channel, key in metric_map.items():
        values = [float(row[key]) for row in boundary_table if math.isfinite(float(row[key]))]
        rows.append(
            {
                "channel": channel,
                "min_onset": min(values) if values else float("nan"),
                "max_onset": max(values) if values else float("nan"),
                "delta_onset": (max(values) - min(values)) if values else float("nan"),
            }
        )
    return rows


def build_fingerprint_ratio_table(result_dirs: dict[int, Path]) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    for n_qubits, result_dir in result_dirs.items():
        grouped: dict[float, dict[str, float]] = {}
        for row in _rows(result_dir, "fingerprint_distance_summary.csv"):
            strength = _float(row, "noise_strength")
            grouped.setdefault(strength, {})[str(row["distance_type"])] = _float(row, "distance_mean")
        for strength, values in sorted(grouped.items()):
            inter = float(values.get("inter", float("nan")))
            intra = float(values.get("intra", float("nan")))
            output.append(
                {
                    "n_qubits": int(n_qubits),
                    "noise_strength": float(strength),
                    "inter_distance_mean": inter,
                    "intra_distance_mean": intra,
                    "inter_intra_ratio": inter / intra if abs(intra) > 1e-12 else float("nan"),
                }
            )
    return output


def _configure_accuracy_axis(ax) -> None:
    ax.set_ylim(0.43, 1.04)
    ax.grid(True, color="#d1d5db", alpha=0.45, linewidth=0.7)
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _configure_metric_axis(ax) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.45, linewidth=0.7)
    ax.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_fig1_scaling_overview(
    output_dir: Path,
    overview: list[dict[str, float | int]],
    boundaries: list[dict[str, float | int]],
) -> Path:
    by_n = {int(row["n_qubits"]): row for row in overview}
    boundary_by_n = {int(row["n_qubits"]): row for row in boundaries}
    n_values = sorted(by_n)
    x = np.asarray(n_values, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.9))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.87, hspace=0.38, wspace=0.28)
    fig.suptitle("Scaling overview: invariant recovery and selective boundary shifts", fontsize=14)

    ax = axes[0, 0]
    ax.plot(x, [by_n[n]["transition_gamma_max_none"] for n in n_values], "--o", color=NONE_COLOR, linewidth=2.0, label="None")
    ax.plot(
        x,
        [by_n[n]["transition_gamma_max_compensated"] for n in n_values],
        "-s",
        color=COMP_COLOR,
        linewidth=2.2,
        label="Compensated",
    )
    ax.set_title(r"A. Dephasing recovery at max $\gamma$", fontsize=11)
    ax.set_xlabel("Qubit count", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _configure_accuracy_axis(ax)

    ax = axes[0, 1]
    ax.plot(x, [by_n[n]["fingerprint_first_drop"] for n in n_values], "-o", color="#5c4b8a", linewidth=2.2)
    ax.set_title("B. Fingerprint degradation onset", fontsize=11)
    ax.set_xlabel("Qubit count", fontsize=10)
    ax.set_ylabel("First accuracy drop", fontsize=10)
    ax.set_xticks(x)
    ax.set_ylim(0.0, 0.5)
    _configure_metric_axis(ax)

    ax = axes[1, 0]
    ax.plot(
        x,
        [boundary_by_n[n]["amplitude_first_collapse"] for n in n_values],
        "-o",
        color=AMPLITUDE_COLOR,
        linewidth=2.2,
        label="Amplitude",
    )
    ax.plot(
        x,
        [boundary_by_n[n]["measurement_first_collapse"] for n in n_values],
        "-s",
        color=MEASUREMENT_COLOR,
        linewidth=2.2,
        label="Measurement",
    )
    ax.set_title("C. Channel collapse ordering", fontsize=11)
    ax.set_xlabel("Qubit count", fontsize=10)
    ax.set_ylabel("First collapse onset", fontsize=10)
    ax.set_xticks(x)
    ax.set_ylim(0.0, 0.13)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _configure_metric_axis(ax)

    ax = axes[1, 1]
    ax.plot(x, [by_n[n]["tamper_ba_0_01"] for n in n_values], "-o", color="#7c6f9e", linewidth=2.2, label=r"$\Delta\phi=0.01$")
    ax.plot(x, [by_n[n]["tamper_ba_0_02"] for n in n_values], "-s", color="#d97b2d", linewidth=2.2, label=r"$\Delta\phi=0.02$")
    ax.set_title("D. Tamper sensitivity", fontsize=11)
    ax.set_xlabel("Qubit count", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _configure_accuracy_axis(ax)

    output_path = output_dir / "fig1_scaling_overview_panel.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_fig2_dephasing_recoverability(output_dir: Path, result_dirs: dict[int, Path]) -> Path:
    fig, axes = plt.subplots(1, len(result_dirs), figsize=(11.2, 3.45), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.78, wspace=0.16)
    fig.suptitle("Dephasing recoverability across qubit count", fontsize=14)

    for ax, (n_qubits, result_dir) in zip(np.ravel(axes), result_dirs.items()):
        rows = _sorted(_rows(result_dir, "transition_summary.csv"), "noise_level")
        x = [_float(row, "noise_level") for row in rows]
        ax.plot(x, [_float(row, "classification_none") for row in rows], "--o", color=NONE_COLOR, linewidth=1.8, label="None")
        ax.plot(
            x,
            [_float(row, "classification_compensated") for row in rows],
            "-s",
            color=COMP_COLOR,
            linewidth=2.2,
            label="Compensated",
        )
        ax.plot(x, [_float(row, "classification_oracle") for row in rows], ":D", color=ORACLE_COLOR, linewidth=2.0, label="Oracle")
        ax.set_title(f"{n_qubits} qubits", fontsize=11)
        ax.set_xlabel(r"Dephasing $\gamma$", fontsize=10)
        if n_qubits == min(result_dirs):
            ax.set_ylabel("Balanced accuracy", fontsize=10)
        _configure_accuracy_axis(ax)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="lower right")

    output_path = output_dir / "fig2_dephasing_recoverability.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_fig3_noise_boundary_scaling(output_dir: Path, boundary_table: list[dict[str, float | int]]) -> Path:
    by_n = {int(row["n_qubits"]): row for row in boundary_table}
    n_values = sorted(by_n)
    x = np.asarray(n_values, dtype=int)

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.84)
    ax.set_title("Noise-channel boundary scaling", fontsize=13)
    ax.plot(x, [by_n[n]["amplitude_first_collapse"] for n in n_values], "-o", color=AMPLITUDE_COLOR, linewidth=2.3, label="Amplitude collapse")
    ax.plot(
        x,
        [by_n[n]["measurement_first_collapse"] for n in n_values],
        "-s",
        color=MEASUREMENT_COLOR,
        linewidth=2.3,
        label="Measurement collapse",
    )
    ax.plot(
        x,
        [by_n[n]["composite_compensated_first_below_0_9"] for n in n_values],
        "-^",
        color=COMPOSITE_COLOR,
        linewidth=2.3,
        label="Composite compensated < 0.9",
    )
    ax.set_xlabel("Qubit count", fontsize=10)
    ax.set_ylabel("Onset value", fontsize=10)
    ax.set_xticks(x)
    ax.set_ylim(0.0, 0.66)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _configure_metric_axis(ax)

    output_path = output_dir / "fig3_noise_channel_boundary_scaling.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_fig4_fingerprint_tamper(
    output_dir: Path,
    result_dirs: dict[int, Path],
    ratio_table: list[dict[str, float | int]],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.80, wspace=0.28)
    fig.suptitle("Fingerprint and tamper scaling", fontsize=14)

    ax = axes[0]
    for n_qubits in sorted(result_dirs):
        rows = [row for row in ratio_table if int(row["n_qubits"]) == n_qubits]
        ax.plot(
            [float(row["noise_strength"]) for row in rows],
            [float(row["inter_intra_ratio"]) for row in rows],
            "-o",
            color=COLORS.get(n_qubits, "#111827"),
            linewidth=2.1,
            label=f"{n_qubits}q",
        )
    ax.set_title("A. Inter/Intra fingerprint ratio", fontsize=11)
    ax.set_xlabel("Common disturbance strength", fontsize=10)
    ax.set_ylabel("Inter/Intra distance ratio", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _configure_metric_axis(ax)

    ax = axes[1]
    for n_qubits, result_dir in result_dirs.items():
        rows = _sorted(_rows(result_dir, "fingerprint_tamper_sweep_summary.csv"), "perturbation_value")
        ax.plot(
            [_float(row, "perturbation_value") for row in rows],
            [_float(row, "balanced_accuracy") for row in rows],
            "-o",
            color=COLORS.get(n_qubits, "#111827"),
            linewidth=2.1,
            label=f"{n_qubits}q",
        )
    ax.set_title("B. Tamper sensitivity", fontsize=11)
    ax.set_xlabel(r"Probe perturbation $\Delta\phi$", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _configure_accuracy_axis(ax)

    output_path = output_dir / "fig4_fingerprint_tamper_scaling.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _format_value(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(fval):
        return "nan"
    return f"{fval:.4g}"


def _markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"## {title}\n\nNo rows.\n"
    headers = list(rows[0].keys())
    lines = [f"## {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_summary(
    output_dir: Path,
    overview: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    improvement: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
) -> Path:
    text = "\n".join(
        [
            "# Two-body QEC Scaling Analysis",
            "",
            "Main message: compensation remains recoverable under dephasing as qubit count increases, while amplitude/measurement collapse ordering stays early and composite-noise tolerance becomes increasingly selective beyond the small-system regime.",
            "",
            _markdown_table("Table 1. Scaling Overview", overview),
            _markdown_table("Table 2. Boundary Onset by Noise Channel", boundaries),
            _markdown_table("Table 3. Improvement and Stability Metrics", improvement),
            _markdown_table("Boundary Consistency", consistency),
        ]
    )
    path = output_dir / "scaling_analysis_tables.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    result_dirs = _parse_result_dirs(args.result_dir)
    output_dir = ensure_dir(
        args.output_dir or Path("results") / f"twobody_scaling_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )

    overview = build_table_scaling_overview(result_dirs, fingerprint_threshold=args.fingerprint_threshold)
    boundaries = build_table_boundary_onsets(result_dirs, threshold=args.threshold)
    improvement = build_table_improvement_stability(result_dirs)
    consistency = build_boundary_consistency(boundaries)
    ratio_table = build_fingerprint_ratio_table(result_dirs)

    write_csv(output_dir / "table1_scaling_overview.csv", overview)
    write_csv(output_dir / "table2_boundary_onsets.csv", boundaries)
    write_csv(output_dir / "table3_improvement_stability.csv", improvement)
    write_csv(output_dir / "boundary_consistency_metrics.csv", consistency)
    write_csv(output_dir / "fingerprint_inter_intra_ratio.csv", ratio_table)
    markdown_path = write_markdown_summary(output_dir, overview, boundaries, improvement, consistency)

    fig1 = plot_fig1_scaling_overview(output_dir, overview, boundaries)
    fig2 = plot_fig2_dephasing_recoverability(output_dir, result_dirs)
    fig3 = plot_fig3_noise_boundary_scaling(output_dir, boundaries)
    fig4 = plot_fig4_fingerprint_tamper(output_dir, result_dirs, ratio_table)

    print(f"output_dir={output_dir}")
    print(f"tables_md={markdown_path}")
    print(f"fig1={fig1}")
    print(f"fig2={fig2}")
    print(f"fig3={fig3}")
    print(f"fig4={fig4}")


if __name__ == "__main__":
    main()
