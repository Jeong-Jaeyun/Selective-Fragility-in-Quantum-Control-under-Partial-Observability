from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_twobody_paper_figures import (  # noqa: E402
    _as_list,
    _build_summary_rows,
    _evaluate_noise_condition,
    _train_classifier,
)
from src.twobody.types import NoiseConfig, SystemConfig  # noqa: E402
from src.utils.io import ensure_dir, load_yaml, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact n=2/3/4 scalability sweeps.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/twobody/scalability_compact.yaml",
        help="Path to the scalability YAML config.",
    )
    return parser.parse_args()


def _system_for_n(base_cfg: dict[str, Any], n_qubits: int) -> SystemConfig:
    data = dict(base_cfg)
    data["n_qubits"] = int(n_qubits)
    return SystemConfig(**data)


def _summarize_by_qubit_count(
    records: list[dict[str, Any]],
    *,
    axis_max_map: dict[str, float],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for n_qubits in sorted({int(row["n_qubits"]) for row in records}):
        subset = [row for row in records if int(row["n_qubits"]) == n_qubits]
        for row in _build_summary_rows(subset, axis_max_map=axis_max_map):
            summary_rows.append({"n_qubits": n_qubits, **row})
    return summary_rows


def _first_drop_level(rows: list[dict[str, Any]], metric: str, threshold: float) -> float:
    ordered = sorted(rows, key=lambda row: float(row["noise_level"]))
    for row in ordered:
        if float(row[metric]) < float(threshold):
            return float(row["noise_level"])
    return float("nan")


def _build_overview_rows(
    summary_rows: list[dict[str, Any]],
    *,
    collapse_threshold: float,
) -> list[dict[str, Any]]:
    overview_rows: list[dict[str, Any]] = []
    keys = sorted({(int(row["n_qubits"]), str(row["noise_axis"])) for row in summary_rows})
    for n_qubits, noise_axis in keys:
        rows = [
            row
            for row in summary_rows
            if int(row["n_qubits"]) == n_qubits and str(row["noise_axis"]) == noise_axis
        ]
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: float(row["noise_level"]))
        baseline = ordered[0]
        stress = ordered[-1]
        onset_none = _first_drop_level(rows, "classification_none", collapse_threshold)
        onset_comp = _first_drop_level(rows, "classification_compensated", collapse_threshold)
        delay = onset_comp - onset_none if np.isfinite(onset_none) and np.isfinite(onset_comp) else float("nan")
        overview_rows.append(
            {
                "n_qubits": int(n_qubits),
                "noise_axis": noise_axis,
                "collapse_threshold": float(collapse_threshold),
                "baseline_noise_level": float(baseline["noise_level"]),
                "stress_noise_level": float(stress["noise_level"]),
                "baseline_none": float(baseline["classification_none"]),
                "baseline_compensated": float(baseline["classification_compensated"]),
                "baseline_oracle": float(baseline["classification_oracle"]),
                "stress_none": float(stress["classification_none"]),
                "stress_compensated": float(stress["classification_compensated"]),
                "stress_oracle": float(stress["classification_oracle"]),
                "stress_control_gain": float(stress["control_gain"]),
                "stress_oracle_gap": float(stress["oracle_gap"]),
                "collapse_onset_none": onset_none,
                "collapse_onset_compensated": onset_comp,
                "collapse_delay": delay,
            }
        )
    return overview_rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg.get("experiment", {})
    transition_cfg = cfg.get("transition", {})
    composite_cfg = cfg.get("composite", {})

    qubit_counts = [int(value) for value in _as_list(exp_cfg.get("qubit_counts", [2, 3, 4]))]
    train_seeds = [int(seed) for seed in _as_list(exp_cfg.get("train_seeds", [11, 17]))]
    test_seeds = [int(seed) for seed in _as_list(exp_cfg.get("test_seeds", [101, 103]))]
    backend_types = [str(value) for value in _as_list(exp_cfg.get("backend_types", ["shot"]))]
    shot_list = [int(value) for value in _as_list(exp_cfg.get("shot_list", [2048]))]
    probe_state_family = str(exp_cfg.get("probe_state_family", "bell"))
    target_state_families = [str(value) for value in _as_list(exp_cfg.get("target_state_families", ["bell_i", "rotated_product"]))]
    feature_names = [str(value) for value in _as_list(exp_cfg.get("feature_names", ["phase_cos_component", "phase_sin_component"]))]
    threshold_feature = str(exp_cfg.get("threshold_feature", feature_names[0]))
    classifier_name = str(exp_cfg.get("classifier", "logistic"))
    survival_feature = str(exp_cfg.get("survival_feature", "phase_sin_component"))
    collapse_threshold = float(exp_cfg.get("collapse_threshold", 0.9))

    transition_axis = str(transition_cfg.get("noise_axis", "gamma_dephasing"))
    transition_values = [float(value) for value in _as_list(transition_cfg.get("noise_values", [0.0, 0.1, 0.2, 0.3]))]
    fixed_phi = float(transition_cfg.get("fixed_phi", 0.8))
    fixed_gamma = float(transition_cfg.get("fixed_gamma", 0.0))

    composite_values = [float(value) for value in _as_list(composite_cfg.get("noise_values", [0.0, 0.25, 0.5, 1.0]))]
    composite_scales = dict(composite_cfg.get("scales", {}))

    exp_prefix = str(exp_cfg.get("name", "twobody_scalability_compact"))
    exp_id = f"{exp_prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(exp_cfg.get("output_dir", "results")) / exp_id)

    axis_max_map = {
        transition_axis: max(transition_values) if transition_values else 1.0,
        "composite_strength": max(composite_values) if composite_values else 1.0,
    }

    records: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for n_qubits in qubit_counts:
        started = perf_counter()
        target_system_cfg = _system_for_n(dict(cfg.get("system", {})), n_qubits)
        probe_system_cfg = _system_for_n(dict(cfg.get("probe_system", cfg.get("system", {}))), n_qubits)
        base_noise_cfg = NoiseConfig(**cfg.get("noise", {}))

        for backend_type in backend_types:
            backend_shots = [0] if backend_type == "density" else shot_list
            for shots in backend_shots:
                classifier, label_map = _train_classifier(
                    train_seeds=train_seeds,
                    target_state_families=target_state_families,
                    target_system_cfg=target_system_cfg,
                    backend_type=backend_type,
                    shots=int(shots),
                    feature_names=feature_names,
                    threshold_feature=threshold_feature,
                    classifier_name=classifier_name,
                )

                for seed in test_seeds:
                    for noise_level in transition_values:
                        noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                        noise_cfg.phi = fixed_phi
                        noise_cfg.gamma_dephasing = fixed_gamma
                        setattr(noise_cfg, transition_axis, float(noise_level))
                        metrics = _evaluate_noise_condition(
                            probe_system_cfg=probe_system_cfg,
                            target_system_cfg=target_system_cfg,
                            noise_cfg=noise_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                            seed=int(seed),
                            probe_state_family=probe_state_family,
                            target_state_families=target_state_families,
                            label_map=label_map,
                            classifier=classifier,
                            feature_names=feature_names,
                            survival_feature=survival_feature,
                        )
                        records.append(
                            {
                                "n_qubits": int(n_qubits),
                                "seed": int(seed),
                                "backend_type": backend_type,
                                "shots": int(shots),
                                "noise_axis": transition_axis,
                                "noise_level": float(noise_level),
                                "phi_true": float(noise_cfg.phi),
                                "gamma_true": float(noise_cfg.gamma_dephasing),
                                "eta_true": float(noise_cfg.eta_amplitude),
                                "p_measurement_true": float(noise_cfg.p_measurement),
                                **metrics,
                            }
                        )

                    for strength in composite_values:
                        noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                        noise_cfg.phi = float(composite_scales.get("phi", 0.0)) * float(strength)
                        noise_cfg.gamma_dephasing = float(composite_scales.get("gamma_dephasing", 0.0)) * float(strength)
                        noise_cfg.eta_amplitude = float(composite_scales.get("eta_amplitude", 0.0)) * float(strength)
                        noise_cfg.p_measurement = float(composite_scales.get("p_measurement", 0.0)) * float(strength)
                        metrics = _evaluate_noise_condition(
                            probe_system_cfg=probe_system_cfg,
                            target_system_cfg=target_system_cfg,
                            noise_cfg=noise_cfg,
                            backend_type=backend_type,
                            shots=int(shots),
                            seed=int(seed),
                            probe_state_family=probe_state_family,
                            target_state_families=target_state_families,
                            label_map=label_map,
                            classifier=classifier,
                            feature_names=feature_names,
                            survival_feature=survival_feature,
                        )
                        records.append(
                            {
                                "n_qubits": int(n_qubits),
                                "seed": int(seed),
                                "backend_type": backend_type,
                                "shots": int(shots),
                                "noise_axis": "composite_strength",
                                "noise_level": float(strength),
                                "phi_true": float(noise_cfg.phi),
                                "gamma_true": float(noise_cfg.gamma_dephasing),
                                "eta_true": float(noise_cfg.eta_amplitude),
                                "p_measurement_true": float(noise_cfg.p_measurement),
                                **metrics,
                            }
                        )

        runtime_rows.append({"n_qubits": int(n_qubits), "runtime_seconds": float(perf_counter() - started)})
        print(f"completed n_qubits={n_qubits}")

    summary_rows = _summarize_by_qubit_count(records, axis_max_map=axis_max_map)
    overview_rows = _build_overview_rows(summary_rows, collapse_threshold=collapse_threshold)

    write_csv(out_dir / "scalability_records.csv", records)
    write_csv(out_dir / "scalability_summary.csv", summary_rows)
    write_csv(out_dir / "scalability_overview.csv", overview_rows)
    write_csv(out_dir / "scalability_runtime.csv", runtime_rows)

    print(f"output_dir={out_dir}")
    print(f"summary_csv={out_dir / 'scalability_summary.csv'}")
    print(f"overview_csv={out_dir / 'scalability_overview.csv'}")


if __name__ == "__main__":
    main()
