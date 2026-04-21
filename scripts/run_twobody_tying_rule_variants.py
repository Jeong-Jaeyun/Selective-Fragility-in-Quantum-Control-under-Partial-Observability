from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from scripts.run_twobody_paper_figures import _evaluate_noise_condition, _train_classifier
from src.twobody.transition import summarize_transition_records
from src.twobody.types import NoiseConfig, SystemConfig
from src.utils.io import ensure_dir, load_yaml, write_csv


RULES: dict[str, dict[str, float]] = {
    "R0_base": {
        "phi": 0.8,
        "gamma_dephasing": 0.3,
        "eta_amplitude": 0.15,
        "p_measurement": 0.1,
    },
    "R1_phase_heavy": {
        "phi": 1.0,
        "gamma_dephasing": 0.45,
        "eta_amplitude": 0.08,
        "p_measurement": 0.05,
    },
    "R2_balanced_alt": {
        "phi": 0.6,
        "gamma_dephasing": 0.25,
        "eta_amplitude": 0.25,
        "p_measurement": 0.12,
    },
    "R3_amplitude_heavy": {
        "phi": 0.5,
        "gamma_dephasing": 0.2,
        "eta_amplitude": 0.4,
        "p_measurement": 0.08,
    },
    "R4_readout_heavy": {
        "phi": 0.5,
        "gamma_dephasing": 0.2,
        "eta_amplitude": 0.1,
        "p_measurement": 0.25,
    },
}


CONFIG_MAP = {
    2: "configs/twobody/paper_figures_final.yaml",
    3: "configs/twobody/paper_figures_final_3q.yaml",
    4: "configs/twobody/paper_figures_final_4q.yaml",
}


DEFAULT_STRENGTHS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
FIXED_WINDOWS = [0.10, 0.15, 0.20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer A deterministic composite tying-rule sweeps.")
    parser.add_argument("--qubits", nargs="*", type=int, default=[2, 3, 4], help="Qubit counts to evaluate.")
    parser.add_argument("--rules", nargs="*", type=str, default=list(RULES.keys()), help="Rule names to evaluate.")
    parser.add_argument(
        "--strength-values",
        nargs="*",
        type=float,
        default=DEFAULT_STRENGTHS,
        help="Composite strength values.",
    )
    parser.add_argument("--threshold", type=float, default=0.9, help="Collapse threshold for BA.")
    parser.add_argument("--output-dir", type=str, default="results", help="Parent output directory.")
    parser.add_argument("--exp-name", type=str, default="twobody_tying_rule_variants")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def _status(message: str, *, progress) -> None:
    if progress is not None:
        progress.set_description(message)
    else:
        print(message, flush=True)


def _series_value(rows: list[dict[str, Any]], strength: float, metric: str) -> float:
    for row in rows:
        if abs(_float(row, "noise_level") - float(strength)) < 1e-12:
            return _float(row, metric)
    return float("nan")


def _first_below(rows: list[dict[str, Any]], metric: str, threshold: float) -> float:
    for row in sorted(rows, key=lambda item: _float(item, "noise_level")):
        if _float(row, metric) < threshold:
            return _float(row, "noise_level")
    return float("nan")


def _build_summary(records: list[dict[str, Any]], strength_max: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["n_qubits"]), str(row["rule_name"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (n_qubits, rule_name), group in sorted(grouped.items()):
        rule_summary = summarize_transition_records(group)
        for row in rule_summary:
            row["n_qubits"] = int(n_qubits)
            row["rule_name"] = rule_name
            row["noise_fraction"] = _float(row, "noise_level") / max(strength_max, 1e-8)
            summary_rows.append(row)
    return summary_rows


def _build_rule_summary(
    summary_rows: list[dict[str, Any]],
    *,
    threshold: float,
    fixed_windows: list[float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(int(row["n_qubits"]), str(row["rule_name"]))].append(row)

    output: list[dict[str, Any]] = []
    for (n_qubits, rule_name), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: _float(item, "noise_level"))
        scales = RULES[rule_name]
        tau_none = _first_below(ordered, "classification_none", threshold)
        tau_comp = _first_below(ordered, "classification_compensated", threshold)
        tau_struct = _first_below(ordered, "classification_structured_oracle", threshold)
        tau_full = _first_below(ordered, "classification_full_oracle", threshold)
        row: dict[str, Any] = {
            "n_qubits": int(n_qubits),
            "rule_name": rule_name,
            "phi_scale": float(scales["phi"]),
            "gamma_scale": float(scales["gamma_dephasing"]),
            "eta_scale": float(scales["eta_amplitude"]),
            "p_measurement_scale": float(scales["p_measurement"]),
            "in_subspace_burden": float(scales["phi"] + scales["gamma_dephasing"]),
            "out_of_subspace_burden": float(scales["eta_amplitude"] + scales["p_measurement"]),
            "tau_none": tau_none,
            "tau_comp": tau_comp,
            "tau_structured_oracle": tau_struct,
            "tau_full_oracle": tau_full,
            "delta_tau": tau_comp - tau_none if math.isfinite(tau_none) and math.isfinite(tau_comp) else float("nan"),
            "delta_tau_structured": tau_struct - tau_comp if math.isfinite(tau_struct) and math.isfinite(tau_comp) else float("nan"),
            "delta_tau_full": tau_full - tau_comp if math.isfinite(tau_full) and math.isfinite(tau_comp) else float("nan"),
        }
        for strength in fixed_windows:
            tag = f"s_{strength:.2f}".replace(".", "_")
            row[f"{tag}_ba_none"] = _series_value(ordered, strength, "classification_none")
            row[f"{tag}_ba_comp"] = _series_value(ordered, strength, "classification_compensated")
            row[f"{tag}_ba_struct"] = _series_value(ordered, strength, "classification_structured_oracle")
            row[f"{tag}_ba_full"] = _series_value(ordered, strength, "classification_full_oracle")
            row[f"{tag}_gap_struct"] = _series_value(ordered, strength, "structured_oracle_gap")
            row[f"{tag}_gap_full"] = _series_value(ordered, strength, "full_oracle_gap")
            row[f"{tag}_latent_error"] = _series_value(ordered, strength, "latent_error")
            row[f"{tag}_observable_mae_comp"] = _series_value(ordered, strength, "observable_mae_compensated")
            row[f"{tag}_separability_auc"] = _series_value(ordered, strength, "separability_auc")
        output.append(row)
    return output


def _build_fixed_window_table(summary_rows: list[dict[str, Any]], fixed_windows: list[float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in summary_rows:
        strength = _float(row, "noise_level")
        if not any(abs(strength - float(value)) < 1e-12 for value in fixed_windows):
            continue
        output.append(
            {
                "n_qubits": int(row["n_qubits"]),
                "rule_name": str(row["rule_name"]),
                "noise_level": float(strength),
                "classification_none": _float(row, "classification_none"),
                "classification_compensated": _float(row, "classification_compensated"),
                "classification_structured_oracle": _float(row, "classification_structured_oracle"),
                "classification_full_oracle": _float(row, "classification_full_oracle"),
                "structured_oracle_gap": _float(row, "structured_oracle_gap"),
                "full_oracle_gap": _float(row, "full_oracle_gap"),
                "latent_error": _float(row, "latent_error"),
                "observable_mae_none": _float(row, "observable_mae_none"),
                "observable_mae_compensated": _float(row, "observable_mae_compensated"),
                "separability_auc": _float(row, "separability_auc"),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    exp_id = f"{args.exp_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(args.output_dir) / exp_id)

    qubit_counts = [int(value) for value in args.qubits]
    rules = [str(value) for value in args.rules]
    unknown_rules = [rule for rule in rules if rule not in RULES]
    if unknown_rules:
        raise ValueError(f"unknown rules requested: {unknown_rules}")
    strength_values = [float(value) for value in args.strength_values]

    records: list[dict[str, Any]] = []
    total_steps = len(qubit_counts) * len(rules) * len(strength_values)
    total_evals = 0
    per_qubit_context: dict[int, dict[str, Any]] = {}
    for n_qubits in qubit_counts:
        cfg = load_yaml(CONFIG_MAP[n_qubits])
        exp_cfg = cfg.get("experiment", {})
        backend_type = str(list(exp_cfg.get("backend_types", ["shot"]))[0])
        shots = int(list(exp_cfg.get("shot_list", [8192]))[0]) if backend_type != "density" else 0
        test_seeds = [int(seed) for seed in list(exp_cfg.get("test_seeds", [101, 103, 107]))]
        total_evals += len(rules) * len(strength_values) * len(test_seeds)
        per_qubit_context[n_qubits] = {
            "cfg": cfg,
            "backend_type": backend_type,
            "shots": shots,
            "test_seeds": test_seeds,
        }

    progress = None
    if (not args.no_progress) and tqdm is not None:
        progress = tqdm(total=total_evals, desc="tying rules", dynamic_ncols=True, file=sys.stdout)

    try:
        for n_qubits in qubit_counts:
            context = per_qubit_context[n_qubits]
            cfg = context["cfg"]
            exp_cfg = cfg.get("experiment", {})
            target_system_cfg = SystemConfig(**cfg.get("system", {}))
            probe_system_cfg = SystemConfig(**cfg.get("probe_system", cfg.get("system", {})))
            base_noise_cfg = NoiseConfig(**cfg.get("noise", {}))
            train_seeds = [int(seed) for seed in list(exp_cfg.get("train_seeds", [11, 17]))]
            test_seeds = context["test_seeds"]
            backend_type = context["backend_type"]
            shots = context["shots"]
            probe_state_family = str(exp_cfg.get("probe_state_family", "bell"))
            target_state_families = [str(value) for value in list(exp_cfg.get("target_state_families", ["bell_i", "rotated_product"]))]
            feature_names = [str(value) for value in list(exp_cfg.get("feature_names", ["phase_cos_component", "phase_sin_component"]))]
            threshold_feature = str(exp_cfg.get("threshold_feature", feature_names[0]))
            classifier_name = str(exp_cfg.get("classifier", "logistic"))
            survival_feature = str(exp_cfg.get("survival_feature", feature_names[0]))

            _status(f"[train] {n_qubits}Q classifier", progress=progress)
            classifier, label_map = _train_classifier(
                train_seeds=train_seeds,
                target_state_families=target_state_families,
                target_system_cfg=target_system_cfg,
                backend_type=backend_type,
                shots=shots,
                feature_names=feature_names,
                threshold_feature=threshold_feature,
                classifier_name=classifier_name,
            )

            for rule_name in rules:
                scales = RULES[rule_name]
                _status(f"[rule] {n_qubits}Q {rule_name}", progress=progress)
                for seed in test_seeds:
                    _status(f"[seed] {n_qubits}Q {rule_name} seed={int(seed)}", progress=progress)
                    for strength in strength_values:
                        noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                        noise_cfg.phi = float(scales["phi"]) * float(strength)
                        noise_cfg.gamma_dephasing = float(scales["gamma_dephasing"]) * float(strength)
                        noise_cfg.eta_amplitude = float(scales["eta_amplitude"]) * float(strength)
                        noise_cfg.p_measurement = float(scales["p_measurement"]) * float(strength)
                        metrics = _evaluate_noise_condition(
                            probe_system_cfg=probe_system_cfg,
                            target_system_cfg=target_system_cfg,
                            noise_cfg=noise_cfg,
                            backend_type=backend_type,
                            shots=shots,
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
                                "rule_name": rule_name,
                                "seed": int(seed),
                                "backend_type": backend_type,
                                "shots": int(shots),
                                "noise_axis": "composite_strength",
                                "noise_level": float(strength),
                                "phi_scale": float(scales["phi"]),
                                "gamma_scale": float(scales["gamma_dephasing"]),
                                "eta_scale": float(scales["eta_amplitude"]),
                                "p_measurement_scale": float(scales["p_measurement"]),
                                "phi_true": float(noise_cfg.phi),
                                "gamma_true": float(noise_cfg.gamma_dephasing),
                                "eta_true": float(noise_cfg.eta_amplitude),
                                "p_measurement_true": float(noise_cfg.p_measurement),
                                **metrics,
                            }
                        )
                        if progress is not None:
                            progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    summary_rows = _build_summary(records, strength_max=max(strength_values) if strength_values else 1.0)
    rule_summary_rows = _build_rule_summary(summary_rows, threshold=float(args.threshold), fixed_windows=FIXED_WINDOWS)
    fixed_window_rows = _build_fixed_window_table(summary_rows, FIXED_WINDOWS)

    write_csv(out_dir / "layerA_tying_rule_records.csv", records)
    write_csv(out_dir / "layerA_tying_rule_summary.csv", summary_rows)
    write_csv(out_dir / "layerA_tying_rule_rule_summary.csv", rule_summary_rows)
    write_csv(out_dir / "layerA_tying_rule_fixed_window.csv", fixed_window_rows)

    print(f"output_dir={out_dir}")
    print(f"records_csv={out_dir / 'layerA_tying_rule_records.csv'}")
    print(f"summary_csv={out_dir / 'layerA_tying_rule_summary.csv'}")
    print(f"rule_summary_csv={out_dir / 'layerA_tying_rule_rule_summary.csv'}")
    print(f"fixed_window_csv={out_dir / 'layerA_tying_rule_fixed_window.csv'}")


if __name__ == "__main__":
    main()
