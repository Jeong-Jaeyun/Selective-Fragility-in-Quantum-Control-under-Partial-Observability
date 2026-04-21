from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twobody import summarize_decision_records
from src.twobody.decision_experiment import run_decision_condition
from src.twobody.types import SystemConfig
from src.utils.io import ensure_dir, load_yaml, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mismatch and generalization experiments for the two-body package.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/twobody/mismatch.yaml",
        help="Path to the mismatch YAML config.",
    )
    return parser.parse_args()


def _as_list(value):
    return list(value) if isinstance(value, list) else [value]


def _build_overview_rows(summary_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in summary_rows:
        key = tuple(row[key_name] for key_name in ["scenario_name", "backend_type", "shots", "classifier", "method"])
        grouped[key].append(row)

    metric_names = ["accuracy", "balanced_accuracy", "f1", "mcc", "roc_auc", "pr_auc", "precision", "recall"]
    overview_rows: list[dict] = []
    for key, group in sorted(grouped.items()):
        row = {
            "scenario_name": key[0],
            "backend_type": key[1],
            "shots": key[2],
            "classifier": key[3],
            "method": key[4],
            "n_cells": len(group),
        }
        for metric_name in metric_names:
            values = [float(item[metric_name]) for item in group]
            row[f"{metric_name}_mean"] = sum(values) / len(values)
        row["generalization_gap_mean"] = float(
            sum(float(item["oracle_gap_to_comp"]) for item in group) / len(group)
        ) if all("oracle_gap_to_comp" in item for item in group) else 0.0
        overview_rows.append(row)
    return overview_rows


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    exp_cfg = cfg.get("experiment", {})
    system_cfg = SystemConfig(**cfg.get("system", {}))
    base_noise_cfg = dict(cfg.get("noise", {}))
    sweep_cfg = cfg.get("sweep", {})
    scenarios = list(cfg.get("scenarios", []))

    exp_prefix = str(exp_cfg.get("name", "twobody_mismatch"))
    exp_id = f"{exp_prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(exp_cfg.get("output_dir", "results")) / exp_id)

    train_seeds = [int(seed) for seed in _as_list(exp_cfg.get("train_seeds", [11, 17, 23]))]
    test_seeds = [int(seed) for seed in _as_list(exp_cfg.get("test_seeds", [101, 103, 107]))]
    backend_types = [str(value) for value in _as_list(exp_cfg.get("backend_types", ["shot"]))]
    shot_list = [int(value) for value in _as_list(exp_cfg.get("shot_list", [2048]))]
    default_probe_state_family = str(exp_cfg.get("probe_state_family", "bell"))
    target_state_families = [str(value) for value in _as_list(exp_cfg.get("target_state_families", ["bell", "bell_i"]))]
    feature_names = [str(value) for value in _as_list(exp_cfg.get("feature_names", ["phase_cos_component", "phase_sin_component"]))]
    threshold_feature = str(exp_cfg.get("threshold_feature", feature_names[0]))
    classifiers = [str(value) for value in _as_list(exp_cfg.get("classifiers", ["threshold", "linear", "logistic"]))]
    default_phi_values = [float(value) for value in _as_list(sweep_cfg.get("phi", [0.0]))]
    default_gamma_values = [float(value) for value in _as_list(sweep_cfg.get("gamma_dephasing", [0.0]))]

    instance_rows: list[dict] = []
    extra_keys = ["scenario_name", "scenario_type", "test_probe_state_family"]

    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        scenario_type = str(scenario.get("type", scenario_name))
        test_probe_state_family = str(scenario.get("test_probe_state_family", default_probe_state_family))
        phi_values = [float(value) for value in _as_list(scenario.get("phi", default_phi_values))]
        gamma_values = [float(value) for value in _as_list(scenario.get("gamma_dephasing", default_gamma_values))]
        train_noise_overrides = dict(scenario.get("train_noise_overrides", {}))
        test_noise_overrides = dict(base_noise_cfg)
        test_noise_overrides.update(dict(scenario.get("test_noise_overrides", {})))
        train_system_overrides = dict(scenario.get("train_system_overrides", {})) or None
        test_system_overrides = dict(scenario.get("test_system_overrides", {})) or None

        rows = run_decision_condition(
            train_seeds=train_seeds,
            test_seeds=test_seeds,
            backend_types=backend_types,
            shot_list=shot_list,
            probe_state_family=test_probe_state_family,
            target_state_families=target_state_families,
            feature_names=feature_names,
            threshold_feature=threshold_feature,
            classifiers=classifiers,
            phi_values=phi_values,
            gamma_values=gamma_values,
            base_system_cfg=system_cfg,
            train_system_overrides=train_system_overrides,
            test_system_overrides=test_system_overrides,
            train_noise_overrides=train_noise_overrides,
            test_noise_overrides=test_noise_overrides,
            record_metadata={
                "scenario_name": scenario_name,
                "scenario_type": scenario_type,
                "test_probe_state_family": test_probe_state_family,
            },
        )
        instance_rows.extend(rows)

    summary_rows = summarize_decision_records(instance_rows, extra_group_keys=extra_keys)

    by_base: dict[tuple, dict[str, float]] = {}
    for row in summary_rows:
        key = (
            row["scenario_name"],
            row["backend_type"],
            row["shots"],
            row["classifier"],
            row["phi_true"],
            row["gamma_true"],
            row["test_probe_state_family"],
        )
        by_base.setdefault(key, {})
        by_base[key][str(row["method"])] = float(row["balanced_accuracy"])

    enriched_rows: list[dict] = []
    for row in summary_rows:
        key = (
            row["scenario_name"],
            row["backend_type"],
            row["shots"],
            row["classifier"],
            row["phi_true"],
            row["gamma_true"],
            row["test_probe_state_family"],
        )
        method_map = by_base.get(key, {})
        enriched = dict(row)
        enriched["comp_gain"] = float(method_map.get("compensated", float("nan")) - method_map.get("none", float("nan")))
        enriched["oracle_gap_to_comp"] = float(method_map.get("oracle", float("nan")) - method_map.get("compensated", float("nan")))
        enriched_rows.append(enriched)

    overview_rows = _build_overview_rows(enriched_rows)

    write_csv(out_dir / "mismatch_instancewise.csv", instance_rows)
    write_csv(out_dir / "mismatch_metrics.csv", enriched_rows)
    write_csv(out_dir / "mismatch_overview.csv", overview_rows)

    print(f"output_dir={out_dir}")
    print(f"instancewise_csv={out_dir / 'mismatch_instancewise.csv'}")
    print(f"metrics_csv={out_dir / 'mismatch_metrics.csv'}")
    print(f"overview_csv={out_dir / 'mismatch_overview.csv'}")


if __name__ == "__main__":
    main()
