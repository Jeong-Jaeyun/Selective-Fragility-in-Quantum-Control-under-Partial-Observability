from __future__ import annotations

import argparse
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


CONFIG_MAP = {
    2: "configs/twobody/paper_figures_final.yaml",
    3: "configs/twobody/paper_figures_final_3q.yaml",
    4: "configs/twobody/paper_figures_final_4q.yaml",
}

POINTS = [
    ("G1_in_subspace", "P01", 0.24, 0.12, 0.02, 0.01),
    ("G1_in_subspace", "P02", 0.32, 0.15, 0.03, 0.01),
    ("G1_in_subspace", "P03", 0.40, 0.18, 0.04, 0.02),
    ("G2_amplitude", "P04", 0.20, 0.10, 0.10, 0.02),
    ("G2_amplitude", "P05", 0.20, 0.10, 0.15, 0.02),
    ("G2_amplitude", "P06", 0.25, 0.12, 0.18, 0.03),
    ("G3_readout", "P07", 0.20, 0.10, 0.03, 0.05),
    ("G3_readout", "P08", 0.20, 0.10, 0.03, 0.08),
    ("G3_readout", "P09", 0.25, 0.12, 0.04, 0.10),
    ("G4_mixed_out", "P10", 0.20, 0.10, 0.10, 0.05),
    ("G4_mixed_out", "P11", 0.25, 0.12, 0.12, 0.06),
    ("G4_mixed_out", "P12", 0.30, 0.15, 0.15, 0.08),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer C representative untied-point diagnostics.")
    parser.add_argument("--qubits", nargs="*", type=int, default=[2, 3, 4])
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default="twobody_untied_points")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _status(message: str, *, progress) -> None:
    if progress is not None:
        progress.set_description(message)
    else:
        print(message, flush=True)


def _build_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["n_qubits"]), str(row["point_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (n_qubits, point_id), group in sorted(grouped.items()):
        summary = summarize_transition_records(group)
        if len(summary) != 1:
            raise ValueError(f"expected one summary row for {n_qubits}Q {point_id}, got {len(summary)}")
        row = summary[0]
        first = group[0]
        row["n_qubits"] = int(n_qubits)
        for key in ("point_group", "point_id", "phi_true", "gamma_true", "eta_true", "p_measurement_true"):
            row[key] = first[key]
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir) / f"{args.exp_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    qubit_counts = [int(value) for value in args.qubits]

    records: list[dict[str, Any]] = []
    total_evals = 0
    per_qubit_context: dict[int, dict[str, Any]] = {}
    for n_qubits in qubit_counts:
        cfg = load_yaml(CONFIG_MAP[n_qubits])
        exp_cfg = cfg.get("experiment", {})
        backend_type = str(list(exp_cfg.get("backend_types", ["shot"]))[0])
        shots = int(list(exp_cfg.get("shot_list", [8192]))[0]) if backend_type != "density" else 0
        test_seeds = [int(seed) for seed in list(exp_cfg.get("test_seeds", [101, 103, 107]))]
        total_evals += len(POINTS) * len(test_seeds)
        per_qubit_context[n_qubits] = {
            "cfg": cfg,
            "backend_type": backend_type,
            "shots": shots,
            "test_seeds": test_seeds,
        }

    progress = None
    if (not args.no_progress) and tqdm is not None:
        progress = tqdm(total=total_evals, desc="untied points", dynamic_ncols=True, file=sys.stdout)

    try:
        for n_qubits in qubit_counts:
            context = per_qubit_context[n_qubits]
            cfg = context["cfg"]
            exp_cfg = cfg.get("experiment", {})
            target_system_cfg = SystemConfig(**cfg.get("system", {}))
            probe_system_cfg = SystemConfig(**cfg.get("probe_system", cfg.get("system", {})))
            backend_type = context["backend_type"]
            shots = context["shots"]
            test_seeds = context["test_seeds"]
            train_seeds = [int(seed) for seed in list(exp_cfg.get("train_seeds", [11, 17]))]
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

            for point_group, point_id, phi, gamma, eta, p_measurement in POINTS:
                _status(f"[point] {n_qubits}Q {point_id}", progress=progress)
                noise_cfg = NoiseConfig(
                    phi=float(phi),
                    gamma_dephasing=float(gamma),
                    eta_amplitude=float(eta),
                    p_measurement=float(p_measurement),
                )
                for seed in test_seeds:
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
                            "point_group": point_group,
                            "point_id": point_id,
                            "seed": int(seed),
                            "backend_type": backend_type,
                            "shots": int(shots),
                            "noise_axis": "untied_point",
                            "noise_level": 0.0,
                            "phi_true": float(phi),
                            "gamma_true": float(gamma),
                            "eta_true": float(eta),
                            "p_measurement_true": float(p_measurement),
                            **metrics,
                        }
                    )
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    summary_rows = _build_summary(records)
    write_csv(out_dir / "layerC_untied_point_records.csv", records)
    write_csv(out_dir / "layerC_untied_point_summary.csv", summary_rows)

    print(f"output_dir={out_dir}")
    print(f"records_csv={out_dir / 'layerC_untied_point_records.csv'}")
    print(f"summary_csv={out_dir / 'layerC_untied_point_summary.csv'}")


if __name__ == "__main__":
    main()
