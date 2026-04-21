from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

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

DEFAULT_STRENGTHS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Layer B random-mixture ensemble.")
    parser.add_argument("--qubits", nargs="*", type=int, default=[2, 3, 4])
    parser.add_argument("--mixtures", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=20260421)
    parser.add_argument("--strength-values", nargs="*", type=float, default=DEFAULT_STRENGTHS)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default="twobody_random_mixtures")
    parser.add_argument("--no-progress", action="store_true")
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


def _generate_mixtures(count: int, seed: int) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(int(seed))
    mixtures: list[dict[str, float | str]] = []
    for idx in range(count):
        w_phi, w_gamma, w_eta, w_pm = rng.dirichlet(np.ones(4))
        phi_scale = 0.3 + 0.9 * float(w_phi)
        gamma_scale = 0.1 + 0.4 * float(w_gamma)
        eta_scale = 0.05 + 0.4 * float(w_eta)
        p_measurement_scale = 0.02 + 0.25 * float(w_pm)
        mixtures.append(
            {
                "mixture_id": f"M{idx + 1:02d}",
                "w_phi": float(w_phi),
                "w_gamma": float(w_gamma),
                "w_eta": float(w_eta),
                "w_pm": float(w_pm),
                "phi_scale": float(phi_scale),
                "gamma_scale": float(gamma_scale),
                "eta_scale": float(eta_scale),
                "p_measurement_scale": float(p_measurement_scale),
                "in_subspace_weight": float(w_phi + w_gamma),
                "out_of_subspace_weight": float(w_eta + w_pm),
                "in_subspace_scale": float(phi_scale + gamma_scale),
                "out_of_subspace_scale": float(eta_scale + p_measurement_scale),
            }
        )
    return mixtures


def _first_below(rows: list[dict[str, Any]], metric: str, threshold: float) -> float:
    for row in sorted(rows, key=lambda item: _float(item, "noise_level")):
        if _float(row, metric) < threshold:
            return _float(row, "noise_level")
    return float("nan")


def _value_at(rows: list[dict[str, Any]], strength: float, metric: str) -> float:
    for row in rows:
        if abs(_float(row, "noise_level") - float(strength)) < 1e-12:
            return _float(row, metric)
    return float("nan")


def _build_summary(records: list[dict[str, Any]], strength_max: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["n_qubits"]), str(row["mixture_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (n_qubits, mixture_id), group in sorted(grouped.items()):
        summary = summarize_transition_records(group)
        first = group[0]
        for row in summary:
            row["n_qubits"] = int(n_qubits)
            row["mixture_id"] = mixture_id
            for key in (
                "w_phi",
                "w_gamma",
                "w_eta",
                "w_pm",
                "phi_scale",
                "gamma_scale",
                "eta_scale",
                "p_measurement_scale",
                "in_subspace_weight",
                "out_of_subspace_weight",
                "in_subspace_scale",
                "out_of_subspace_scale",
            ):
                row[key] = first[key]
            row["noise_fraction"] = _float(row, "noise_level") / max(strength_max, 1e-8)
            output.append(row)
    return output


def _build_mixture_table(summary_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(int(row["n_qubits"]), str(row["mixture_id"]))].append(row)

    output: list[dict[str, Any]] = []
    for (n_qubits, mixture_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: _float(item, "noise_level"))
        first = ordered[0]
        tau_none = _first_below(ordered, "classification_none", threshold)
        tau_comp = _first_below(ordered, "classification_compensated", threshold)
        output.append(
            {
                "n_qubits": int(n_qubits),
                "mixture_id": mixture_id,
                "w_phi": first["w_phi"],
                "w_gamma": first["w_gamma"],
                "w_eta": first["w_eta"],
                "w_pm": first["w_pm"],
                "phi_scale": first["phi_scale"],
                "gamma_scale": first["gamma_scale"],
                "eta_scale": first["eta_scale"],
                "p_measurement_scale": first["p_measurement_scale"],
                "in_subspace_weight": first["in_subspace_weight"],
                "out_of_subspace_weight": first["out_of_subspace_weight"],
                "in_subspace_scale": first["in_subspace_scale"],
                "out_of_subspace_scale": first["out_of_subspace_scale"],
                "tau_none": tau_none,
                "tau_comp": tau_comp,
                "delta_tau": tau_comp - tau_none if np.isfinite(tau_none) and np.isfinite(tau_comp) else float("nan"),
                "tau_comp_4q": tau_comp if int(n_qubits) == 4 else float("nan"),
                "full_gap_s015": _value_at(ordered, 0.15, "full_oracle_gap"),
                "struct_gap_s015": _value_at(ordered, 0.15, "structured_oracle_gap"),
                "comp_ba_s015": _value_at(ordered, 0.15, "classification_compensated"),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir) / f"{args.exp_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    qubit_counts = [int(value) for value in args.qubits]
    strength_values = [float(value) for value in args.strength_values]
    mixtures = _generate_mixtures(int(args.mixtures), int(args.random_seed))

    records: list[dict[str, Any]] = []
    total_evals = 0
    per_qubit_context: dict[int, dict[str, Any]] = {}
    for n_qubits in qubit_counts:
        cfg = load_yaml(CONFIG_MAP[n_qubits])
        exp_cfg = cfg.get("experiment", {})
        backend_type = str(list(exp_cfg.get("backend_types", ["shot"]))[0])
        shots = int(list(exp_cfg.get("shot_list", [8192]))[0]) if backend_type != "density" else 0
        test_seeds = [int(seed) for seed in list(exp_cfg.get("test_seeds", [101, 103, 107]))]
        total_evals += len(mixtures) * len(strength_values) * len(test_seeds)
        per_qubit_context[n_qubits] = {
            "cfg": cfg,
            "backend_type": backend_type,
            "shots": shots,
            "test_seeds": test_seeds,
        }

    progress = None
    if (not args.no_progress) and tqdm is not None:
        progress = tqdm(total=total_evals, desc="random mixtures", dynamic_ncols=True, file=sys.stdout)

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

            for mixture in mixtures:
                _status(f"[mixture] {n_qubits}Q {mixture['mixture_id']}", progress=progress)
                for seed in test_seeds:
                    _status(f"[seed] {n_qubits}Q {mixture['mixture_id']} seed={int(seed)}", progress=progress)
                    for strength in strength_values:
                        noise_cfg = NoiseConfig(**base_noise_cfg.to_dict())
                        noise_cfg.phi = float(mixture["phi_scale"]) * float(strength)
                        noise_cfg.gamma_dephasing = float(mixture["gamma_scale"]) * float(strength)
                        noise_cfg.eta_amplitude = float(mixture["eta_scale"]) * float(strength)
                        noise_cfg.p_measurement = float(mixture["p_measurement_scale"]) * float(strength)
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
                                "mixture_id": mixture["mixture_id"],
                                "seed": int(seed),
                                "backend_type": backend_type,
                                "shots": int(shots),
                                "noise_axis": "composite_strength",
                                "noise_level": float(strength),
                                **mixture,
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
    mixture_table = _build_mixture_table(summary_rows, threshold=float(args.threshold))

    write_csv(out_dir / "layerB_random_mixture_records.csv", records)
    write_csv(out_dir / "layerB_random_mixture_summary.csv", summary_rows)
    write_csv(out_dir / "layerB_random_mixture_table.csv", mixture_table)

    print(f"output_dir={out_dir}")
    print(f"records_csv={out_dir / 'layerB_random_mixture_records.csv'}")
    print(f"summary_csv={out_dir / 'layerB_random_mixture_summary.csv'}")
    print(f"mixture_table_csv={out_dir / 'layerB_random_mixture_table.csv'}")


if __name__ == "__main__":
    main()
