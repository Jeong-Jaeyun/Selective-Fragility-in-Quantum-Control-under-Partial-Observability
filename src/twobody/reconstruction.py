from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.twobody.observables import ObservableSpec, coerce_observable_specs


def transverse_pauli_count(pauli: str) -> int:
    return sum(1 for axis in pauli if axis in {"X", "Y"})


def dephasing_attenuation(pauli: str, gamma: float, eps: float = 1e-8) -> float:
    gamma_clipped = min(max(float(gamma), 0.0), 1.0)
    base = max(1.0 - gamma_clipped, eps)
    return float(base ** (0.5 * transverse_pauli_count(pauli)))


def compensate_expectations(
    expectations: dict[str, float],
    observables: Sequence[ObservableSpec] | None,
    *,
    phi: float,
    gamma: float,
    eps: float = 1e-8,
) -> dict[str, float]:
    specs = coerce_observable_specs(observables, 2)
    pauli_to_name = {spec.pauli: spec.name for spec in specs}

    corrected: dict[str, float] = {}
    for spec in specs:
        attenuation = dephasing_attenuation(spec.pauli, gamma, eps=eps)
        corrected[spec.name] = float(expectations[spec.name] / attenuation)

    cos_phi = float(np.cos(phi))
    sin_phi = float(np.sin(phi))

    # Undo the local frame rotation on qubit 0 for each fixed suffix on the second qubit.
    grouped_suffixes = sorted({spec.pauli[1:] for spec in specs if spec.pauli[0] in {"X", "Y"}})
    for suffix in grouped_suffixes:
        x_pauli = f"X{suffix}"
        y_pauli = f"Y{suffix}"
        if x_pauli not in pauli_to_name or y_pauli not in pauli_to_name:
            continue
        x_name = pauli_to_name[x_pauli]
        y_name = pauli_to_name[y_pauli]
        x_obs = corrected[x_name]
        y_obs = corrected[y_name]
        corrected[x_name] = float(cos_phi * x_obs + sin_phi * y_obs)
        corrected[y_name] = float(-sin_phi * x_obs + cos_phi * y_obs)

    return corrected


def observable_mae(
    clean_expectations: dict[str, float],
    target_expectations: dict[str, float],
    observables: Sequence[ObservableSpec] | None = None,
) -> float:
    if observables is None:
        names = sorted(clean_expectations)
    else:
        names = [spec.name for spec in observables]
    return float(np.mean([abs(float(target_expectations[name]) - float(clean_expectations[name])) for name in names]))


def observable_mae_by_family(
    clean_expectations: dict[str, float],
    target_expectations: dict[str, float],
    observables: Sequence[ObservableSpec],
) -> dict[str, float]:
    grouped_names: dict[str, list[str]] = defaultdict(list)
    for spec in observables:
        family = "correlator" if len(spec.pauli) == 2 and transverse_pauli_count(spec.pauli) >= 2 else "local"
        grouped_names[family].append(spec.name)

    out: dict[str, float] = {}
    for family, names in grouped_names.items():
        out[f"{family}_mae"] = float(
            np.mean([abs(float(target_expectations[name]) - float(clean_expectations[name])) for name in names])
        )
    out["all_mae"] = observable_mae(clean_expectations, target_expectations, observables)
    return out


def summarize_reconstruction_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = (str(row["backend_type"]), str(row["method"]), int(row["shots"]))
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend_type, method, shots), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "backend_type": backend_type,
                "method": method,
                "shots": shots,
                "n_records": len(group),
                "all_mae_mean": float(np.mean([float(row["all_mae"]) for row in group])),
                "all_mae_std": float(np.std([float(row["all_mae"]) for row in group])),
                "local_mae_mean": float(np.mean([float(row["local_mae"]) for row in group])),
                "correlator_mae_mean": float(np.mean([float(row["correlator_mae"]) for row in group])),
            }
        )
    return summary_rows
