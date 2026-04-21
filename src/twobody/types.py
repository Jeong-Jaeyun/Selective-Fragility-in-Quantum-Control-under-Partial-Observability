from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SystemConfig:
    n_qubits: int = 2
    state_family: str = "zero"
    hamiltonian_type: str = "xx_zz"
    evolution_time: float = 1.0
    trotter_steps: int = 1
    evolution_method: str = "exact"
    jx: float = 1.0
    jy: float = 0.0
    jz: float = 0.5
    hx: float = 0.0
    hz: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NoiseConfig:
    phi: float = 0.0
    phi_corr: float = 0.0
    gamma_dephasing: float = 0.0
    eta_amplitude: float = 0.0
    p_depolarizing: float = 0.0
    p_measurement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BackendConfig:
    density_enabled: bool = True
    shot_enabled: bool = False
    shots: int = 2048
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    name: str
    pauli: str


@dataclass(slots=True)
class DensityExperimentResult:
    density_matrix: np.ndarray
    expectations: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

