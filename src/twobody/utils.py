from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeVar

import numpy as np

from src.twobody.types import BackendConfig, NoiseConfig, SystemConfig

T = TypeVar("T", BackendConfig, NoiseConfig, SystemConfig)


def dataclass_from_mapping(cls: type[T], mapping: T | dict[str, Any] | None) -> T:
    if mapping is None:
        return cls()  # type: ignore[call-arg]
    if isinstance(mapping, cls):
        return mapping
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in mapping.items() if key in allowed}
    return cls(**kwargs)  # type: ignore[arg-type]


def coerce_system_config(cfg: SystemConfig | dict[str, Any] | None) -> SystemConfig:
    return dataclass_from_mapping(SystemConfig, cfg)


def coerce_backend_config(cfg: BackendConfig | dict[str, Any] | None) -> BackendConfig:
    return dataclass_from_mapping(BackendConfig, cfg)


def coerce_noise_config(cfg: NoiseConfig | dict[str, Any] | None) -> NoiseConfig:
    return dataclass_from_mapping(NoiseConfig, cfg)


def rng_from_seed(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def validate_two_qubit_system(cfg: SystemConfig) -> None:
    if cfg.n_qubits != 2:
        raise ValueError(f"two-body phase-1 modules currently support n_qubits=2 only, got {cfg.n_qubits}")
