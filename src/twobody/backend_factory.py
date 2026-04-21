from __future__ import annotations

from typing import Any

from qiskit_aer import AerSimulator

from src.twobody.types import BackendConfig
from src.twobody.utils import coerce_backend_config


def get_density_backend(seed: int | None = None, noise_model: Any | None = None) -> AerSimulator:
    kwargs: dict[str, Any] = {"method": "density_matrix"}
    if seed is not None:
        kwargs["seed_simulator"] = int(seed)
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    return AerSimulator(**kwargs)


def get_shot_backend(
    seed: int | None = None,
    shots: int | None = None,
    noise_model: Any | None = None,
) -> AerSimulator:
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed_simulator"] = int(seed)
    if noise_model is not None:
        kwargs["noise_model"] = noise_model
    backend = AerSimulator(**kwargs)
    if shots is not None:
        backend.set_options(shots=int(shots))
    return backend


def backend_from_config(cfg: BackendConfig | dict[str, Any] | None) -> AerSimulator:
    backend_cfg = coerce_backend_config(cfg)
    if backend_cfg.density_enabled:
        return get_density_backend(seed=backend_cfg.seed)
    return get_shot_backend(seed=backend_cfg.seed, shots=backend_cfg.shots)
