from __future__ import annotations

from collections.abc import Iterable

from qiskit_aer.noise import NoiseModel

SINGLE_QUBIT_BASIS_GATES: tuple[str, ...] = (
    "id",
    "rz",
    "sx",
    "x",
    "u",
    "u1",
    "u2",
    "u3",
    "rx",
    "ry",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "p",
)

TWO_QUBIT_BASIS_GATES: tuple[str, ...] = (
    "cx",
    "cz",
    "ecr",
    "swap",
    "rzz",
)


def _existing_basis_gates(model: NoiseModel, candidate_gates: Iterable[str]) -> list[str]:
    basis = set(model.basis_gates)
    return [gate for gate in candidate_gates if gate in basis]
