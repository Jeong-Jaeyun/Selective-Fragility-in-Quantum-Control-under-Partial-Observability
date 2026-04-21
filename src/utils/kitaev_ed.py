from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

from src.utils.quantum import I2, X, Y, Z

SIGMA_PLUS = (X + 1j * Y) * 0.5
SIGMA_MINUS = (X - 1j * Y) * 0.5


def _kron_all(ops: Iterable[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for op in ops:
        out = np.kron(out, np.asarray(op, dtype=np.complex128))
    return out


def pauli_string_operator(paulis: str) -> np.ndarray:
    op_map = {"I": I2, "X": X, "Y": Y, "Z": Z}
    return _kron_all(op_map[c] for c in paulis)


@lru_cache(maxsize=8)
def jw_c(n_sites: int, j_site: int) -> np.ndarray:
    if not (0 <= j_site < n_sites):
        raise ValueError(f"site index out of range: j={j_site}, n={n_sites}")
    factors = []
    for idx in range(n_sites):
        if idx < j_site:
            factors.append(Z)
        elif idx == j_site:
            factors.append(SIGMA_PLUS)
        else:
            factors.append(I2)
    return _kron_all(factors)


@lru_cache(maxsize=8)
def jw_cdag(n_sites: int, j_site: int) -> np.ndarray:
    return jw_c(n_sites, j_site).conjugate().T


@lru_cache(maxsize=4)
def _kitaev_templates(n_sites: int) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2 for Kitaev chain")
    c_ops = tuple(jw_c(n_sites, j) for j in range(n_sites))
    cdag_ops = tuple(c.conjugate().T for c in c_ops)
    dim = 1 << n_sites
    eye = np.eye(dim, dtype=np.complex128)

    hop_terms = []
    pair_terms = []
    number_terms = []
    for j in range(n_sites - 1):
        hop_terms.append(cdag_ops[j] @ c_ops[j + 1])
        pair_terms.append(c_ops[j] @ c_ops[j + 1])
    for j in range(n_sites):
        number_terms.append(cdag_ops[j] @ c_ops[j])

    return eye, tuple(hop_terms), tuple(pair_terms), tuple(number_terms)


def build_kitaev_hamiltonian(n_sites: int, mu: float, t_hop: float, delta: float) -> np.ndarray:
    eye, hop_terms, pair_terms, number_terms = _kitaev_templates(n_sites)
    h = np.zeros_like(eye)

    for hop in hop_terms:
        h = h - t_hop * hop - np.conjugate(t_hop) * hop.conjugate().T
    for pair in pair_terms:
        h = h - delta * pair - np.conjugate(delta) * pair.conjugate().T
    for number in number_terms:
        h = h - mu * (number - 0.5 * eye)

    return 0.5 * (h + h.conjugate().T)


def ground_state_density(h: np.ndarray) -> tuple[np.ndarray, float]:
    eigvals, eigvecs = np.linalg.eigh(h)
    idx = int(np.argmin(eigvals))
    psi0 = eigvecs[:, idx]
    rho = np.outer(psi0, psi0.conjugate())
    return rho, float(np.real(eigvals[idx]))


@lru_cache(maxsize=4)
def feature_operators(n_sites: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if n_sites < 2:
        raise ValueError("n_sites must be at least 2 for feature operators")
    c_left = jw_c(n_sites, 0)
    c_right = jw_c(n_sites, n_sites - 1)
    cdag_left = c_left.conjugate().T
    cdag_right = c_right.conjugate().T

    gamma_1 = c_left + cdag_left
    gamma_2n = -1j * (c_right - cdag_right)
    edge_majorana = 1j * (gamma_1 @ gamma_2n)
    edge_majorana = 0.5 * (edge_majorana + edge_majorana.conjugate().T)

    parity = pauli_string_operator("Z" * n_sites)
    string_body = "Z" * max(0, n_sites - 2)
    jw_string = pauli_string_operator(f"X{string_body}X")

    return ("global_parity", "edge_majorana", "jw_string"), parity, edge_majorana, jw_string


def expectations(rho: np.ndarray, operators: Iterable[np.ndarray]) -> np.ndarray:
    vals = []
    for op in operators:
        exp_val = float(np.real(np.trace(rho @ op)))
        vals.append(float(np.clip(exp_val, -1.0, 1.0)))
    return np.asarray(vals, dtype=float)


def map_to_unit_interval(exp_vals: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(exp_vals, dtype=float) + 1.0) * 0.5, 0.0, 1.0)
