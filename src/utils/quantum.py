from __future__ import annotations

from functools import lru_cache

import numpy as np

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kron2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)


def two_qubit_z_unitary(phi_c: float, phi_d: float) -> np.ndarray:
    diag = np.array(
        [
            np.exp(-0.5j * (2.0 * phi_c)),
            np.exp(-0.5j * (2.0 * phi_d)),
            np.exp(-0.5j * (-2.0 * phi_d)),
            np.exp(-0.5j * (-2.0 * phi_c)),
        ],
        dtype=np.complex128,
    )
    return np.diag(diag)


def dephase_qubit(rho: np.ndarray, p: float, qubit: int) -> np.ndarray:
    if qubit == 0:
        z_op = kron2(Z, I2)
    elif qubit == 1:
        z_op = kron2(I2, Z)
    else:
        raise ValueError("qubit index for 2-qubit state must be 0 or 1")
    return (1.0 - p) * rho + p * (z_op @ rho @ z_op)


def apply_two_qubit_dephasing(rho: np.ndarray, p: float) -> np.ndarray:
    rho_1 = dephase_qubit(rho, p, qubit=0)
    return dephase_qubit(rho_1, p, qubit=1)


def infer_n_qubits_from_rho(rho: np.ndarray) -> int:
    dim = int(rho.shape[0])
    n_qubits = int(round(np.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError(f"rho dimension {dim} is not a power of two")
    return n_qubits


def global_z_unitary(n_qubits: int, phi: float) -> np.ndarray:
    dim = 1 << n_qubits
    weights = np.asarray([bin(i).count("1") for i in range(dim)], dtype=np.int16)
    z_sums = n_qubits - 2 * weights
    diag = np.exp(-0.5j * phi * z_sums.astype(np.float64))
    return np.diag(diag.astype(np.complex128))


@lru_cache(maxsize=32)
def _z_sign_vector(n_qubits: int, qubit: int) -> np.ndarray:
    if not (0 <= qubit < n_qubits):
        raise ValueError(f"qubit index out of range: qubit={qubit}, n={n_qubits}")
    dim = 1 << n_qubits
    # qubit=0 is the left-most factor in kron order.
    bit_idx = n_qubits - 1 - qubit
    vals = np.empty(dim, dtype=np.float64)
    for state in range(dim):
        vals[state] = 1.0 if ((state >> bit_idx) & 1) == 0 else -1.0
    return vals


def apply_n_qubit_dephasing(rho: np.ndarray, p: float) -> np.ndarray:
    n_qubits = infer_n_qubits_from_rho(rho)
    out = np.asarray(rho, dtype=np.complex128)
    if p <= 0.0:
        return out
    if p >= 1.0:
        p = 1.0
    for qubit in range(n_qubits):
        signs = _z_sign_vector(n_qubits, qubit)
        out = (1.0 - p) * out + p * (signs[:, None] * out * signs[None, :])
    return out
