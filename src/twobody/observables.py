from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from src.twobody.types import ObservableSpec


def _pauli_label(n_qubits: int, axes: dict[int, str]) -> str:
    label = ["I"] * n_qubits
    for qubit, axis in axes.items():
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(f"qubit index {qubit} is outside n_qubits={n_qubits}")
        label[int(qubit)] = str(axis)
    return "".join(label)


def _local_name(axis: str, qubit: int) -> str:
    return f"{axis}{qubit + 1}"


def _pair_name(left_axis: str, left_qubit: int, right_axis: str, right_qubit: int) -> str:
    return f"{left_axis}{left_qubit + 1}{right_axis}{right_qubit + 1}"


def get_observable_specs(n_qubits: int, include_cross_terms: bool = False) -> list[ObservableSpec]:
    if n_qubits < 2:
        raise ValueError(f"observable set requires n_qubits >= 2, got {n_qubits}")

    specs: list[ObservableSpec] = []
    for axis in ("Z", "X"):
        for qubit in range(n_qubits):
            specs.append(
                ObservableSpec(
                    name=_local_name(axis, qubit),
                    pauli=_pauli_label(n_qubits, {qubit: axis}),
                )
            )

    for left in range(n_qubits - 1):
        right = left + 1
        for axis in ("Z", "X", "Y"):
            specs.append(
                ObservableSpec(
                    name=_pair_name(axis, left, axis, right),
                    pauli=_pauli_label(n_qubits, {left: axis, right: axis}),
                )
            )

    if include_cross_terms:
        for left in range(n_qubits - 1):
            right = left + 1
            for left_axis, right_axis in (("X", "Y"), ("Y", "X")):
                specs.append(
                    ObservableSpec(
                        name=_pair_name(left_axis, left, right_axis, right),
                        pauli=_pauli_label(n_qubits, {left: left_axis, right: right_axis}),
                    )
                )
    return specs


def observable_to_pauli_op(spec: ObservableSpec) -> SparsePauliOp:
    return SparsePauliOp.from_list([(spec.pauli, 1.0)])


def build_measurement_circuit_for_pauli(qc: QuantumCircuit, pauli_str: str) -> QuantumCircuit:
    if len(pauli_str) != qc.num_qubits:
        raise ValueError(f"pauli string length {len(pauli_str)} does not match num_qubits {qc.num_qubits}")

    measured = QuantumCircuit(qc.num_qubits, qc.num_qubits, name=f"meas:{pauli_str}")
    measured.compose(qc, inplace=True)
    for qubit, pauli in enumerate(pauli_str):
        if pauli == "X":
            measured.h(qubit)
        elif pauli == "Y":
            measured.sdg(qubit)
            measured.h(qubit)
        elif pauli in {"Z", "I"}:
            continue
        else:
            raise ValueError(f"unsupported Pauli axis in '{pauli_str}': {pauli}")
    measured.measure(range(qc.num_qubits), range(qc.num_qubits))
    return measured


def coerce_observable_specs(observables: Sequence[ObservableSpec] | None, n_qubits: int) -> list[ObservableSpec]:
    return list(observables) if observables is not None else get_observable_specs(n_qubits)
