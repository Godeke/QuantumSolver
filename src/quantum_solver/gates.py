"""Definitions for common quantum gates and their application helpers."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


Matrix = Tuple[Tuple[complex, ...], ...]


def _is_unitary(matrix: Matrix, *, tolerance: float = 1e-9) -> bool:
    size = len(matrix)
    for row in matrix:
        if len(row) != size:
            return False
    for i in range(size):
        for j in range(size):
            total = 0j
            for k in range(size):
                total += matrix[i][k] * matrix[j][k].conjugate()
            if i == j:
                if not math.isclose(total.real, 1.0, rel_tol=tolerance, abs_tol=tolerance):
                    return False
                if not math.isclose(total.imag, 0.0, abs_tol=tolerance):
                    return False
            else:
                if not math.isclose(total.real, 0.0, abs_tol=tolerance):
                    return False
                if not math.isclose(total.imag, 0.0, abs_tol=tolerance):
                    return False
    return True


@dataclass(frozen=True)
class Gate:
    """Unitary matrix describing a quantum gate."""

    name: str
    matrix: Matrix
    num_qubits: int

    def __post_init__(self) -> None:
        expected = 2**self.num_qubits
        if len(self.matrix) != expected:
            raise ValueError(
                f"Gate {self.name} expects {expected} rows, received {len(self.matrix)} instead."
            )
        if not all(len(row) == expected for row in self.matrix):
            raise ValueError(f"Gate {self.name} matrix must be square.")
        if not _is_unitary(self.matrix):
            raise ValueError(f"Gate {self.name} matrix is not unitary within tolerance.")


@dataclass(frozen=True)
class GateOperation:
    """Concrete placement of a gate over a set of qubits."""

    gate: Gate
    targets: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.targets) != self.gate.num_qubits:
            raise ValueError(
                f"Gate {self.gate.name} expects {self.gate.num_qubits} targets, "
                f"received {len(self.targets)}."
            )
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("Target qubits must be unique for a gate operation.")

    def apply(self, state: Tuple[complex, ...], num_qubits: int) -> Tuple[complex, ...]:
        """Apply the gate operation to a flat state vector."""

        if len(state) != 2**num_qubits:
            raise ValueError(
                f"State length {len(state)} does not match expected dimension 2**{num_qubits}."
            )

        return apply_gate_matrix(state, self.gate.matrix, self.targets, num_qubits)

    def describe(self) -> str:
        if self.gate.num_qubits == 1:
            return f"{self.gate.name} q{self.targets[0]}"
        if self.gate.name.upper() == "CNOT":
            return f"CNOT q{self.targets[0]}->q{self.targets[1]}"
        target_list = ",".join(f"q{q}" for q in self.targets)
        return f"{self.gate.name} ({target_list})"

    def __str__(self) -> str:  # pragma: no cover - convenience wrapper
        return self.describe()


def apply_gate_matrix(
    state: Tuple[complex, ...], gate_matrix: Matrix, targets: Sequence[int], num_qubits: int
) -> Tuple[complex, ...]:
    """Apply a gate matrix to the provided state vector."""

    targets = tuple(targets)
    if any(q < 0 or q >= num_qubits for q in targets):
        raise ValueError(f"Targets {targets} are invalid for {num_qubits} qubits.")

    dimension = 1 << num_qubits
    result = [0j] * dimension
    target_mask = 0
    for qubit in targets:
        target_mask |= 1 << qubit

    block_size = 1 << len(targets)
    reversed_targets = tuple(reversed(targets))

    for base_index in range(dimension):
        if base_index & target_mask:
            continue
        indices = []
        for pattern in range(block_size):
            idx = base_index
            for offset, qubit in enumerate(reversed_targets):
                if (pattern >> offset) & 1:
                    idx |= 1 << qubit
            indices.append(idx)

        vector = [state[idx] for idx in indices]
        transformed = []
        for row in gate_matrix:
            total = 0j
            for coefficient, amplitude in zip(row, vector):
                total += coefficient * amplitude
            transformed.append(total)

        for idx, value in zip(indices, transformed):
            result[idx] = value

    return tuple(result)


def _create_single_qubit_gate(name: str, matrix: Iterable[Iterable[complex]]) -> Gate:
    mat = tuple(tuple(complex(value) for value in row) for row in matrix)
    return Gate(name=name, matrix=mat, num_qubits=1)


def _create_two_qubit_gate(name: str, matrix: Iterable[Iterable[complex]]) -> Gate:
    mat = tuple(tuple(complex(value) for value in row) for row in matrix)
    return Gate(name=name, matrix=mat, num_qubits=2)


SQRT_HALF = 1.0 / math.sqrt(2.0)

X_GATE = _create_single_qubit_gate(
    "X",
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ],
)

Y_GATE = _create_single_qubit_gate(
    "Y",
    [
        [0.0, -1.0j],
        [1.0j, 0.0],
    ],
)

Z_GATE = _create_single_qubit_gate(
    "Z",
    [
        [1.0, 0.0],
        [0.0, -1.0],
    ],
)

H_GATE = _create_single_qubit_gate(
    "H",
    [
        [SQRT_HALF, SQRT_HALF],
        [SQRT_HALF, -SQRT_HALF],
    ],
)

S_GATE = _create_single_qubit_gate(
    "S",
    [
        [1.0, 0.0],
        [0.0, 1.0j],
    ],
)

T_GATE = _create_single_qubit_gate(
    "T",
    [
        [1.0, 0.0],
        [0.0, cmath.exp(1.0j * math.pi / 4.0)],
    ],
)

ID_GATE = _create_single_qubit_gate(
    "I",
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ],
)

CNOT_GATE = _create_two_qubit_gate(
    "CNOT",
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
)

SUPPORTED_GATES = {
    gate.name: gate
    for gate in (X_GATE, Y_GATE, Z_GATE, H_GATE, S_GATE, T_GATE, ID_GATE, CNOT_GATE)
}
