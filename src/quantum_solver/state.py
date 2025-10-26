"""Quantum state representation and helper utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from .gates import GateOperation


AmplitudeVector = Tuple[complex, ...]


def _validate_dimension(length: int) -> int:
    if length == 0:
        raise ValueError("Quantum state must contain at least one amplitude.")
    num_qubits = int(math.log2(length))
    if 2**num_qubits != length:
        raise ValueError(
            f"State vector length {length} is not a power of two and cannot represent qubits."
        )
    return num_qubits


def _normalise(amplitudes: AmplitudeVector) -> AmplitudeVector:
    norm_squared = sum(abs(value) ** 2 for value in amplitudes)
    if math.isclose(norm_squared, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return amplitudes
    if math.isclose(norm_squared, 0.0, abs_tol=1e-12):
        raise ValueError("Cannot normalise the zero vector.")
    scale = math.sqrt(norm_squared)
    return tuple(value / scale for value in amplitudes)


def amplitudes_from_components(components: Sequence[Sequence[float]]) -> AmplitudeVector:
    """Build a complex array from a sequence of (real, imag) pairs."""

    data = []
    for index, pair in enumerate(components):
        if len(pair) != 2:
            raise ValueError(
                f"Amplitude at index {index} must have real and imaginary components."
            )
        real, imag = pair
        data.append(complex(real, imag))
    return tuple(data)


@dataclass(frozen=True)
class QuantumState:
    """Immutable wrapper for an n-qubit state vector."""

    amplitudes: AmplitudeVector
    num_qubits: int

    @classmethod
    def from_amplitudes(
        cls, amplitudes: Iterable[complex], *, normalise: bool = True
    ) -> "QuantumState":
        vector = tuple(complex(value) for value in amplitudes)
        num_qubits = _validate_dimension(len(vector))
        if normalise:
            vector = _normalise(vector)
        return cls(amplitudes=vector, num_qubits=num_qubits)

    @classmethod
    def from_real_imag_pairs(
        cls, components: Sequence[Sequence[float]], *, normalise: bool = True
    ) -> "QuantumState":
        vector = amplitudes_from_components(components)
        return cls.from_amplitudes(vector, normalise=normalise)

    def distance(self, other: "QuantumState") -> float:
        if self.num_qubits != other.num_qubits:
            raise ValueError("Cannot compare states with different qubit counts.")
        diff = [self.amplitudes[i] - other.amplitudes[i] for i in range(len(self.amplitudes))]
        return math.sqrt(sum(abs(value) ** 2 for value in diff))

    def apply(self, operation: GateOperation) -> "QuantumState":
        new_state = operation.apply(self.amplitudes, self.num_qubits)
        return QuantumState(amplitudes=new_state, num_qubits=self.num_qubits)

    def as_probability_distribution(self) -> Tuple[float, ...]:
        return tuple(abs(value) ** 2 for value in self.amplitudes)

    def copy(self) -> "QuantumState":
        return QuantumState(amplitudes=tuple(self.amplitudes), num_qubits=self.num_qubits)

