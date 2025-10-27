"""Search-based solver to synthesise quantum gate sequences."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .gates import SUPPORTED_GATES, Gate, GateOperation
from .state import QuantumState


AmplitudeKey = Tuple[int, ...]


@dataclass
class SolverResult:
    success: bool
    sequence: List[GateOperation]
    layers_used: int
    final_state: QuantumState
    distance: float
    states: List[QuantumState]

    def as_gate_names(self) -> List[str]:
        return [op.describe() for op in self.sequence]


class GateSequenceSolver:
    """Explore sequences of quantum gates to reach a target state."""

    def __init__(
        self,
        num_qubits: int,
        *,
        allowed_gates: Optional[Iterable[str]] = None,
        tolerance: float = 1e-6,
        quantisation_decimals: int = 8,
        fixed_operations: Optional[Dict[int, GateOperation]] = None,
    ) -> None:
        if num_qubits <= 0:
            raise ValueError("Solver must operate on at least one qubit.")
        self.num_qubits = num_qubits
        self.tolerance = tolerance
        self.quantisation_decimals = quantisation_decimals
        self._quantisation_scale = 10**quantisation_decimals

        gate_symbols = allowed_gates if allowed_gates is not None else SUPPORTED_GATES.keys()
        gates: List[Gate] = []
        for symbol in gate_symbols:
            if symbol not in SUPPORTED_GATES:
                raise ValueError(f"Gate '{symbol}' is not supported.")
            gates.append(SUPPORTED_GATES[symbol])
        self.gates = gates
        self.operations = self._build_operations()
        self.fixed_operations: Dict[int, GateOperation] = (
            dict(fixed_operations) if fixed_operations is not None else {}
        )
        for layer, operation in self.fixed_operations.items():
            if layer < 0:
                raise ValueError("Fixed operation layers must be non-negative indices.")
            for target in operation.targets:
                if target < 0 or target >= self.num_qubits:
                    raise ValueError(
                        f"Fixed operation '{operation.describe()}' targets invalid qubit {target} "
                        f"for {self.num_qubits}-qubit solver."
                    )
        self._fixed_layers = sorted(self.fixed_operations.keys())

    def _build_operations(self) -> List[GateOperation]:
        operations: List[GateOperation] = []
        for gate in self.gates:
            if gate.num_qubits == 1:
                for target in range(self.num_qubits):
                    operations.append(GateOperation(gate=gate, targets=(target,)))
            elif gate.num_qubits == 2:
                if self.num_qubits < 2:
                    continue
                for control in range(self.num_qubits):
                    for target in range(self.num_qubits):
                        if control == target:
                            continue
                        operations.append(GateOperation(gate=gate, targets=(control, target)))
            else:
                raise NotImplementedError(
                    f"Gate {gate.name} with arity {gate.num_qubits} is not supported in solver."
                )
        return operations

    def _state_key(self, amplitudes: Tuple[complex, ...]) -> AmplitudeKey:
        scale = self._quantisation_scale
        key_parts: List[int] = []
        for amplitude in amplitudes:
            key_parts.append(int(round(amplitude.real * scale)))
            key_parts.append(int(round(amplitude.imag * scale)))
        return tuple(key_parts)

    def _operations_for_depth(self, depth: int) -> Sequence[GateOperation]:
        fixed_operation = self.fixed_operations.get(depth)
        if fixed_operation is not None:
            return (fixed_operation,)
        return self.operations

    def _fixed_layers_satisfied(self, sequence_length: int) -> bool:
        if not self._fixed_layers:
            return True
        last_applied_index = sequence_length - 1
        for layer in self._fixed_layers:
            if layer > last_applied_index:
                return False
        return True

    def _evolve_states(
        self, start: QuantumState, sequence: Sequence[GateOperation]
    ) -> List[QuantumState]:
        states: List[QuantumState] = []
        current = start
        for operation in sequence:
            current = current.apply(operation)
            states.append(current)
        return states

    def solve(
        self,
        start: QuantumState,
        target: QuantumState,
        *,
        max_layers: int,
    ) -> SolverResult:
        if start.num_qubits != self.num_qubits or target.num_qubits != self.num_qubits:
            raise ValueError("Solver and states disagree on qubit count.")

        if self._fixed_layers and max(self._fixed_layers) >= max_layers:
            raise ValueError(
                "A fixed gate is defined beyond the configured maximum number of layers."
            )

        initial_distance = start.distance(target)
        if initial_distance <= self.tolerance and not self._fixed_layers:
            return SolverResult(
                success=True,
                sequence=[],
                layers_used=0,
                final_state=start.copy(),
                distance=initial_distance,
                states=[],
            )

        frontier = deque([(start.amplitudes, [])])
        visited: Dict[AmplitudeKey, int] = {}
        visited[self._state_key(start.amplitudes)] = 0
        best_state = start.copy()
        best_sequence: List[GateOperation] = []
        best_distance = initial_distance

        while frontier:
            state_vector, sequence = frontier.popleft()
            depth = len(sequence)
            if depth >= max_layers:
                continue

            for operation in self._operations_for_depth(depth):
                new_vector = operation.apply(state_vector, self.num_qubits)
                new_state = QuantumState.from_amplitudes(new_vector, normalise=True)
                new_distance = new_state.distance(target)
                new_sequence = sequence + [operation]
                if new_distance < best_distance - self.tolerance * 0.1:
                    best_distance = new_distance
                    best_state = new_state
                    best_sequence = list(new_sequence)

                if new_distance <= self.tolerance and self._fixed_layers_satisfied(
                    len(new_sequence)
                ):
                    states = self._evolve_states(start, new_sequence)
                    return SolverResult(
                        success=True,
                        sequence=new_sequence,
                        layers_used=len(new_sequence),
                        final_state=states[-1] if states else start.copy(),
                        distance=new_distance,
                        states=states,
                    )

                key = self._state_key(new_state.amplitudes)
                recorded_depth = visited.get(key)
                if recorded_depth is None or len(new_sequence) < recorded_depth:
                    visited[key] = len(new_sequence)
                    frontier.append((new_state.amplitudes, new_sequence))

        states = self._evolve_states(start, best_sequence)
        final_state = states[-1] if states else best_state

        return SolverResult(
            success=False,
            sequence=best_sequence,
            layers_used=len(best_sequence),
            final_state=final_state,
            distance=best_distance,
            states=states,
        )
