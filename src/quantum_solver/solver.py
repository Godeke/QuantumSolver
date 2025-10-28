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
        layer_gate_allowlists: Optional[Dict[int, Sequence[str]]] = None,
        default_layer_gate_allowlist: Optional[Sequence[str]] = None,
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
        self._operations_by_gate: Dict[str, List[GateOperation]] = {}
        for operation in self.operations:
            self._operations_by_gate.setdefault(operation.gate.name, []).append(operation)
        self._identity_operation: Optional[GateOperation] = None
        identity_candidates = self._operations_by_gate.get("I")
        if identity_candidates:
            self._identity_operation = identity_candidates[0]
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
        gate_names = {gate.name for gate in self.gates}
        self._layer_specific_operations: Dict[int, Tuple[GateOperation, ...]] = {}
        if layer_gate_allowlists is not None:
            for layer, allowed in layer_gate_allowlists.items():
                if layer < 0:
                    raise ValueError("Layer gate constraint indices must be non-negative.")
                allowed_names = tuple(dict.fromkeys(allowed))
                if not allowed_names:
                    raise ValueError(f"Layer gate constraint at layer {layer} must list gates.")
                unknown_gates = [name for name in allowed_names if name not in gate_names]
                if unknown_gates:
                    raise ValueError(
                        f"Layer {layer} constraint references unsupported gates: {unknown_gates}."
                    )
                operations: List[GateOperation] = []
                for name in allowed_names:
                    ops = self._operations_by_gate.get(name)
                    if not ops:
                        raise ValueError(
                            f"Layer {layer} constraint references gate '{name}' "
                            f"which has no operations for {self.num_qubits} qubits."
                        )
                    operations.extend(ops)
                self._layer_specific_operations[layer] = tuple(operations)
        self._default_layer_operations: Optional[Tuple[GateOperation, ...]] = None
        if default_layer_gate_allowlist is not None:
            default_names = tuple(dict.fromkeys(default_layer_gate_allowlist))
            if not default_names:
                raise ValueError("Global gate allowlist must list at least one gate.")
            unknown_default = [name for name in default_names if name not in gate_names]
            if unknown_default:
                raise ValueError(
                    f"Global gate allowlist references unsupported gates: {unknown_default}."
                )
            default_ops: List[GateOperation] = []
            for name in default_names:
                ops = self._operations_by_gate.get(name)
                if not ops:
                    raise ValueError(
                        f"Global gate allowlist references gate '{name}' which has no operations "
                        f"for {self.num_qubits} qubits."
                    )
                default_ops.extend(ops)
            self._default_layer_operations = tuple(default_ops)
        for layer, operation in self.fixed_operations.items():
            allowed = self._layer_specific_operations.get(layer)
            if allowed is not None and operation not in allowed:
                raise ValueError(
                    f"Layer {layer} has a fixed gate '{operation.gate.name}' "
                    f"that is not allowed by the layer gate constraint."
                )

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
        constrained = self._layer_specific_operations.get(depth)
        if constrained is not None:
            return constrained
        if self._default_layer_operations is not None:
            return self._default_layer_operations
        return self.operations

    def _select_identity_for_layer(self, depth: int) -> Optional[GateOperation]:
        fixed_operation = self.fixed_operations.get(depth)
        if fixed_operation is not None:
            return fixed_operation
        constrained = self._layer_specific_operations.get(depth)
        if constrained is not None:
            for operation in constrained:
                if operation.gate.name == "I":
                    return operation
            return None
        if self._default_layer_operations is not None:
            for operation in self._default_layer_operations:
                if operation.gate.name == "I":
                    return operation
        if self._identity_operation is not None:
            return self._identity_operation
        for operation in self.operations:
            if operation.gate.name == "I":
                return operation
        return None

    def _pad_sequence_to_layers(
        self, sequence: Sequence[GateOperation], *, max_layers: int
    ) -> List[GateOperation]:
        if len(sequence) >= max_layers:
            return list(sequence)
        padded = list(sequence)
        for depth in range(len(sequence), max_layers):
            identity_op = self._select_identity_for_layer(depth)
            if identity_op is None:
                break
            padded.append(identity_op)
        return padded

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
        if self._layer_specific_operations and max(self._layer_specific_operations) >= max_layers:
            raise ValueError(
                "A layer gate constraint is defined beyond the configured maximum number of layers."
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
        visited_layers: Dict[int, set[AmplitudeKey]] = {0: {self._state_key(start.amplitudes)}}
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

                if new_distance <= self.tolerance and self._fixed_layers_satisfied(len(new_sequence)):
                    final_sequence = self._pad_sequence_to_layers(new_sequence, max_layers=max_layers)
                    states = self._evolve_states(start, final_sequence)
                    final_state = states[-1] if states else start.copy()
                    final_distance = final_state.distance(target)
                    return SolverResult(
                        success=True,
                        sequence=final_sequence,
                        layers_used=len(new_sequence),
                        final_state=final_state,
                        distance=final_distance,
                        states=states,
                    )

                key = self._state_key(new_state.amplitudes)
                next_depth = len(new_sequence)
                layer_bucket = visited_layers.setdefault(next_depth, set())
                if key not in layer_bucket:
                    layer_bucket.add(key)
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
