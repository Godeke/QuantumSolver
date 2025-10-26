"""ASCII timeline renderer for gate sequences and state evolution."""

from __future__ import annotations

from typing import List, Sequence

from .gates import GateOperation
from .state import QuantumState


def _format_amplitude(amplitude: complex, *, precision: int = 6) -> str:
    return f"{amplitude.real:.{precision}f}{amplitude.imag:+.{precision}f}i"


def _format_probability(amplitude: complex, *, precision: int = 6) -> str:
    probability = abs(amplitude) ** 2
    return f"{probability:.{precision}f}"


def format_state(state: QuantumState, *, precision: int = 6) -> List[str]:
    lines: List[str] = []
    width = state.num_qubits
    for index, amplitude in enumerate(state.amplitudes):
        label = format(index, f"0{width}b")
        lines.append(
            f"|{label}> amplitude={_format_amplitude(amplitude, precision=precision)}, "
            f"prob={_format_probability(amplitude, precision=precision)}"
        )
    return lines


def _render_layer_lines(operation: GateOperation, num_qubits: int, width: int = 7) -> List[str]:
    center = width // 2
    wires: List[List[str]] = [["─"] * width for _ in range(num_qubits)]

    if operation.gate.num_qubits == 1:
        target = operation.targets[0]
        symbol = operation.gate.name[:1] or "?"
        wires[target][center] = symbol
    elif operation.gate.name.upper() == "CNOT":
        control, target = operation.targets
        top, bottom = sorted((control, target))
        wires[control][center] = "●"
        wires[target][center] = "X"
        for idx in range(top + 1, bottom):
            wires[idx][center] = "│"
    else:
        symbol = operation.gate.name[:1] or "?"
        for qubit in operation.targets:
            wires[qubit][center] = symbol

    return [f"q{idx} " + "".join(chars) for idx, chars in enumerate(wires)]


def render_timeline(
    start: QuantumState,
    operations: Sequence[GateOperation],
    *,
    intermediate_states: Sequence[QuantumState],
    final_state: QuantumState,
    precision: int = 6,
) -> str:
    lines: List[str] = []
    lines.append("Initial state:")
    lines.extend(format_state(start, precision=precision))
    lines.append("")

    if not operations:
        lines.append("Timeline: (no operations)")
        lines.append("")
        lines.append("Final state:")
        lines.extend(format_state(final_state, precision=precision))
        return "\n".join(lines)

    lines.append("Timeline:")
    for layer_index, (operation, state) in enumerate(zip(operations, intermediate_states), start=1):
        lines.append(f"Layer {layer_index}: {operation.describe()}")
        for wire_line in _render_layer_lines(operation, start.num_qubits):
            lines.append("    " + wire_line)
        lines.append("    State after layer {}:".format(layer_index))
        for state_line in format_state(state, precision=precision):
            lines.append("        " + state_line)
        lines.append("")

    lines.append("Final state:")
    lines.extend(format_state(final_state, precision=precision))
    return "\n".join(lines)
