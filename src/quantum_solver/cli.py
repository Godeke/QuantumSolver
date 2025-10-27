"""Command-line interface for the QuantumSolver project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .gates import GateOperation, SUPPORTED_GATES
from .persistence import result_to_payload, write_result
from .solver import GateSequenceSolver
from .timeline import render_timeline
from .state import QuantumState


def _load_config(path: str) -> dict:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text())


def _parse_amplitudes(raw: Sequence[Sequence[float]], label: str) -> QuantumState:
    try:
        return QuantumState.from_real_imag_pairs(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {label} state: {exc}") from exc


def _parse_fixed_gates(raw: object, *, num_qubits: int) -> Dict[int, GateOperation]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ValueError("Configuration field 'fixed_gates' must be a list.")

    fixed_operations: Dict[int, GateOperation] = {}
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                "Each item in 'fixed_gates' must be an object with 'step', 'gate', and 'targets'."
            )

        step_value = item.get("step", item.get("layer"))
        if step_value is None:
            raise ValueError("Fixed gate entry is missing 'step' (1-based index).")
        try:
            step = int(step_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fixed gate step at index {index} must be an integer.") from exc
        if step <= 0:
            raise ValueError(f"Fixed gate step must be positive; received {step}.")
        layer_index = step - 1

        gate_name = item.get("gate")
        if not isinstance(gate_name, str):
            raise ValueError(f"Fixed gate at step {step} must specify a gate name.")
        if gate_name not in SUPPORTED_GATES:
            raise ValueError(f"Fixed gate '{gate_name}' at step {step} is not supported.")

        targets_raw = item.get("targets")
        if not isinstance(targets_raw, (list, tuple)):
            raise ValueError(f"Fixed gate at step {step} must define 'targets' as a list.")
        try:
            targets = tuple(int(target) for target in targets_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fixed gate targets at step {step} must be integers.") from exc

        for target in targets:
            if target < 0 or target >= num_qubits:
                raise ValueError(
                    f"Fixed gate '{gate_name}' at step {step} targets qubit {target}, "
                    f"but the solver is configured for {num_qubits} qubits."
                )

        if layer_index in fixed_operations:
            raise ValueError(f"Multiple fixed gates defined for step {step}.")

        operation = GateOperation(gate=SUPPORTED_GATES[gate_name], targets=targets)
        fixed_operations[layer_index] = operation

    return fixed_operations


def _format_complex(value: complex, *, precision: int = 6) -> str:
    real = round(value.real, precision)
    imag = round(value.imag, precision)
    sign = "+" if imag >= 0 else "-"
    return f"{real:.{precision}f} {sign} {abs(imag):.{precision}f}i"


def _print_result(result, num_qubits: int, max_layers: int) -> None:
    if result.success:
        print(f"Solved target state in {result.layers_used} layer(s).")
    else:
        print(f"Failed to reach target within {max_layers} layers.")
    print(f"Final distance: {result.distance:.6e}")

    if result.sequence:
        print("Gate sequence:")
        for idx, operation in enumerate(result.sequence, start=1):
            print(f"  {idx}. {operation.describe()}")
    else:
        print("Gate sequence: (empty)")

    print("Final state amplitudes:")
    basis_width = num_qubits
    for index, amplitude in enumerate(result.final_state.amplitudes):
        basis_label = format(index, f"0{basis_width}b")
        print(f"  |{basis_label}> = {_format_complex(amplitude)}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search for a sequence of quantum gates that maps one state to another."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON configuration file or '-' to read from stdin.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        help="Override the maximum number of layers supplied in the config file.",
    )
    parser.add_argument(
        "--allowed-gates",
        nargs="+",
        help="Override the allowed gates list supplied in the config file.",
    )
    parser.add_argument(
        "--no-timeline",
        dest="timeline",
        action="store_false",
        help="Disable ASCII timeline output.",
    )
    parser.set_defaults(timeline=True)
    parser.add_argument(
        "--output",
        help="Persist the solver result JSON to this path. Use '-' for stdout.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = _load_config(args.config)

    try:
        num_qubits = int(config["num_qubits"])
        if num_qubits <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("Configuration must define a positive integer 'num_qubits'.") from exc

    initial_state_raw = config.get("initial_state")
    target_state_raw = config.get("target_state")
    if initial_state_raw is None or target_state_raw is None:
        raise SystemExit("Configuration must include 'initial_state' and 'target_state'.")

    initial_state = _parse_amplitudes(initial_state_raw, "initial")
    target_state = _parse_amplitudes(target_state_raw, "target")

    if initial_state.num_qubits != num_qubits or target_state.num_qubits != num_qubits:
        raise SystemExit(
            f"State vectors represent {initial_state.num_qubits} and {target_state.num_qubits} qubits, "
            f"but solver is configured for {num_qubits}."
        )

    max_layers = args.max_layers or int(config.get("layers", 0))
    if max_layers <= 0:
        raise SystemExit("Maximum number of layers must be a positive integer.")

    allowed_gates: List[str] | None = None
    if args.allowed_gates is not None:
        allowed_gates = args.allowed_gates
    elif "allowed_gates" in config:
        allowed_gates = list(config["allowed_gates"])

    try:
        fixed_gates = _parse_fixed_gates(config.get("fixed_gates"), num_qubits=num_qubits)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    solver = GateSequenceSolver(
        num_qubits=num_qubits,
        allowed_gates=allowed_gates,
        tolerance=float(config.get("tolerance", 1e-6)),
        fixed_operations=fixed_gates,
    )
    result = solver.solve(initial_state, target_state, max_layers=max_layers)
    _print_result(result, num_qubits=num_qubits, max_layers=max_layers)
    if args.timeline:
        print()
        timeline = render_timeline(
            initial_state,
            result.sequence,
            intermediate_states=result.states,
            final_state=result.final_state,
        )
        print(timeline)
    output_path = args.output if args.output is not None else config.get("output_path")
    if output_path:
        payload = result_to_payload(result)
        if output_path == "-":
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            write_result(result, output_path)
            print(f"Persisted result to {output_path}")
    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
