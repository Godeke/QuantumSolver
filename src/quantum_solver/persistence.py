"""Helper routines for persisting solver results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .gates import GateOperation
from .solver import SolverResult


def _complex_to_pair(value: complex) -> List[float]:
    return [float(value.real), float(value.imag)]


def amplitudes_to_pairs(amplitudes: Iterable[complex]) -> List[List[float]]:
    return [_complex_to_pair(value) for value in amplitudes]


def serialize_sequence(sequence: Iterable[GateOperation]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for operation in sequence:
        serialized.append(
            {
                "gate": operation.gate.name,
                "targets": list(operation.targets),
            }
        )
    return serialized


def _state_payload(state) -> Dict[str, Any]:
    return {
        "num_qubits": state.num_qubits,
        "amplitudes": amplitudes_to_pairs(state.amplitudes),
        "probabilities": list(state.as_probability_distribution()),
    }


def result_to_payload(result: SolverResult) -> Dict[str, Any]:
    steps = []
    for index, (operation, state) in enumerate(zip(result.sequence, result.states), start=1):
        steps.append(
            {
                "layer": index,
                "operation": {
                    "gate": operation.gate.name,
                    "targets": list(operation.targets),
                },
                "state": _state_payload(state),
            }
        )

    return {
        "success": result.success,
        "distance": result.distance,
        "layers_used": result.layers_used,
        "sequence": serialize_sequence(result.sequence),
        "steps": steps,
        "final_state": _state_payload(result.final_state),
    }


def write_result(result: SolverResult, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result_to_payload(result)
    path.write_text(json.dumps(payload, indent=2))
    return path
