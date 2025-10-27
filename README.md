# QuantumSolver

Constraint solver for quantum states using standard quantum gates.

## Requirements

The project uses the Python standard library only. Any Python 3.10+ interpreter should work.

## Usage

1. Prepare a JSON configuration describing the initial and target states. Examples are provided in `examples/bell_state.json` and `examples/fixed_gate_compensation.json`.
2. Run the solver:
   ```bash
   PYTHONPATH=src python3 -m quantum_solver.cli --config examples/bell_state.json
   ```
   Use `--max-layers`, `--allowed-gates`, `--output`, or `--no-timeline` to control execution. For example:
   ```bash
   PYTHONPATH=src python3 -m quantum_solver.cli \
     --config examples/bell_state.json \
     --output artifacts/bell_result.json
   ```
   The `--output -` form prints the persisted result JSON to stdout.
   The CLI renders an ASCII timeline by default to help visualise the state after each gate.

## Configuration schema

- `num_qubits` (int): Number of qubits in the system.
- `layers` (int): Maximum number of gate applications.
- `initial_state` (list[list[float, float]]): Complex amplitudes expressed as `[real, imag]`.
- `target_state` (list[list[float, float]]): Desired amplitudes in the same format.
- `allowed_gates` (list[str], optional): Gate symbols to explore. Defaults to all supported gates.
- `fixed_gates` (list[object], optional): Gates that are enforced at specific steps. Each entry requires
  a 1-based `step`, a `gate` name, and a list of target qubits under `targets`. The solver honours these
  placements while searching the remaining layers.
- `layer_gate_constraints` (list[object], optional): Restrict which gates may be used at particular steps.
  Each entry provides a 1-based `step` and an `allowed_gates` list specifying the permitted symbols for
  that layer.
- `tolerance` (float, optional): Distance threshold for considering the target reached. Defaults to `1e-6`.
- `output_path` (str, optional): Persist solver results as JSON. Equivalent to the `--output` CLI flag.

## Supported gates

Single-qubit: `I`, `X`, `Y`, `Z`, `H`, `S`, `T`  
Two-qubit: `CNOT` (control listed first).
