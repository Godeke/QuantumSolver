# QuantumSolver

Constraint solver for quantum states using standard quantum gates.

## Requirements

The project uses the Python standard library only. Any Python 3.10+ interpreter should work.

## Usage

1. Prepare a JSON configuration describing the initial and target states. An example is provided in `examples/bell_state.json`.
2. Run the solver:
   ```bash
   PYTHONPATH=src python3 -m quantum_solver.cli --config examples/bell_state.json
   ```
   Use `--max-layers`, `--allowed-gates`, or `--output` flags to override the configuration file. For example:
   ```bash
   PYTHONPATH=src python3 -m quantum_solver.cli \
     --config examples/bell_state.json \
     --output artifacts/bell_result.json
   ```
   The `--output -` form prints the persisted result JSON to stdout.

## Configuration schema

- `num_qubits` (int): Number of qubits in the system.
- `layers` (int): Maximum number of gate applications.
- `initial_state` (list[list[float, float]]): Complex amplitudes expressed as `[real, imag]`.
- `target_state` (list[list[float, float]]): Desired amplitudes in the same format.
- `allowed_gates` (list[str], optional): Gate symbols to explore. Defaults to all supported gates.
- `tolerance` (float, optional): Distance threshold for considering the target reached. Defaults to `1e-6`.
- `output_path` (str, optional): Persist solver results as JSON. Equivalent to the `--output` CLI flag.

## Supported gates

Single-qubit: `I`, `X`, `Y`, `Z`, `H`, `S`, `T`  
Two-qubit: `CNOT` (control listed first).
