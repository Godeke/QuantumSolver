import json
import math
import tempfile
import unittest

from quantum_solver.solver import GateSequenceSolver
from quantum_solver.state import QuantumState


class GateSequenceSolverTest(unittest.TestCase):
    def test_single_qubit_x_gate(self) -> None:
        start = QuantumState.from_amplitudes([1.0, 0.0])
        target = QuantumState.from_amplitudes([0.0, 1.0])
        solver = GateSequenceSolver(num_qubits=1, allowed_gates=["X"])
        result = solver.solve(start, target, max_layers=1)
        self.assertTrue(result.success)
        self.assertEqual(len(result.sequence), 1)
        self.assertAlmostEqual(result.distance, 0.0, places=7)

    def test_two_qubit_bell_state(self) -> None:
        amp = 1.0 / math.sqrt(2.0)
        start = QuantumState.from_amplitudes([1.0, 0.0, 0.0, 0.0])
        target = QuantumState.from_amplitudes([amp, 0.0, 0.0, amp])
        solver = GateSequenceSolver(num_qubits=2, allowed_gates=["H", "CNOT"])
        result = solver.solve(start, target, max_layers=3)
        self.assertTrue(result.success)
        self.assertLessEqual(len(result.sequence), 2)

    def test_failure_when_layers_too_small(self) -> None:
        amp = 1.0 / math.sqrt(2.0)
        start = QuantumState.from_amplitudes([1.0, 0.0, 0.0, 0.0])
        target = QuantumState.from_amplitudes([amp, 0.0, 0.0, amp])
        solver = GateSequenceSolver(num_qubits=2, allowed_gates=["H", "CNOT"])
        result = solver.solve(start, target, max_layers=1)
        self.assertFalse(result.success)
        self.assertGreater(result.distance, 0.0)

    def test_write_result_persists_sequence(self) -> None:
        from pathlib import Path
        from quantum_solver.persistence import write_result

        start = QuantumState.from_amplitudes([1.0, 0.0])
        target = QuantumState.from_amplitudes([0.0, 1.0])
        solver = GateSequenceSolver(num_qubits=1, allowed_gates=["X"])
        result = solver.solve(start, target, max_layers=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            write_result(result, path)

            payload = json.loads(path.read_text())
            self.assertTrue(payload["success"])
            self.assertEqual(payload["sequence"], [{"gate": "X", "targets": [0]}])
            amplitudes = payload["final_state"]["amplitudes"]
            self.assertEqual(amplitudes[0], [0.0, 0.0])
            self.assertEqual(amplitudes[1], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
