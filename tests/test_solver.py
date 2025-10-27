import json
import math
import tempfile
import unittest

from quantum_solver.gates import GateOperation, SUPPORTED_GATES
from quantum_solver.solver import GateSequenceSolver
from quantum_solver.state import QuantumState
from quantum_solver.timeline import render_timeline


class GateSequenceSolverTest(unittest.TestCase):
    def test_single_qubit_x_gate(self) -> None:
        start = QuantumState.from_amplitudes([1.0, 0.0])
        target = QuantumState.from_amplitudes([0.0, 1.0])
        solver = GateSequenceSolver(num_qubits=1, allowed_gates=["X"])
        result = solver.solve(start, target, max_layers=1)
        self.assertTrue(result.success)
        self.assertEqual(len(result.sequence), 1)
        self.assertAlmostEqual(result.distance, 0.0, places=7)
        self.assertEqual(len(result.states), len(result.sequence))
        self.assertEqual(result.states[-1].amplitudes, result.final_state.amplitudes)

    def test_two_qubit_bell_state(self) -> None:
        amp = 1.0 / math.sqrt(2.0)
        start = QuantumState.from_amplitudes([1.0, 0.0, 0.0, 0.0])
        target = QuantumState.from_amplitudes([amp, 0.0, 0.0, amp])
        solver = GateSequenceSolver(num_qubits=2, allowed_gates=["H", "CNOT"])
        result = solver.solve(start, target, max_layers=3)
        self.assertTrue(result.success)
        self.assertLessEqual(len(result.sequence), 2)
        self.assertEqual(len(result.states), len(result.sequence))

    def test_failure_when_layers_too_small(self) -> None:
        amp = 1.0 / math.sqrt(2.0)
        start = QuantumState.from_amplitudes([1.0, 0.0, 0.0, 0.0])
        target = QuantumState.from_amplitudes([amp, 0.0, 0.0, amp])
        solver = GateSequenceSolver(num_qubits=2, allowed_gates=["H", "CNOT"])
        result = solver.solve(start, target, max_layers=1)
        self.assertFalse(result.success)
        self.assertGreater(result.distance, 0.0)
        self.assertEqual(len(result.states), len(result.sequence))

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
            self.assertEqual(len(payload["steps"]), 1)
            self.assertEqual(payload["steps"][0]["operation"], {"gate": "X", "targets": [0]})
            amplitudes = payload["final_state"]["amplitudes"]
            self.assertEqual(amplitudes[0], [0.0, 0.0])
            self.assertEqual(amplitudes[1], [1.0, 0.0])

    def test_render_timeline_no_operations(self) -> None:
        state = QuantumState.from_amplitudes([1.0, 0.0])
        timeline = render_timeline(
            state,
            [],
            intermediate_states=[],
            final_state=state,
        )
        self.assertIn("Timeline: (no operations)", timeline)
        self.assertIn("|0>", timeline)

    def test_render_timeline_with_operations(self) -> None:
        start = QuantumState.from_amplitudes([1.0, 0.0, 0.0, 0.0])
        target_amp = 1.0 / math.sqrt(2.0)
        target = QuantumState.from_amplitudes([target_amp, 0.0, 0.0, target_amp])
        solver = GateSequenceSolver(num_qubits=2, allowed_gates=["H", "CNOT"])
        result = solver.solve(start, target, max_layers=3)
        timeline = render_timeline(
            start,
            result.sequence,
            intermediate_states=result.states,
            final_state=result.final_state,
        )
        self.assertIn("Layer 1", timeline)
        self.assertIn("H q0", timeline)
        self.assertIn("CNOT q0->q1", timeline)
        self.assertIn("Final state:", timeline)

    def test_fixed_gate_requires_compensation(self) -> None:
        start = QuantumState.from_amplitudes([1.0, 0.0])
        target = QuantumState.from_amplitudes([1.0, 0.0])
        fixed = {1: GateOperation(gate=SUPPORTED_GATES["X"], targets=(0,))}
        solver = GateSequenceSolver(num_qubits=1, allowed_gates=["X"], fixed_operations=fixed)
        result = solver.solve(start, target, max_layers=2)
        self.assertTrue(result.success)
        self.assertEqual(len(result.sequence), 2)
        self.assertEqual([op.gate.name for op in result.sequence], ["X", "X"])
        self.assertAlmostEqual(result.distance, 0.0, places=7)
        self.assertEqual(result.final_state.amplitudes, target.amplitudes)

    def test_fixed_gate_beyond_max_layers_raises(self) -> None:
        start = QuantumState.from_amplitudes([1.0, 0.0])
        target = QuantumState.from_amplitudes([1.0, 0.0])
        fixed = {2: GateOperation(gate=SUPPORTED_GATES["X"], targets=(0,))}
        solver = GateSequenceSolver(num_qubits=1, allowed_gates=["X"], fixed_operations=fixed)
        with self.assertRaises(ValueError):
            solver.solve(start, target, max_layers=2)

    def test_layer_gate_constraints_with_fixed_sequence(self) -> None:
        start = QuantumState.from_amplitudes([1.0, 0.0])
        target = QuantumState.from_amplitudes([0.0, 1.0])
        fixed = {
            0: GateOperation(gate=SUPPORTED_GATES["H"], targets=(0,)),
            1: GateOperation(gate=SUPPORTED_GATES["S"], targets=(0,)),
            2: GateOperation(gate=SUPPORTED_GATES["S"], targets=(0,)),
            3: GateOperation(gate=SUPPORTED_GATES["S"], targets=(0,)),
            4: GateOperation(gate=SUPPORTED_GATES["S"], targets=(0,)),
            5: GateOperation(gate=SUPPORTED_GATES["Z"], targets=(0,)),
        }
        layer_constraints = {6: ["H", "S", "Z"]}
        solver = GateSequenceSolver(
            num_qubits=1,
            allowed_gates=["H", "S", "Z", "X"],
            fixed_operations=fixed,
            layer_gate_allowlists=layer_constraints,
        )
        result = solver.solve(start, target, max_layers=7)
        self.assertTrue(result.success)
        self.assertEqual(len(result.sequence), 7)
        self.assertEqual([op.gate.name for op in result.sequence[:6]], ["H", "S", "S", "S", "S", "Z"])
        self.assertEqual(result.sequence[6].gate.name, "H")
        self.assertAlmostEqual(result.distance, 0.0, places=7)
        final = result.final_state.amplitudes
        self.assertAlmostEqual(final[0].real, target.amplitudes[0].real, places=7)
        self.assertAlmostEqual(final[1].real, target.amplitudes[1].real, places=7)
        self.assertAlmostEqual(final[0].imag, target.amplitudes[0].imag, places=7)
        self.assertAlmostEqual(final[1].imag, target.amplitudes[1].imag, places=7)

    def test_layer_gate_constraint_conflict_with_fixed_gate_raises(self) -> None:
        fixed = {0: GateOperation(gate=SUPPORTED_GATES["H"], targets=(0,))}
        layer_constraints = {0: ["S"]}
        with self.assertRaises(ValueError):
            GateSequenceSolver(
                num_qubits=1,
                allowed_gates=["H", "S"],
                fixed_operations=fixed,
                layer_gate_allowlists=layer_constraints,
            )


if __name__ == "__main__":
    unittest.main()
