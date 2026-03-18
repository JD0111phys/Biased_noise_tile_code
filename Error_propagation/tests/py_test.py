"""Unit tests for pauli_distribution.pauli_strings."""

from pathlib import Path
import json
import sys

import pytest


# Ensure local package import works when tests are run from workspace root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pauli_distribution import (  # noqa: E402
	convert_gate_sequence,
	effective_pauli_probabilities_from_counts,
	error_propagation_simulation,
	get_pauli_string,
	load_running_counts,
	save_running_counts,
	update_running_counts,
)


def test_get_pauli_string_length_and_value_range() -> None:
	samples = get_pauli_string(
		keep_qubits=[0, 2],
		samples=6,
		p=0.01,
		system_bias=10.0,
		gate_sequence=[("H", [0]), ("CX", [0, 2])],
		ancilla=[1],
		qubit_platform="superconducting",
		random_seed=123,
	)

	# Used qubits are sorted(set([0,2] U [1])) == [0,1,2], so len == 6 * 3.
	assert len(samples) == 18
	assert all(v in (0, 1, 2, 3) for v in samples)


@pytest.mark.parametrize(
	"kwargs,exc",
	[
		({"samples": -1}, ValueError),
		({"p": -0.1}, ValueError),
		({"p": 1.1}, ValueError),
		({"system_bias": -1.0}, ValueError),
		({"use_compressed_space": False}, NotImplementedError),
	],
)
def test_get_pauli_string_input_validation(kwargs: dict, exc: type[Exception]) -> None:
	base = {
		"keep_qubits": [0],
		"samples": 1,
		"p": 0.003,
		"system_bias": 1000.0,
		"gate_sequence": [],
		"ancilla": [],
	}
	base.update(kwargs)
	with pytest.raises(exc):
		get_pauli_string(**base)


def test_convert_gate_sequence_cz_native_transforms_cx_and_cy() -> None:
	seq = [("CX", [0, 1, 2, 3]), ("CY", [4, 5]), ("H", [6])]
	out = convert_gate_sequence(seq, "CZ_native")

	assert out[0] == ("H", [1, 3])
	assert out[1] == ("CZ", [0, 1, 2, 3])
	assert out[2] == ("H", [1, 3])
	assert out[3] == ("S", [5])
	assert out[4] == ("H", [5])
	assert out[5] == ("CZ", [4, 5])
	assert out[6] == ("H", [5])
	assert out[7] == ("S_DAG", [5])
	assert out[8] == ("H", [6])


def test_update_running_counts_updates_in_place() -> None:
	running = {0: 1, 1: 2, 2: 3, 3: 4}
	out = update_running_counts(running, [0, 0, 2, 3, 3, 3])

	assert out is running
	assert running == {0: 3, 1: 2, 2: 4, 3: 7}


def test_effective_probabilities_handles_empty_counts() -> None:
	probs = effective_pauli_probabilities_from_counts({})
	assert probs == {"I": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0}


def test_save_load_running_counts_handles_jsonl_and_malformed_lines(tmp_path: Path) -> None:
	p = tmp_path / "counts.jsonl"
	save_running_counts({0: 2, 1: 1, 2: 0, 3: 0}, str(p), append=False, seed=7)

	# Append legacy format using string keys, then malformed and non-dict lines.
	with p.open("a", encoding="utf-8") as f:
		f.write(json.dumps({"0": 1, "1": 0, "2": 2, "3": 3}) + "\n")
		f.write("{ malformed json\n")
		f.write(json.dumps([1, 2, 3]) + "\n")

	loaded = load_running_counts(str(p))
	assert loaded == {0: 3, 1: 1, 2: 2, 3: 3}


def test_error_propagation_simulation_validates_iteration_params() -> None:
	with pytest.raises(ValueError):
		error_propagation_simulation(
			keep_qubits=[0],
			ancilla=[],
			p_param=0.003,
			system_bias=10.0,
			qubit_platform="ideal",
			gate_sequence=[],
			samples_per_iteration=0,
			total_samples=10,
			chosen_seed=1,
			timestamp="t0",
		)


def test_error_propagation_simulation_respects_total_sample_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)

	progress_file, counts_file = error_propagation_simulation(
		keep_qubits=[0],
		ancilla=[],
		p_param=0.003,
		system_bias=10.0,
		qubit_platform="ideal",
		gate_sequence=[],
		samples_per_iteration=4,
		total_samples=10,
		chosen_seed=2,
		timestamp="unit",
	)

	assert Path(progress_file).exists()
	assert Path(counts_file).exists()

	# Initial + two loop batches (4 + 4 + 2) => 3 JSONL entries.
	line_count = sum(1 for _ in Path(counts_file).open("r", encoding="utf-8"))
	assert line_count == 3


def test_ideal_platform_skips_hardware_gate_error_channel() -> None:
	"""Regression: ideal platform means no hardware gate-channel noise injection."""
	gate_seq = [("CX", [0, 1])] * 20

	ideal_samples = get_pauli_string(
		keep_qubits=[0, 1],
		samples=50,
		p=0.0,
		system_bias=10.0,
		gate_sequence=gate_seq,
		ancilla=[],
		qubit_platform="ideal",
		random_seed=123,
	)
	hw_samples = get_pauli_string(
		keep_qubits=[0, 1],
		samples=50,
		p=0.0,
		system_bias=10.0,
		gate_sequence=gate_seq,
		ancilla=[],
		qubit_platform="superconducting",
		random_seed=123,
	)

	# p=0 disables generic stochastic insertion; ideal path should remain identity-only.
	assert all(v == 0 for v in ideal_samples)
	# Hardware platform can still inject gate-channel noise independently of p.
	assert any(v != 0 for v in hw_samples)
