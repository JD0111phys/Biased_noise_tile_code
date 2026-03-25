"""Unit tests for pauli_distribution.pauli_strings."""

from pathlib import Path
import json
import sys
import logging
import pytest
import random
from stim import PauliString
from typing import List
logger = logging.getLogger(__name__)


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
	apply_gate_error_channel,
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


def test_get_pauli_string_rejects_mismatched_initial_pauli_string_length() -> None:
	with pytest.raises(ValueError, match="initial_pauli_string length"):
		get_pauli_string(
			keep_qubits=[0, 2],
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[],
			ancilla=[1],
			qubit_platform="ideal",
			initial_pauli_string="__",
		)


def test_get_pauli_string_uses_custom_initial_pauli_string() -> None:
	samples = get_pauli_string(
		keep_qubits=[0],
		samples=4,
		p=0.0,
		system_bias=10.0,
		gate_sequence=[],
		ancilla=[],
		qubit_platform="ideal",
		random_seed=123,
		initial_pauli_string="Z",
	)

	# With p=0 and no gates, the initial state should be preserved sample-by-sample.
	assert samples == [3, 3, 3, 3]


# Error at the end of circuit Tests
# Load XXZX 4-qubit circuit Initial Pauli strings and Error propagations
with open(Path(__file__).parent / "xzzx_export.json", "r") as f:
    data = json.load(f)

init_Pauli_strings = data["seq_strings"]
error_Propagation_lists = data["pauli_numeric_list_of_lists"]

def test_4_qubit_XZZX_circuit_initial_Pauli_propagation() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		samples = get_pauli_string(
			keep_qubits=[0,1,2,3,4],
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
			ancilla=[0],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		assert samples == error_P

def test_4_qubit_XZZX_circuit_initial_Pauli_propagation_converted_gate_sequence_native_CNOT() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		samples = get_pauli_string(
			keep_qubits=[0,1,2,3,4],
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=convert_gate_sequence([("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])], "CNOT_native"),
			ancilla=[0],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		assert samples == error_P
	
def test_4_qubit_XZZX_circuit_initial_Pauli_propagation_converted_gate_sequence_native_CZ() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		samples = get_pauli_string(
			keep_qubits=[0,1,2,3,4],
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=convert_gate_sequence([("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])], "CZ_native"),
			ancilla=[0],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		assert samples == error_P

def test_4_qubit_XZZX_circuit_random_seed_consistency() -> None:
	samples1 = get_pauli_string(
		keep_qubits=[0,1,2,3,4],
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		ancilla=[0],
		qubit_platform="ideal",
		random_seed=123,
	)

	samples2 = get_pauli_string(
		keep_qubits=[0,1,2,3,4],
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		ancilla=[0],
		qubit_platform="ideal",
		random_seed=123,
	)
	logger.info("Samples1: %s", samples1)
	logger.info("Samples2: %s", samples2)

	assert samples1 == samples2

def test_4_qubit_XZZX_circuit_random_seed_consistency_2() -> None:
	samples1 = get_pauli_string(
		keep_qubits=[0,1,2,3,4],
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		ancilla=[0],
		qubit_platform="ideal",
		random_seed=123,
	)

	samples2 = get_pauli_string(
		keep_qubits=[0,1,2,3,4],
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		ancilla=[0],
		qubit_platform="ideal",
		random_seed=124,
	)
	logger.info("Samples1: %s", samples1)
	logger.info("Samples2: %s", samples2)

	assert samples1 != samples2
# Load XYHS 4-qubit circuit Initial Pauli strings and Error propagations
with open(Path(__file__).parent / "xyhs_export.json", "r") as f:
    data_xyhs = json.load(f)

init_Pauli_strings_xyhs = data_xyhs["seq_strings"]
error_Propagation_lists_xyhs = data_xyhs["pauli_numeric_list_of_lists"]

def test_4_qubit_XYHS_circuit_initial_Pauli_propagation() -> None:
	for P_string, error_P in zip(init_Pauli_strings_xyhs, error_Propagation_lists_xyhs):
		samples = get_pauli_string(
			keep_qubits=[0,1,2,3,4],
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("CX", [0, 1]), ("CY", [0, 2]), ("H", [3]), ("S", [4])],
			ancilla=[0],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		assert samples == error_P

def test_4_qubit_XYHS_circuit_initial_Pauli_propagation_2() -> None:
	for P_string, error_P in zip(init_Pauli_strings_xyhs, error_Propagation_lists_xyhs):
		samples = get_pauli_string(
			keep_qubits=[0,1,2,3,4],
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("S_DAG", [4]),("H",[3]),("CY", [0, 2]),("CX", [0, 1])],
			ancilla=[0],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		assert samples == error_P

def test_hadamard_independent_of_platform_specific_pauli_channels() -> None:
	"""Test that the Hadamard gate behaves independently of platform-specific Pauli channels."""
	samples_super = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("H", [0])],
		qubit_platform="superconducting",
		random_seed=123,
	)

	
	x_count_super = samples_super.count(1)
	y_count_super = samples_super.count(2)
	z_count_super = samples_super.count(3)

	logger.info("Hadamard gate error counts superconducting: X=%d, Y=%d, Z=%d", x_count_super, y_count_super, z_count_super)

	samples_NA = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("H", [0])],
		qubit_platform="neutral_atom",
		random_seed=123,
	)

	
	x_count_NA = samples_NA.count(1)
	y_count_NA = samples_NA.count(2)
	z_count_NA = samples_NA.count(3)

	logger.info("Hadamard gate error counts neutral_atom: X=%d, Y=%d, Z=%d", x_count_NA, y_count_NA, z_count_NA)

	samples_Tr_CNOT = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("H", [0])],
		qubit_platform="trapped_ion_cnot",
		random_seed=123,
	)

	
	x_count_Tr_CNOT = samples_Tr_CNOT.count(1)
	y_count_Tr_CNOT = samples_Tr_CNOT.count(2)
	z_count_Tr_CNOT = samples_Tr_CNOT.count(3)

	logger.info("Hadamard gate error counts trapped_ion_CNOT: X=%d, Y=%d, Z=%d", x_count_Tr_CNOT, y_count_Tr_CNOT, z_count_Tr_CNOT)

	samples_Tr_CZ = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("H", [0])],
		qubit_platform="trapped_ion_cz",
		random_seed=123,
	)
	x_count_Tr_CZ = samples_Tr_CZ.count(1)
	y_count_Tr_CZ = samples_Tr_CZ.count(2)
	z_count_Tr_CZ = samples_Tr_CZ.count(3)

	logger.info("Hadamard gate error counts trapped_ion_CZ: X=%d, Y=%d, Z=%d", x_count_Tr_CZ, y_count_Tr_CZ, z_count_Tr_CZ)

	assert x_count_super == x_count_NA == x_count_Tr_CNOT == x_count_Tr_CZ
	assert y_count_super == y_count_NA == y_count_Tr_CNOT == y_count_Tr_CZ
	assert z_count_super == z_count_NA == z_count_Tr_CNOT == z_count_Tr_CZ


def test_S_gate_independent_of_platform_specific_pauli_channels() -> None:
	"""Test that the S gate behaves independently of platform-specific Pauli channels."""
	samples_super = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("S", [0])],
		qubit_platform="superconducting",
		random_seed=123,
	)

	
	x_count_super = samples_super.count(1)
	y_count_super = samples_super.count(2)
	z_count_super = samples_super.count(3)

	logger.info("S gate error counts superconducting: X=%d, Y=%d, Z=%d", x_count_super, y_count_super, z_count_super)

	samples_NA = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("S", [0])],
		qubit_platform="neutral_atom",
		random_seed=123,
	)

	
	x_count_NA = samples_NA.count(1)
	y_count_NA = samples_NA.count(2)
	z_count_NA = samples_NA.count(3)

	logger.info("S gate error counts neutral_atom: X=%d, Y=%d, Z=%d", x_count_NA, y_count_NA, z_count_NA)

	samples_Tr_CNOT = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("S", [0])],
		qubit_platform="trapped_ion_cnot",
		random_seed=123,
	)

	
	x_count_Tr_CNOT = samples_Tr_CNOT.count(1)
	y_count_Tr_CNOT = samples_Tr_CNOT.count(2)
	z_count_Tr_CNOT = samples_Tr_CNOT.count(3)

	logger.info("S gate error counts trapped_ion_CNOT: X=%d, Y=%d, Z=%d", x_count_Tr_CNOT, y_count_Tr_CNOT, z_count_Tr_CNOT)

	samples_Tr_CZ = get_pauli_string(
		keep_qubits=[0],
		samples=1000,
		p=0.1,
		system_bias=10.0,
		gate_sequence=[("S", [0])],
		qubit_platform="trapped_ion_cz",
		random_seed=123,
	)
	x_count_Tr_CZ = samples_Tr_CZ.count(1)
	y_count_Tr_CZ = samples_Tr_CZ.count(2)
	z_count_Tr_CZ = samples_Tr_CZ.count(3)

	logger.info("S gate error counts trapped_ion_CZ: X=%d, Y=%d, Z=%d", x_count_Tr_CZ, y_count_Tr_CZ, z_count_Tr_CZ)

	assert x_count_super == x_count_NA == x_count_Tr_CNOT == x_count_Tr_CZ
	assert y_count_super == y_count_NA == y_count_Tr_CNOT == y_count_Tr_CZ
	assert z_count_super == z_count_NA == z_count_Tr_CNOT == z_count_Tr_CZ

def test_hadamard_error_channel() -> None:
	gate = ("H", [0])
	identity = "_"
	samples = 100000
	results: List[int] = []
	#random.seed(123)
	for _ in range(samples):
		pauli = PauliString(identity)
		pauli = apply_gate_error_channel(pauli, gate[0], gate[1], identity, "superconducting")
		results.extend(list(pauli))
	running_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # I, X, Y, Z
    # Update running counts with initial samples
	running_counts = update_running_counts(running_counts, results)
	effective_probs = effective_pauli_probabilities_from_counts(running_counts)
	logger.info("Hadamard gate effective Pauli probabilities: %s", effective_probs)

	err_dict_hadamard = {'I': 0.9970831318064503,
						'X': 0.0004876108844646676,
						'Y': 0.0009721305333010022,
						'Z': 0.0014571267757840511}
	assert all(abs(effective_probs[pauli] - err_dict_hadamard[pauli]) < 0.01 for pauli in err_dict_hadamard)

def test_s_gate_error_channel() -> None:
	gate = ("S", [0])
	identity = "_"
	samples = 100000
	results: List[int] = []
	#random.seed(123)
	for _ in range(samples):
		pauli = PauliString(identity)
		pauli = apply_gate_error_channel(pauli, gate[0], gate[1], identity, "superconducting")
		results.extend(list(pauli))
	running_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # I, X, Y, Z
    # Update running counts with initial samples
	running_counts = update_running_counts(running_counts, results)
	effective_probs = effective_pauli_probabilities_from_counts(running_counts)
	logger.info("S gate effective Pauli probabilities: %s", effective_probs)

	err_dict_s = {'I': 0.9970240579376571,
					'X': 1.4922560645502791e-07,
					'Y': 1.4922560645502791e-07,
					'Z': 0.00297564361112998}
	assert all(abs(effective_probs[pauli] - err_dict_s[pauli]) < 0.01 for pauli in err_dict_s)




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


def test_ideal_propagation_h_gate() -> None:
	# X through H -> Z
	samples_x = get_pauli_string(
		keep_qubits=[0], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("H", [0])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="X"
	)
	assert samples_x == [3]  # Z is 3

	# Z through H -> X
	samples_z = get_pauli_string(
		keep_qubits=[0], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("H", [0])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="Z"
	)
	assert samples_z == [1]  # X is 1


def test_ideal_propagation_s_gate() -> None:
	# X through S -> Y
	samples_x = get_pauli_string(
		keep_qubits=[0], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("S", [0])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="X"
	)
	assert samples_x == [2]  # Y is 2

	# Y through S -> X
	samples_y = get_pauli_string(
		keep_qubits=[0], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("S", [0])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="Y"
	)
	assert samples_y == [1]  # X is 1


def test_ideal_propagation_cx_gate() -> None:
	# XI through CX -> XX
	samples_xi = get_pauli_string(
		keep_qubits=[0, 1], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("CX", [0, 1])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="XI"
	)
	assert samples_xi == [1, 1]  # XX

	# IX through CX -> IX
	samples_ix = get_pauli_string(
		keep_qubits=[0, 1], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("CX", [0, 1])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="IX"
	)
	assert samples_ix == [0, 1]  # IX

	# ZI through CX -> ZI
	samples_zi = get_pauli_string(
		keep_qubits=[0, 1], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("CX", [0, 1])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="ZI"
	)
	assert samples_zi == [3, 0]  # ZI

	# IZ through CX -> ZZ
	samples_iz = get_pauli_string(
		keep_qubits=[0, 1], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("CX", [0, 1])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="IZ"
	)
	assert samples_iz == [3, 3]  # ZZ


def test_ideal_propagation_cz_gate() -> None:
	# XI through CZ -> XZ
	samples_xi = get_pauli_string(
		keep_qubits=[0, 1], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("CZ", [0, 1])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="XI"
	)
	assert samples_xi == [1, 3]  # XZ

	# IX through CZ -> ZX
	samples_ix = get_pauli_string(
		keep_qubits=[0, 1], samples=1, p=0.0, system_bias=10.0,
		gate_sequence=[("CZ", [0, 1])], ancilla=[], qubit_platform="ideal", random_seed=123,
		initial_pauli_string="IX"
	)
	assert samples_ix == [3, 1]  # ZX

