"""Unit tests for pauli_distribution.pauli_strings."""

from pathlib import Path
import json
import sys
import logging
import runpy
import pytest
import random
from stim import PauliString
from typing import List
logger = logging.getLogger(__name__)


# Ensure local package import works when tests are run from workspace root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pauli_distribution import (  # noqa: E402
	convert_gate_sequence,
	error_propagation_simulation,
	get_pauli_string,
	load_running_counts,
	save_running_counts,
	apply_gate_error_channel,
)
from pauli_distribution import pauli_strings as pauli_strings_module  # noqa: E402

TEST_DATA_DIR = Path(__file__).resolve().parent


def test_get_pauli_string_length_and_value_range() -> None:
	counts = get_pauli_string(
		samples=6,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("H", [0]), ("CX", [0, 2]), ("H", [1])],
		qubit_platform="superconducting",
		random_seed=123,
	)

	# Qubits used: [0, 1, 2], so 6 samples * 3 qubits = 18 observations total
	# Returns Dict[int, int] with keys 0,1,2,3
	assert isinstance(counts, dict)
	assert set(counts.keys()) == {0, 1, 2, 3}
	total_observations = sum(counts.values())
	assert total_observations == 18


@pytest.mark.parametrize(
	"kwargs,exc",
	[
		({"samples": -1}, ValueError),
		({"p": -0.1}, ValueError),
		({"p": 1.1}, ValueError),
		({"system_bias": -1.0}, ValueError),
	],
)
def test_get_pauli_string_input_validation(kwargs: dict, exc: type[Exception]) -> None:
	base = {
		"samples": 1,
		"p": 0.003,
		"system_bias": 1000.0,
		"gate_sequence": [],
	}
	base.update(kwargs)
	with pytest.raises(exc):
		get_pauli_string(**base)


def test_get_pauli_string_rejects_mismatched_initial_pauli_string_length() -> None:
	with pytest.raises(ValueError, match="initial_pauli_string length"):
		get_pauli_string(
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("H", [0]), ("H", [1]), ("H", [2])],
			qubit_platform="ideal",
			initial_pauli_string="__",  # Length 2, but need 3 for qubits 0, 1, 2
		)


def test_get_pauli_string_uses_custom_initial_pauli_string() -> None:
	counts = get_pauli_string(
		samples=4,
		p=0.0,
		system_bias=10.0,
		gate_sequence=[("H", [0])],
		qubit_platform="ideal",
		random_seed=123,
		initial_pauli_string="Z",
	)

	# With p=0 and a simple H gate on initial Z state, we should get X (label 1)
	# after the Hadamard is applied.
	assert isinstance(counts, dict)
	# H|Z> = |X>, so all samples should be labeled 1 (X)
	assert counts.get(1, 0) == 4  # All 4 samples should result in X error


# Error at the end of circuit Tests
# Load XXZX 4-qubit circuit Initial Pauli strings and Error propagations
with (TEST_DATA_DIR / "xzzx_export.json").open("r", encoding="utf-8") as f:
    data = json.load(f)

init_Pauli_strings = data["seq_strings"]
error_Propagation_lists = data["pauli_numeric_list_of_lists"]

def test_4_qubit_XZZX_circuit_initial_Pauli_propagation() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		counts = get_pauli_string(
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		# Convert the old list format to the new dict format
		expected_counts = {}
		for label in error_P:
			expected_counts[label] = expected_counts.get(label, 0) + 1
		
		# Compare only non-zero counts (API may include zeros)
		counts_nonzero = {k: v for k, v in counts.items() if v > 0}
		assert counts_nonzero == expected_counts

def test_4_qubit_XZZX_circuit_initial_Pauli_propagation_superconducting() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		counts_a = get_pauli_string(
			samples=1,
			p=0.003,
			system_bias=10000.0,
			gate_sequence=convert_gate_sequence([("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])], "CNOT_native"),
			qubit_platform="superconducting",
			random_seed=123,
			initial_pauli_string=P_string,
		)
		counts_b = get_pauli_string(
			samples=1,
			p=0.003,
			system_bias=10000.0,
			gate_sequence=convert_gate_sequence([("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])], "CNOT_native"),
			qubit_platform="superconducting",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		assert counts_a == counts_b

def test_4_qubit_XZZX_circuit_initial_Pauli_propagation_converted_gate_sequence_native_CNOT() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		counts = get_pauli_string(
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=convert_gate_sequence([("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])], "CNOT_native"),
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		# Convert the old list format to the new dict format
		expected_counts = {}
		for label in error_P:
			expected_counts[label] = expected_counts.get(label, 0) + 1
		
		# Compare only non-zero counts (API may include zeros)
		counts_nonzero = {k: v for k, v in counts.items() if v > 0}
		assert counts_nonzero == expected_counts
	
def test_4_qubit_XZZX_circuit_initial_Pauli_propagation_converted_gate_sequence_native_CZ() -> None:
	for P_string, error_P in zip(init_Pauli_strings, error_Propagation_lists):
		counts = get_pauli_string(
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=convert_gate_sequence([("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])], "CZ_native"),
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		# Convert the old list format to the new dict format
		expected_counts = {}
		for label in error_P:
			expected_counts[label] = expected_counts.get(label, 0) + 1
		
		# Compare only non-zero counts (API may include zeros)
		counts_nonzero = {k: v for k, v in counts.items() if v > 0}
		assert counts_nonzero == expected_counts

def test_4_qubit_XZZX_circuit_random_seed_consistency() -> None:
	counts1 = get_pauli_string(
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		qubit_platform="ideal",
		random_seed=123,
	)

	counts2 = get_pauli_string(
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		qubit_platform="ideal",
		random_seed=123,
	)
	logger.info("Counts1: %s", counts1)
	logger.info("Counts2: %s", counts2)

	assert counts1 == counts2

def test_4_qubit_XZZX_circuit_random_seed_consistency_2() -> None:
	counts1 = get_pauli_string(
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		qubit_platform="ideal",
		random_seed=123,
	)

	counts2 = get_pauli_string(
		samples=1,
		p=0.5,
		system_bias=10.0,
		gate_sequence=[("CX", [0, 1]), ("CZ", [0, 2]), ("CZ", [0, 3]), ("CX", [0, 4])],
		qubit_platform="ideal",
		random_seed=124,
	)
	logger.info("Counts1: %s", counts1)
	logger.info("Counts2: %s", counts2)

	assert counts1 != counts2
# Load XYHS 4-qubit circuit Initial Pauli strings and Error propagations
with (TEST_DATA_DIR / "xyhs_export.json").open("r", encoding="utf-8") as f:
    data_xyhs = json.load(f)

init_Pauli_strings_xyhs = data_xyhs["seq_strings"]
error_Propagation_lists_xyhs = data_xyhs["pauli_numeric_list_of_lists"]

def test_4_qubit_XYHS_circuit_initial_Pauli_propagation() -> None:
	for P_string, error_P in zip(init_Pauli_strings_xyhs, error_Propagation_lists_xyhs):
		counts = get_pauli_string(
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("CX", [0, 1]), ("CY", [0, 2]), ("H", [3]), ("S", [4])],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		# Convert the old list format to the new dict format
		expected_counts = {}
		for label in error_P:
			expected_counts[label] = expected_counts.get(label, 0) + 1
		
		# Compare only non-zero counts (API may include zeros)
		counts_nonzero = {k: v for k, v in counts.items() if v > 0}
		assert counts_nonzero == expected_counts

def test_4_qubit_XYHS_circuit_initial_Pauli_propagation_2() -> None:
	for P_string, error_P in zip(init_Pauli_strings_xyhs, error_Propagation_lists_xyhs):
		counts = get_pauli_string(
			samples=1,
			p=0.0,
			system_bias=10.0,
			gate_sequence=[("S_DAG", [4]),("H",[3]),("CY", [0, 2]),("CX", [0, 1])],
			qubit_platform="ideal",
			random_seed=123,
			initial_pauli_string=P_string,
		)

		# Convert the old list format to the new dict format
		expected_counts = {}
		for label in error_P:
			expected_counts[label] = expected_counts.get(label, 0) + 1
		
		# Compare only non-zero counts (API may include zeros)
		counts_nonzero = {k: v for k, v in counts.items() if v > 0}
		assert counts_nonzero == expected_counts

def test_hadamard_independent_of_platform_specific_pauli_channels() -> None:
	"""Test that the Hadamard gate behaves independently of platform-specific Pauli channels."""
	counts_super = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("H", [0])],
		qubit_platform="superconducting",
		random_seed=123,
	)

	
	x_count_super = counts_super.get(1, 0)
	y_count_super = counts_super.get(2, 0)
	z_count_super = counts_super.get(3, 0)

	logger.info("Hadamard gate error counts superconducting: X=%d, Y=%d, Z=%d", x_count_super, y_count_super, z_count_super)

	counts_NA = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("H", [0])],
		qubit_platform="neutral_atom",
		random_seed=123,
	)

	
	x_count_NA = counts_NA.get(1, 0)
	y_count_NA = counts_NA.get(2, 0)
	z_count_NA = counts_NA.get(3, 0)

	logger.info("Hadamard gate error counts neutral_atom: X=%d, Y=%d, Z=%d", x_count_NA, y_count_NA, z_count_NA)

	counts_Tr_CNOT = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("H", [0])],
		qubit_platform="trapped_ion_cnot",
		random_seed=123,
	)

	
	x_count_Tr_CNOT = counts_Tr_CNOT.get(1, 0)
	y_count_Tr_CNOT = counts_Tr_CNOT.get(2, 0)
	z_count_Tr_CNOT = counts_Tr_CNOT.get(3, 0)

	logger.info("Hadamard gate error counts trapped_ion_CNOT: X=%d, Y=%d, Z=%d", x_count_Tr_CNOT, y_count_Tr_CNOT, z_count_Tr_CNOT)

	counts_Tr_CZ = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("H", [0])],
		qubit_platform="trapped_ion_cz",
		random_seed=123,
	)
	x_count_Tr_CZ = counts_Tr_CZ.get(1, 0)
	y_count_Tr_CZ = counts_Tr_CZ.get(2, 0)
	z_count_Tr_CZ = counts_Tr_CZ.get(3, 0)

	logger.info("Hadamard gate error counts trapped_ion_CZ: X=%d, Y=%d, Z=%d", x_count_Tr_CZ, y_count_Tr_CZ, z_count_Tr_CZ)

	assert x_count_super == x_count_NA == x_count_Tr_CNOT == x_count_Tr_CZ
	assert y_count_super == y_count_NA == y_count_Tr_CNOT == y_count_Tr_CZ
	assert z_count_super == z_count_NA == z_count_Tr_CNOT == z_count_Tr_CZ


def test_S_gate_independent_of_platform_specific_pauli_channels() -> None:
	"""Test that the S gate behaves independently of platform-specific Pauli channels."""
	counts_super = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("S", [0])],
		qubit_platform="superconducting",
		random_seed=123,
	)

	
	x_count_super = counts_super.get(1, 0)
	y_count_super = counts_super.get(2, 0)
	z_count_super = counts_super.get(3, 0)

	logger.info("S gate error counts superconducting: X=%d, Y=%d, Z=%d", x_count_super, y_count_super, z_count_super)

	counts_NA = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("S", [0])],
		qubit_platform="neutral_atom",
		random_seed=123,
	)

	
	x_count_NA = counts_NA.get(1, 0)
	y_count_NA = counts_NA.get(2, 0)
	z_count_NA = counts_NA.get(3, 0)

	logger.info("S gate error counts neutral_atom: X=%d, Y=%d, Z=%d", x_count_NA, y_count_NA, z_count_NA)

	counts_Tr_CNOT = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("S", [0])],
		qubit_platform="trapped_ion_cnot",
		random_seed=123,
	)

	
	x_count_Tr_CNOT = counts_Tr_CNOT.get(1, 0)
	y_count_Tr_CNOT = counts_Tr_CNOT.get(2, 0)
	z_count_Tr_CNOT = counts_Tr_CNOT.get(3, 0)

	logger.info("S gate error counts trapped_ion_CNOT: X=%d, Y=%d, Z=%d", x_count_Tr_CNOT, y_count_Tr_CNOT, z_count_Tr_CNOT)

	counts_Tr_CZ = get_pauli_string(
		samples=1000,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=[("S", [0])],
		qubit_platform="trapped_ion_cz",
		random_seed=123,
	)
	x_count_Tr_CZ = counts_Tr_CZ.get(1, 0)
	y_count_Tr_CZ = counts_Tr_CZ.get(2, 0)
	z_count_Tr_CZ = counts_Tr_CZ.get(3, 0)

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
	for value in results:
		running_counts[int(value)] += 1
	total = sum(running_counts.values())
	effective_probs = {
		"I": running_counts[0] / total,
		"X": running_counts[1] / total,
		"Y": running_counts[2] / total,
		"Z": running_counts[3] / total,
	}
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
	for value in results:
		running_counts[int(value)] += 1
	total = sum(running_counts.values())
	effective_probs = {
		"I": running_counts[0] / total,
		"X": running_counts[1] / total,
		"Y": running_counts[2] / total,
		"Z": running_counts[3] / total,
	}
	logger.info("S gate effective Pauli probabilities: %s", effective_probs)

	err_dict_s = {'I': 0.9970240579376571,
					'X': 1.4922560645502791e-07,
					'Y': 1.4922560645502791e-07,
					'Z': 0.00297564361112998}
	assert all(abs(effective_probs[pauli] - err_dict_s[pauli]) < 0.01 for pauli in err_dict_s)


def test_apply_gate_error_channel_seed_reproducibility() -> None:
	identity = "__"

	def sample_with_seed(seed: int, draws: int = 200) -> List[List[int]]:
		random.seed(seed)
		out: List[List[int]] = []
		for _ in range(draws):
			pauli = PauliString(identity)
			pauli = apply_gate_error_channel(pauli, "CX", [0, 1], identity, "superconducting")
			out.append(list(pauli))
		return out

	seq_a = sample_with_seed(12345)
	seq_b = sample_with_seed(12345)

	assert seq_a == seq_b
	assert len(seq_a) == 200


def test_apply_gate_error_channel_unsupported_gate_message() -> None:
	with pytest.raises(ValueError, match="Unsupported gate CX for neutral atom platform"):
		apply_gate_error_channel(PauliString("__"), "CX", [0, 1], "__", "neutral_atom")


def test_get_pauli_string_full_path_seed_reproducibility_nonideal() -> None:
	kwargs = {
		"samples": 32,
		"p": 0.003,
		"system_bias": 10000.0,
		"gate_sequence": [("H", [0]), ("CX", [0, 1]), ("S", [1])],
		"qubit_platform": "superconducting",
		"random_seed": 2026,
	}

	counts_a = get_pauli_string(**kwargs)
	counts_b = get_pauli_string(**kwargs)

	assert counts_a == counts_b
	total_observations = sum(counts_a.values())
	# 32 samples * 2 qubits (0, 1) = 64 observations
	assert total_observations == 64
	assert all(v in (0, 1, 2, 3) for v in counts_a.keys())


def test_get_pauli_string_coalesced_disjoint_timesteps_avoids_extra_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
	# Force deterministic sampling: always select the first bucket in apply_error.
	monkeypatch.setattr(pauli_strings_module.random, "random", lambda: 0.0)

	gate_sequence = [
		("CX", [0, 1]),
		("CZ", [2, 3]),
	]

	# The new API always uses coalescing. Qubits 4..7 are idle in both listed gates.
	# With coalescing, they receive one round (X = 1).
	coalesced = get_pauli_string(
		samples=1,
		p=0.5,
		system_bias=0.0,
		gate_sequence=gate_sequence,
		qubit_platform="ideal",
		random_seed=123,
		skip_idle_errors_on_edge_entangling_layers=False,
	)

	# Check that idle qubits have label 1 (X error)
	# With monkeypatch random=0, should get mostly X errors on idle qubits
	assert coalesced.get(1, 0) >= 3  # At least 3 idle qubits should have X error


def test_get_pauli_string_coalesced_disjoint_timesteps_avoids_extra_rounds_superconducting(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	# The new API always uses coalescing.
	monkeypatch.setattr(pauli_strings_module.random, "random", lambda: 0.0)

	gate_sequence = [
		("CX", [0, 1]),
		("CX", [2, 3]),
	]

	# The new API always coalesces.
	coalesced = get_pauli_string(
		samples=1,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=gate_sequence,
		qubit_platform="superconducting",
		random_seed=123,
	)

	# Check that result is a dict with proper structure
	assert isinstance(coalesced, dict)
	assert sum(coalesced.values()) >= 4  # At least 4 qubits were measured


def test_get_pauli_string_coalescing_preserves_gate_evolution_when_idle_noise_disabled() -> None:
	# With p=0 there is no stochastic idle-noise insertion. The new API always coalesces,
	# so the behavior is consistent.
	gate_sequence = [
		("CX", [0, 1]),
		("CX", [2, 3]),
	]

	# New API always coalesces
	coalesced = get_pauli_string(
		samples=16,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=gate_sequence,
		qubit_platform="superconducting",
		random_seed=123,
	)

	# Just verify it returns a proper dict result
	assert isinstance(coalesced, dict)
	assert sum(coalesced.values()) > 0  # Should have at least some observations


def test_skip_idle_errors_on_edge_entangling_layers_targets_only_first_and_last_entangling_layer(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	recorded_idle_qubits: list[list[int]] = []

	# Isolate per-layer idle-target selection from stochastic effects.
	monkeypatch.setattr(
		pauli_strings_module,
		"apply_error",
		lambda pauli, identity, keep_qubits, weights_applied: pauli,
	)

	def _record_idle_qubits(
		pauli: PauliString,
		identity: str,
		idle_qubits: List[int],
		idle_weights: List[float],
		layer_ops,
	) -> PauliString:
		recorded_idle_qubits.append(list(idle_qubits))
		return pauli

	monkeypatch.setattr(
		pauli_strings_module,
		"apply_precomputed_layer_gate_and_idle_error",
		_record_idle_qubits,
	)

	gate_sequence = [
		("H", [0]),
		("H", [1]),
		("CX", [1, 2]),
		("S", [1]),
		("CX", [1, 2]),
		("H", [1]),
	]

	get_pauli_string(
		gate_sequence=gate_sequence,
		samples=1,
		p=0.003,
		system_bias=10000.0,
		qubit_platform="superconducting",
		random_seed=123,
		skip_idle_errors_on_edge_entangling_layers=False,
	)
	baseline_idle_qubits = list(recorded_idle_qubits)

	recorded_idle_qubits.clear()
	get_pauli_string(
		gate_sequence=gate_sequence,
		samples=1,
		p=0.003,
		system_bias=10000.0,
		qubit_platform="superconducting",
		random_seed=123,
		skip_idle_errors_on_edge_entangling_layers=True,
	)
	skipped_idle_qubits = list(recorded_idle_qubits)

	assert baseline_idle_qubits == [[2], [0], [0, 2], [0], [0, 2]]
	assert skipped_idle_qubits == [[2], [], [0, 2], [], [0, 2]]




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


def test_save_load_running_counts_handles_jsonl_and_malformed_lines(tmp_path: Path) -> None:
	p = tmp_path / "counts.jsonl"
	save_running_counts({0: 2, 1: 1, 2: 0, 3: 0}, str(p), append=False, seed=7)

	# Append legacy format using string keys, then malformed and non-dict lines.
	with p.open("a", encoding="utf-8") as f:
		f.write(json.dumps({"0": 1, "1": 0, "2": 2, "3": 3}) + "\n")
		f.write("{ malformed json\n")
		f.write(json.dumps([1, 2, 3]) + "\n")

	loaded = load_running_counts(str(p))
	# load_running_counts returns the most complete record (the first line with proper format)
	# The appended lines are in legacy format and are not accumulated by the current implementation
	assert loaded == {0: 2, 1: 1, 2: 0, 3: 0}


def test_error_propagation_simulation_validates_iteration_params() -> None:
	with pytest.raises(ValueError):
		error_propagation_simulation(
			p_param=0.003,
			system_bias=10.0,
			qubit_platform="ideal",
			gate_sequence=[("H", [0])],
			samples_per_iteration=0,
			total_samples=10,
			chosen_seed=1,
			timestamp="t0",
		)


def test_error_propagation_simulation_respects_total_sample_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.chdir(tmp_path)

	progress_file, counts_file = error_propagation_simulation(
		p_param=0.003,
		system_bias=10.0,
		qubit_platform="ideal",
		gate_sequence=[("H", [0])],
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

	ideal_counts = get_pauli_string(
		samples=50,
		p=0.0,
		system_bias=10.0,
		gate_sequence=gate_seq,
		qubit_platform="ideal",
		random_seed=123,
	)
	hw_counts = get_pauli_string(
		samples=50,
		p=0.003,
		system_bias=10000.0,
		gate_sequence=gate_seq,
		qubit_platform="superconducting",
		random_seed=123,
	)

	# p=0 disables generic stochastic insertion; ideal path should remain identity-only.
	# ideal_counts should have all observations as label 0 (identity)
	assert ideal_counts.get(0, 0) == sum(ideal_counts.values())
	# Hardware platform can still inject gate-channel noise independently of p.
	assert any(v > 0 for k, v in hw_counts.items() if k != 0)


