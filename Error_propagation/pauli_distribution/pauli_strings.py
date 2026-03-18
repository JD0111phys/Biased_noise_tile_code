# pauli_strings.py
#Utilities for Pauli-error propagation sampling and convergence analysis.

#This module provides helpers to:
#- sample Pauli error strings under gate evolution,
#- apply platform-specific gate error channels,
#- accumulate and store running count statistics,
#- compute effective Pauli probabilities,
#- run iterative simulations until convergence.

from stim import PauliString, Tableau
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional, Any, cast
import json

GateOp = Tuple[str, List[int]]

#----------------- APPLY ERROR FUNCTIONS -----------------#


def apply_error(
        pauli: PauliString,
        identity: str,
        keep_qubits: List[int],
        weights: List[float]
) -> PauliString:
    """
    Applies random Pauli errors to a selected subset of qubits based on specified weights.

    Args:
        pauli: Initial Pauli string to modify.
        identity: A string of '_' of length equal to the number of qubits, used as the
            template for building the error Pauli string.
        keep_qubits: Indices of the qubits to which errors may be applied.
        weights: Sampling weights [pX, pY, pZ, pI] for Pauli error selection.

    Returns:
        Updated Pauli string with errors applied to the qubits in keep_qubits.
    """
    error = list(identity)
    # Applying Error probability to every qubit in pauli string
    for j in keep_qubits:
        error_char = random.choices(['X', 'Y', 'Z', 'I'], weights=weights, k=1)[0]
        error[j] = error_char
    error_str = ''.join(error)
    # Updating pauli string
    pauli *= PauliString(error_str)
    return pauli

def pairwise_tuples(lst: List[int]) -> List[Tuple[int, int]]:
    """
    Groups a flat list of integers into consecutive (control, target) pairs.

    Args:
        lst: A flat list of qubit indices with an even number of elements,
            arranged as [control0, target0, control1, target1, ...].

    Returns:
        A list of (control, target) tuples.

    Raises:
        ValueError: If lst contains an odd number of elements.
    """
    if len(lst) % 2 != 0:
        raise ValueError("List must contain an even number of elements.")
    return [(lst[i], lst[i+1]) for i in range(0, len(lst), 2)]

def apply_gate_error_channel(pauli: PauliString, gate_name: str, targets: List[int], identity: str, qubit_platform: str) -> PauliString:
    """
    Applies a hardware-specific gate error channel to a Pauli string.

    Samples a Pauli error from the gate's noise model and multiplies it into
    the input Pauli string.

    Args:
        pauli: The Pauli string to modify.
        gate_name: The Clifford gate whose error channel is applied. Supported
            values depend on the platform:
            - 'superconducting': 'CX', 'H', 'S', 'S_DAG'
            - 'trapped_ion_cnot': 'CX', 'H', 'S', 'S_DAG'
            - 'trapped_ion_cz': 'CZ', 'H', 'S', 'S_DAG'
            - 'neutral_atom': 'CZ', 'H', 'S', 'S_DAG'
            - 'ideal': any gate (no-op, no error applied)
        targets: Flat list of qubit indices for the gate. For two-qubit gates
            (CX, CZ) the list is interpreted as consecutive (control, target)
            pairs via pairwise_tuples.
        identity: A string of '_' of length equal to the compressed qubit-space
            size, used as the template for building the error Pauli string.
        qubit_platform: Hardware platform that determines which noise model to
            use. Supported values: 'superconducting', 'trapped_ion_cnot',
            'trapped_ion_cz', 'neutral_atom', 'ideal'.

    Returns:
        The modified Pauli string after the gate's error channel has been applied.

    Raises:
        ValueError: If gate_name is not supported for the given qubit_platform.
    """
    error = list(identity)
    if gate_name in ['CX', 'CZ']:
        tupled_targets = pairwise_tuples(targets)
    else:
        tupled_targets = []

    if qubit_platform == 'superconducting':
        # error probabilities for superconducting CX gate
        err_dict = {'II': 0.9971089697418947,
 'IX': 2.7934102919680015e-08,
 'IY': 2.3735336432129106e-06,
 'IZ': 0.0007172714715934017,
 'XI': 2.7711807848440628e-08,
 'XX': 2.5207188356080046e-07,
 'XY': 0.00023518786621833793,
 'XZ': 2.5886150639697902e-08,
 'YI': 2.3732742409909857e-06,
 'YX': 0.00023518786621884447,
 'YY': 2.5207741582294885e-07,
 'YZ': 2.5808276786498663e-08,
 'ZI': 0.0007125772927424889,
 'ZX': 2.5887020388415394e-08,
 'ZY': 2.579754807691126e-08,
 'ZZ': 0.0009853957792419835}
        err_dict_values = list(err_dict.values())
        if gate_name == 'CX':
            for control, target in tupled_targets:
                error_char = random.choices(['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY','XZ',
                                             'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'],
                                            weights=err_dict_values, k=1)[0]
                error_char_list = list(error_char)
                error[control] = error_char_list[0]
                error[target] = error_char_list[1]
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'H':
            #err_dict_1000_bias = {'II': 0.9982477375294029,
            #                        'XI': 0.0003010202390699157,
            #                        'YI': 0.0005806958705816545,
            #                        'ZI': 0.0008705463609456343}
            err_dict_1000_bias = {'II': 0.9970831318064503,
 'XI': 0.0004876108844646676,
 'YI': 0.0009721305333010022,
 'ZI': 0.0014571267757840511}
            err_dict_values = list(err_dict_1000_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'S' or gate_name == 'S_DAG':
            err_dict_1000_bias = {'II': 0.9970240579376571,
 'IX': 1.4922560645502791e-07,
 'IY': 1.4922560645502791e-07,
 'IZ': 0.00297564361112998}
            err_dict_values = list(err_dict_1000_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        else:
            raise ValueError(f"Unsupported gate {gate_name} for superconducting platform.")
    elif qubit_platform == 'trapped_ion_cnot':
        if gate_name == 'CX':
            for control, target in tupled_targets:
                # error probabilities for trapped ion CX gate
                err_dict = {'II': 0.9970978744492904,
 'IX': 3.3560719574221576e-07,
 'IY': 0.00028982952089751796,
 'IZ': 0.00038654305587448867,
 'XI': 0.0005800408405701243,
 'XX': 3.275762831961293e-07,
 'XY': 9.677438302829744e-05,
 'XZ': 2.860259729967063e-07,
 'YI': 0.0002907835269792061,
 'YX': 9.66060922966e-05,
 'YY': 1.386814579007467e-07,
 'YZ': 2.580023215695282e-07,
 'ZI': 0.0009661361035463306,
 'ZX': 1.1755776622990322e-07,
 'ZY': 3.137816749348987e-07,
 'ZZ': 0.00019363479484440366}
                err_dict_values = list(err_dict.values())
                error_char = random.choices(['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY','XZ',
                                             'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'],
                                            weights=err_dict_values, k=1)[0]
                error_char_list = list(error_char)
                error[control] = error_char_list[0]
                error[target] = error_char_list[1]
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'H':
            # error probabilities for trapped ion Hadamard gate
            err_dict_1000_bias = {'II': 0.9970831318064503,
 'XI': 0.0004876108844646676,
 'YI': 0.0009721305333010022,
 'ZI': 0.0014571267757840511}
            err_dict_values = list(err_dict_1000_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'S' or gate_name == 'S_DAG':
            err_dict_1000_bias = {'II': 0.9970240579376571,
 'IX': 1.4922560645502791e-07,
 'IY': 1.4922560645502791e-07,
 'IZ': 0.00297564361112998}
            err_dict_values = list(err_dict_1000_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        else:
            raise ValueError(f"Unsupported gate {gate_name} for trapped ion platform.")
    elif qubit_platform == 'trapped_ion_cz':
        if gate_name == 'CZ':
            for control, target in tupled_targets:
                # error probabilities for trapped ion CZ gate
                err_dict = {'II': 0.997064919955593,
 'IX': 2.6383893972359296e-08,
 'IY': 2.6383893986237084e-08,
 'IZ': 0.0009970331256010934,
 'XI': 2.6383893972359296e-08,
 'XX': 2.3561940468153075e-08,
 'XY': 2.3561940468153075e-08,
 'XZ': 2.3567419182857208e-08,
 'YI': 2.6383893972359296e-08,
 'YX': 2.3561940468153075e-08,
 'YY': 2.3561940468153075e-08,
 'YZ': 2.3567419182857208e-08,
 'ZI': 0.0009970331256011072,
 'ZX': 2.356741916897942e-08,
 'ZY': 2.3567419182857208e-08,
 'ZZ': 0.0009407197401905265}
                err_dict_values = list(err_dict.values())
                error_char = random.choices(['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY','XZ',
                                             'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'],
                                            weights=err_dict_values, k=1)[0]
                error_char_list = list(error_char)
                error[control] = error_char_list[0]
                error[target] = error_char_list[1]
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'H':
            # error probabilities for trapped ion Hadamard gate
            err_dict_100_bias ={'II': 0.9970831318064503,
 'XI': 0.0004876108844646676,
 'YI': 0.0009721305333010022,
 'ZI': 0.0014571267757840511}
            err_dict_values = list(err_dict_100_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'S' or gate_name == 'S_DAG':
            err_dict_100_bias = {'II': 0.9970240579376571,
 'IX': 1.4922560645502791e-07,
 'IY': 1.4922560645502791e-07,
 'IZ': 0.00297564361112998}
            err_dict_values = list(err_dict_100_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        else:
            raise ValueError(f"Unsupported gate {gate_name} for trapped ion_cz platform.")
    elif qubit_platform == 'neutral_atom':
        if gate_name == 'CZ':
            for control, target in tupled_targets:
                # error probabilities for neutral atom CZ gate
                err_dict = {'II': 0.997068734751876,
 'IX': 2.0905858801045785e-08,
 'IY': 2.090585879410689e-08,
 'IZ': 0.0009635217661934578,
 'XI': 2.0489466380502197e-08,
 'XX': 2.04892314434324e-08,
 'XY': 2.0489231478126868e-08,
 'XZ': 2.0489466380502197e-08,
 'YI': 2.0489466380502197e-08,
 'YX': 2.0489231471187974e-08,
 'YY': 2.0489231450371292e-08,
 'YZ': 2.048946638744109e-08,
 'ZI': 0.0008600034582691082,
 'ZX': 2.0905858801045785e-08,
 'ZY': 2.090585879410689e-08,
 'ZZ': 0.0009735718210506714}
                err_dict_values = list(err_dict.values())
                error_char = random.choices(['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY','XZ',
                                             'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ'],
                                            weights=err_dict_values, k=1)[0]
                error_char_list = list(error_char)
                error[control] = error_char_list[0]
                error[target] = error_char_list[1]
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'H':
            # error probabilities for neutral atom Hadamard gate
            err_dict_100_bias = {'II': 0.9970831318064503,
 'XI': 0.0004876108844646676,
 'YI': 0.0009721305333010022,
 'ZI': 0.0014571267757840511}
            err_dict_values = list(err_dict_100_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        elif gate_name == 'S' or gate_name == 'S_DAG':
            err_dict_100_bias = {'II': 0.9970240579376571,
 'IX': 1.4922560645502791e-07,
 'IY': 1.4922560645502791e-07,
 'IZ': 0.00297564361112998}
            err_dict_values = list(err_dict_100_bias.values())
            for target in targets:
                error_char = random.choices(['I', 'X', 'Y', 'Z'],
                                            weights=err_dict_values, k=1)[0]
                error[target] = error_char
            error_str = ''.join(error)
            # Updating pauli string
            pauli *= PauliString(error_str)
        else:
            raise ValueError(f"Unsupported gate {gate_name} for neutral atom platform.")
    elif qubit_platform == 'ideal':
        # No error for ideal platform
        pass
    else:
        raise ValueError(f"Unsupported qubit platform: {qubit_platform}")

    # Placeholder for actual error channel logic
    # This function can be expanded based on specific error models for each gate
    return pauli  # No operation for now

#----------------- GATE FUNCTIONS -----------------#

def convert_gate_sequence(gate_sequence: List[GateOp], conversion_type: str) -> List[GateOp]:
    """
    Convert a gate sequence by decomposing gates into a platform-native two-qubit gate.

    Single-qubit helper gates (H, S, S_DAG) are inserted on the **target** qubit of
    each two-qubit pair (i.e., every second element of the qubit list: index 1, 3, 5…).

    Args:
        gate_sequence: List of (gate_name, qubit_indices) tuples describing the
            original circuit, where qubit_indices for two-qubit gates follows the
            flat [ctrl0, tgt0, ctrl1, tgt1, …] convention.
        conversion_type: Decomposition strategy to apply:
            - ``"CZ_native"``: Replaces CX with ``H CZ H`` and CY with
              ``S H CZ H S_DAG`` (helper gates on target qubits only).
            - ``"CNOT_native"``: Replaces CZ with ``H CX H`` and CY with
              ``S CX S_DAG`` (helper gates on target qubits only).
            Gates not matched by the strategy are passed through unchanged.

    Returns:
        A new list of (gate_name, qubit_indices) tuples with the same structure
        as gate_sequence, containing the decomposed gate operations.
    """
    converted_sequence: List[GateOp] = []
    
    for gate_type, qubits in gate_sequence:
        if conversion_type == "CZ_native" and gate_type == "CX":
            # Convert CX to H CZ H, with H on odd entries
            odd_qubits = [qubits[i] for i in range(1, len(qubits), 2)]
            converted_sequence.append(("H", odd_qubits))
            converted_sequence.append(("CZ", qubits))
            converted_sequence.append(("H", odd_qubits))
        elif conversion_type == "CZ_native" and gate_type == "CY":
            # Convert CY to S H CZ H SDAG, with S, H, and SDAG on odd entries
            odd_qubits = [qubits[i] for i in range(1, len(qubits), 2)]
            converted_sequence.append(("S", odd_qubits))
            converted_sequence.append(("H", odd_qubits))
            converted_sequence.append(("CZ", qubits))
            converted_sequence.append(("H", odd_qubits))
            converted_sequence.append(("S_DAG", odd_qubits))
        elif conversion_type == "CNOT_native" and gate_type == "CZ":
            # Convert CZ to H CX H, with H on odd entries  
            odd_qubits = [qubits[i] for i in range(1, len(qubits), 2)]
            converted_sequence.append(("H", odd_qubits))
            converted_sequence.append(("CX", qubits))
            converted_sequence.append(("H", odd_qubits))
        elif conversion_type == "CNOT_native" and gate_type == "CY":
            # Convert CY to S CX SDAG, with S and SDAG on odd entries
            odd_qubits = [qubits[i] for i in range(1, len(qubits), 2)]
            converted_sequence.append(("S", odd_qubits))
            converted_sequence.append(("CX", qubits))
            converted_sequence.append(("S_DAG", odd_qubits))
        else:
            # Keep the gate as is
            converted_sequence.append((gate_type, qubits))
    
    return converted_sequence

def gate_operation(
        pauli: PauliString,
        gate_name: str,
        targets: List[int],
        TABLEAUS: Dict[str, Tableau]
) -> PauliString:
    """
    Applies a Clifford gate under conjugation to a Pauli string.

    Computes ``G P G†`` for Clifford gate G and Pauli string P using the
    precomputed Tableau representation of G.

    Args:
        pauli: The Pauli string to transform.
        gate_name: Name of the Clifford gate (e.g., 'CX', 'CZ', 'H').
            Must be a key present in TABLEAUS.
        targets: Qubit indices the gate acts on, in the order expected by
            the gate's Tableau (e.g., [control, target] for CX/CZ).
        TABLEAUS: Mapping from gate name to its precomputed stim Tableau.

    Returns:
        The transformed Pauli string after gate conjugation.

    Raises:
        KeyError: If gate_name is not a key in TABLEAUS.
    """
    return pauli.after(TABLEAUS[gate_name], targets=targets)

#----------------- FUNCTION TO GET SAMPLES OF PAULI STRINGS -----------------#

def get_pauli_string(
        keep_qubits: List[int],
        samples: int = 100,
        p: float = 0.003,
        system_bias: float = 1000.0,
    gate_sequence: Optional[List[GateOp]] = None,
        ancilla: Optional[List[int]] = None,
        qubit_platform: str = "superconducting",
        random_seed: Optional[int] = None,
        use_compressed_space: bool = True
) -> List[int]:
    """
    Generates a flattened list of noisy Pauli values by simulating gate operations and error insertion.

    Args:
        keep_qubits: List of qubit indices to keep in the output (including ancilla).
        samples: Number of Pauli string samples to generate.
        p: Total single-qubit error probability, split as
            px = py = p / (2 + system_bias) and
            pz = p * system_bias / (2 + system_bias).
        system_bias: Dephasing bias parameter controlling the relative weight of
            Z errors against X/Y errors.
        gate_sequence: List of gates to apply, each as (gate_name, [targets]).
            For two-qubit gates, targets must be a flat list of control-target
            pairs: [ctrl0, tgt0, ctrl1, tgt1, ...].
        ancilla: List of ancilla qubit indices used for measurement errors.
        qubit_platform: The platform type of the qubits (e.g., "superconducting", "neutral_atom").
            "ideal" means no hardware-specific gate error channel is applied
            (native gate compilation), but generic stochastic error insertion
            is still performed.
        random_seed: Random seed for reproducible error generation (optional).
        use_compressed_space: If True, operates only on used qubits for massive speedup (default).

    Returns:
        A flattened list of Pauli values (0=I, 1=X, 2=Y, 3=Z) with length
        samples * len(sorted(set(keep_qubits) | set(ancilla))). Data is appended
        sample-by-sample, with qubits ordered by compressed index induced from
        sorted(set(keep_qubits) | set(ancilla)).

    Raises:
        NotImplementedError: If use_compressed_space is False.
        ValueError: If samples is negative, p is outside [0, 1], or system_bias is negative.
    """
    if gate_sequence is None:
        gate_sequence = []
    if ancilla is None:
        ancilla = []

    if samples < 0:
        raise ValueError("samples must be non-negative.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1].")
    if system_bias < 0.0:
        raise ValueError("system_bias must be non-negative.")

    # Set random seed if provided for reproducible results
    if random_seed is not None:
        random.seed(random_seed)

    if not use_compressed_space:
        raise NotImplementedError("Non-compressed path is not supported. Set use_compressed_space=True.")

    # Use compressed qubit space for massive speedup
    all_used_qubits = sorted(set(keep_qubits) | set(ancilla))
    compressed_size = len(all_used_qubits)
    # Create mapping: original_qubit_idx -> compressed_idx
    qubit_to_compressed = {original: compressed for compressed, original in enumerate(all_used_qubits)}
    
    # Compress gate sequences
    compressed_gate_sequence: List[GateOp] = []
    for gate_name, targets in gate_sequence:
        compressed_targets = [qubit_to_compressed[t] for t in targets if t in qubit_to_compressed]
        if compressed_targets:  # Only add if targets exist in compressed space
            compressed_gate_sequence.append((gate_name, compressed_targets))
    
    # Compress qubit lists
    compressed_keep_qubits = [qubit_to_compressed[q] for q in keep_qubits if q in qubit_to_compressed]
    compressed_ancilla = [qubit_to_compressed[q] for q in ancilla if q in qubit_to_compressed]
    
    identity = "_" * compressed_size
    effective_gate_sequence = compressed_gate_sequence
    effective_keep_qubits = compressed_keep_qubits
    effective_ancilla = compressed_ancilla
    # Interval [0,1]. X ->[0,px) Y -> [px,px+py) Z -> [px+py,p) I -> [p, 1 - p] p =px + py + pz
    pz = p * (system_bias / (system_bias + 2))  # Adjust pz based on bias 
    px = p / (2+system_bias)
    py = px
    p = px + py + pz
    weights = [px, py, pz, 1 - p]

    # Precompute gate's Tableau
    TABLEAU_CACHE = {name: Tableau.from_named_gate(name) for name, _ in effective_gate_sequence}
    flattened_result: List[int] = []
    if qubit_platform != 'ideal':
        for _ in range(samples):

            # Convert pauli string to stim's PauliString
            pauli = PauliString(identity)
            # INIT error
            pauli = apply_error(pauli, identity, effective_keep_qubits, weights) # init error |+> -> |->
            # Circuit
            for gate in effective_gate_sequence:
                # ERROR BEFORE/AFTER GATE
                #pauli = apply_error(pauli, identity, effective_keep_qubits, weights)
                # Gate operation under conjugation
                pauli = gate_operation(pauli, gate[0], gate[1], TABLEAUS=TABLEAU_CACHE)
                # GATE's error channel
                pauli = apply_gate_error_channel(pauli, gate[0], gate[1], identity, qubit_platform=qubit_platform)
                # ERROR IDLE Qubits during gate
                pauli = apply_error(pauli, identity, [j for j in effective_keep_qubits if j not in gate[1]], weights)

            # ERROR BEFORE MEASUREMENT
            pauli = apply_error(pauli, identity, effective_ancilla, weights)

            flattened_result.extend(list(pauli)) # type: ignore
    else:
        # Ideal compilation path: skip hardware-specific gate error channel.
        for _ in range(samples):
            pauli = PauliString(identity)
            # INIT error
            pauli = apply_error(pauli, identity, effective_keep_qubits, weights)
            for gate in effective_gate_sequence:
                pauli = gate_operation(pauli, gate[0], gate[1], TABLEAUS=TABLEAU_CACHE)
                pauli = apply_error(pauli, identity, effective_keep_qubits, weights)
            # ERROR BEFORE MEASUREMENT
            pauli = apply_error(pauli, identity, effective_ancilla, weights)

            flattened_result.extend(list(pauli)) # type: ignore

    return flattened_result

#----------------- STORING DATA FUNCTIONS -----------------#

def save_running_counts(running_counts: Dict[int, int], output_file: str, append: bool = False, seed: Optional[int] = None) -> None:
    """
    Save running counts and seed to a JSONL file.
    
    Args:
        running_counts: Dictionary with counts {0: I_count, 1: X_count, 2: Y_count, 3: Z_count}.
        output_file: Path to the output file.
        append: If True, append one JSON object line; if False, overwrite file.
        seed: Optional random seed used for this iteration.

    Notes:
        Each write appends a single JSON object followed by a newline (JSONL).
    """
    mode = 'a' if append else 'w'
    data: Dict[str, Any] = {
        'counts': running_counts,
        'seed': seed
    } 
    with open(output_file, mode) as f:
        json.dump(data, f)
        f.write('\n')

def load_running_counts(input_file: str) -> Dict[int, int]:
    """
    Load cumulative running counts from a JSONL file.
    
    Args:
        input_file: Path to the input file.
        
    Returns:
        Dictionary with cumulative counts {0: I_count, 1: X_count, 2: Y_count, 3: Z_count}.

    Notes:
        - Supports legacy lines that store counts directly (without a 'counts' key).
        - Ignores seed metadata if present.
        - Skips malformed JSON lines and continues processing.
    """
    running_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        # Keep loading valid lines even if one line is malformed.
                        continue
                    if not isinstance(data, dict):
                        continue
                    data_dict = cast(Dict[Any, Any], data)
                    # Handle both old format (direct counts) and new format (with seed)
                    if 'counts' in data_dict:
                        counts = data_dict['counts']
                    else:
                        counts = data_dict  # Backward compatibility with old format
                    if not isinstance(counts, dict):
                        continue
                    counts_dict = cast(Dict[Any, Any], counts)
                    for pauli_type in [0, 1, 2, 3]:
                        running_counts[pauli_type] += int(
                            counts_dict.get(str(pauli_type), counts_dict.get(pauli_type, 0))
                        )
    except FileNotFoundError:
        pass  # Return initialized counts if file doesn't exist
    return running_counts

def update_running_counts(
    running_counts: Dict[int, int], 
    new_pauli_values: List[int]
) -> Dict[int, int]:
    """
    Update running counts in place with new samples.
    
    Args:
        running_counts: Current counts for each Pauli type {0: I_count, 1: X_count, ...}.
        new_pauli_values: Flattened list of new Pauli values (0=I, 1=X, 2=Y, 3=Z).
        
    Returns:
        The same running_counts dictionary after in-place update.
    """
    new_counts = Counter(new_pauli_values)
    
    # Add to running counts (assumes all pauli_types are pre-initialized)
    for pauli_type in [0, 1, 2, 3]:  # I, X, Y, Z
        running_counts[pauli_type] += new_counts.get(pauli_type, 0)
    
    return running_counts

#----------------- CALCULATE EFFECTIVE PROBABILITIES -----------------#

def effective_pauli_probabilities_from_counts(
    running_counts: Dict[int, int]
) -> Dict[str, float]:
    """
    Calculates effective probabilities from precomputed running counts.
    
    Args:
        running_counts: Dictionary with counts {0: I_count, 1: X_count, 2: Y_count, 3: Z_count}.
            Missing keys are treated as zero.
        
    Returns:
        Dictionary with fixed keys 'I', 'X', 'Y', 'Z' mapped to their effective
        probabilities. If total count is zero, all returned probabilities are 0.0.
    """
    total = sum(running_counts.values())
    pauli_map = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    return {pauli_map[k]: running_counts.get(k, 0) / total if total > 0 else 0.0 for k in pauli_map}

#----------------- MAIN SIMULATION FUNCTION WITH CONVERGENCE CHECK -----------------#

def error_propagation_simulation(
    keep_qubits: List[int], ancilla: List[int],
    p_param: float, system_bias: float, qubit_platform: str,
    gate_sequence: List[GateOp],
    samples_per_iteration: int,total_samples: int, chosen_seed: int,
    timestamp: str
    ) -> Tuple[str,str]: 
    """
    Run iterative Pauli-error propagation simulation until convergence or sample cap.

    Per iteration, this function samples new Pauli outcomes, updates cumulative
    counts, writes progress to disk, and tracks convergence based on changes in
    effective X/Y/Z probabilities.

    Args:
        keep_qubits: Qubit indices included in the sampled output.
        ancilla: Ancilla qubit indices for measurement-error insertion.
        p_param: Total single-qubit error probability.
        system_bias: Dephasing bias parameter used to split p_param into X/Y/Z components.
        qubit_platform: Hardware platform noise model identifier.
        gate_sequence: Circuit gate sequence as (gate_name, targets) tuples.
        samples_per_iteration: Number of samples generated per iteration.
        total_samples: Maximum total sample budget used to compute max iterations.
        chosen_seed: Base random seed. Iteration i uses chosen_seed + i.
        timestamp: Suffix used in output filenames.

    Returns:
        Tuple of (progress_file, counts_file) output paths.

    Raises:
        ValueError: If samples_per_iteration is not positive or total_samples is negative.

    Notes:
        Output files are overwritten when the same timestamp and qubit_platform
        are reused.
    """
    if samples_per_iteration <= 0:
        raise ValueError("samples_per_iteration must be > 0.")
    if total_samples < 0:
        raise ValueError("total_samples must be >= 0.")

    initial_samples = min(samples_per_iteration, total_samples)
    new_samples = get_pauli_string(
        samples=initial_samples,
        keep_qubits=keep_qubits,
        p=p_param,
        system_bias=system_bias,
        qubit_platform=qubit_platform,
        gate_sequence=gate_sequence,
        ancilla=ancilla,
        random_seed=chosen_seed
    )

    # Initialize running counts for efficient probability calculation
    running_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # I, X, Y, Z
    # Update running counts with initial samples
    running_counts = update_running_counts(running_counts, new_samples)
    # Calculate effective probabilities
    effective_current = effective_pauli_probabilities_from_counts(running_counts)
    # Calculate bias for initial probabilities
    denominator = effective_current["X"] + effective_current["Y"]
    bias = effective_current["Z"] / denominator if denominator != 0 else float('inf')

    # Save initial running counts
    counts_file = f"running_counts_{qubit_platform}_{timestamp}.jsonl"
    save_running_counts(running_counts, counts_file, append=False, seed=chosen_seed)

    # Initialize progress file with header and initial results
    progress_file = f"effective_probs_{qubit_platform}_{timestamp}.txt"
    with open(progress_file, "w") as f:
        f.write("# Iteration,I_Probability,X_Probability,Y_Probability,Z_Probability,BIAS,I_Convergence,X_Convergence,Y_Convergence,Z_Convergence,Max_Convergence,Consecutive_Convergence_Count\n")
        f.write(f"0,{effective_current['I']:.8f},{effective_current['X']:.8f},{effective_current['Y']:.8f},{effective_current['Z']:.8f},{bias},Initial,Initial,Initial,Initial,Initial,0\n")

    convergence = 100.0
    iteration = 0
    generated_samples = initial_samples
    consecutive_convergence_count = 0  # Track consecutive iterations with convergence < 1e-05
    required_consecutive_iterations = 30  # Number of consecutive iterations required

    # convergence threshold is set to 1e-07 to ensure we are well below the 1e-05 target for bias convergence, accounting for fluctuations in X and Y probabilities
    while consecutive_convergence_count < required_consecutive_iterations and generated_samples < total_samples:
        iteration += 1
        current_batch_samples = min(samples_per_iteration, total_samples - generated_samples)
        
        # Use different random seed for each iteration
        current_seed = chosen_seed + iteration
        
        new_samples = get_pauli_string(
                samples=current_batch_samples,
                keep_qubits=keep_qubits,
                p=p_param,
                system_bias=system_bias,
                qubit_platform=qubit_platform,
                gate_sequence=gate_sequence,
                ancilla=ancilla,
                random_seed=current_seed
        )
        generated_samples += current_batch_samples
        
        # Update running counts efficiently instead of recalculating from entire history
        running_counts = update_running_counts(running_counts, new_samples)
        effective_new = effective_pauli_probabilities_from_counts(running_counts)
        
        # Save updated running counts
        save_running_counts(running_counts, counts_file, append=True, seed=current_seed)
        
        # Calculate absolute difference for convergence of all Pauli operators
        convergence_I = abs(effective_current["I"] - effective_new["I"])
        convergence_X = abs(effective_current["X"] - effective_new["X"])
        convergence_Y = abs(effective_current["Y"] - effective_new["Y"])
        convergence_Z = abs(effective_current["Z"] - effective_new["Z"])
        
        denominator = effective_new["X"] + effective_new["Y"]
        bias = effective_new["Z"] / denominator if denominator != 0 else float('inf')
        
        # Use maximum convergence across error operators only (I is redundant due to normalization)
        convergence = max(convergence_X, convergence_Y, convergence_Z)
        
        # Check convergence criteria and update consecutive count
        if convergence < 1e-07:
            consecutive_convergence_count += 1
        else:
            consecutive_convergence_count = 0  # Reset counter if convergence is not met
        
        # Save progress to same file (append)
        with open(progress_file, "a") as f:
            f.write(f"{iteration},{effective_new['I']:.8f},{effective_new['X']:.8f},{effective_new['Y']:.8f},{effective_new['Z']:.8f},{bias},{convergence_I:.2e},{convergence_X:.2e},{convergence_Y:.2e},{convergence_Z:.2e},{convergence:.2e},{consecutive_convergence_count}\n")
        
        effective_current = effective_new

    print(f"\nConvergence {'achieved' if consecutive_convergence_count >= required_consecutive_iterations else 'not achieved'} "
        f"after {iteration} iterations")
    print(f"Final probabilities: I={effective_current['I']:.8f}, X={effective_current['X']:.8f}, Y={effective_current['Y']:.8f}, Z={effective_current['Z']:.8f}")
    print(f"Final bias: {bias}")
    print(f"Final convergence value: {convergence:.2e}")
    print(f"Consecutive convergence iterations: {consecutive_convergence_count}/{required_consecutive_iterations}")

    # Append final summary to progress file
    with open(progress_file, "a") as f:
        f.write(f"# Final: Convergence {'achieved' if consecutive_convergence_count >= required_consecutive_iterations else 'not achieved'} after {iteration} iterations\n")
        f.write(f"# Final probabilities: I={effective_current['I']:.8f}, X={effective_current['X']:.8f}, Y={effective_current['Y']:.8f}, Z={effective_current['Z']:.8f}\n")
        f.write(f"# Final convergence value: {convergence:.2e}\n")
        f.write(f"# Final bias: {bias}\n")
        f.write(f"# Consecutive convergence iterations: {consecutive_convergence_count}/{required_consecutive_iterations}\n")
    
    return progress_file, counts_file
