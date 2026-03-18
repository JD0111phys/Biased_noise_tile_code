"""
Tile-code and inherited code-family circuit generation with biased Pauli noise.

This code is inspired by the circuit-level surface-code study framework of
Oscar Higgott, and adapts that style of circuit construction to the present
tile-code memory experiment using Stim + sinter + BP-OSD decoding.

Notes
-----
- The active rotated-memory construction in this file corresponds to the
  tile code used in this project.
- Some helper functions and task labels retain inherited naming such as
  "surface_code" and "toric_code" for compatibility with the original
  framework structure.
- The full logic and supported branches from the provided script are preserved.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import sinter
import stim
from stimbposd import sinter_decoders


def xyz_from_bias(p_total: float, r_bias: float) -> Tuple[float, float, float]:
    """Split total single-qubit error probability into X/Y/Z components."""
    r = float(r_bias)
    denom = r + 2.0
    pz = p_total * (r / denom)
    px = p_total * (1.0 / denom)
    py = p_total * (1.0 / denom)
    return px, py, pz


@dataclass
class CircuitGenParameters:
    code_name: str
    task: str
    rounds: int
    distance: Optional[int] = None
    x_distance: Optional[int] = None
    z_distance: Optional[int] = None
    after_clifford_depolarization: float = 0.0
    before_round_data_depolarization: float = 0.0
    before_measure_flip_probability: float = 0.0
    after_reset_flip_probability: float = 0.0
    after_single_clifford_probability: float = 0.0
    exclude_other_basis_detectors: bool = False
    bias: float = 10000.0

    def append_begin_round_tick(self, circuit: stim.Circuit, data_qubits: List[int]) -> None:
        circuit.append_operation("TICK", [])
        if self.before_round_data_depolarization > 0:
            px, py, pz = xyz_from_bias(self.before_round_data_depolarization, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", data_qubits, [px, py, pz])

    def append_unitary_1(self, circuit: stim.Circuit, name: str, targets: List[int]) -> None:
        circuit.append_operation(name, targets)
        if self.after_clifford_depolarization > 0:
            px, py, pz = xyz_from_bias(self.after_clifford_depolarization, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])

    def append_unitary_2(self, circuit: stim.Circuit, name: str, targets: List[int]) -> None:
        circuit.append_operation(name, targets)
        if self.after_clifford_depolarization > 0:
            px, py, pz = xyz_from_bias(self.after_clifford_depolarization, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])

    def append_unitary_3(self, circuit: stim.Circuit, name: str, targets: List[int]) -> None:
        circuit.append_operation(name, targets)
        if self.after_single_clifford_probability > 0:
            px, py, pz = xyz_from_bias(self.after_single_clifford_probability, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])

    def append_reset(self, circuit: stim.Circuit, targets: List[int], basis: str) -> None:
        circuit.append_operation("R" + basis, targets)
        if self.after_reset_flip_probability > 0:
            px, py, pz = xyz_from_bias(self.after_reset_flip_probability, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])

    def append_measure(self, circuit: stim.Circuit, targets: List[int], basis: str) -> None:
        if self.before_measure_flip_probability > 0:
            px, py, pz = xyz_from_bias(self.before_measure_flip_probability, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])
        circuit.append_operation("M" + basis, targets)

    def append_measure_reset(self, circuit: stim.Circuit, targets: List[int], basis: str) -> None:
        if self.before_measure_flip_probability > 0:
            px, py, pz = xyz_from_bias(self.before_measure_flip_probability, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])
        circuit.append_operation("MR" + basis, targets)
        if self.after_reset_flip_probability > 0:
            px, py, pz = xyz_from_bias(self.after_reset_flip_probability, self.bias)
            circuit.append_operation("PAULI_CHANNEL_1", targets, [px, py, pz])


def finish_tile_code_circuit(
    coord_to_index: Callable[[complex], int],
    data_coords: Set[complex],
    x_measure_coords: Set[complex],
    z_measure_coords: Set[complex],
    params: CircuitGenParameters,
    x_order: List[complex],
    z_order: List[complex],
    x_observables: List[List[complex]],
    z_observables: List[List[complex]],
    is_memory_x: bool,
    *,
    exclude_other_basis_detectors: bool = False,
    wraparound_length: Optional[int] = None,
) -> stim.Circuit:
    """Finalize the circuit by assembling the head, body, and tail."""
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")
    if params.x_distance is not None and (params.x_distance < 2 or params.z_distance < 2):
        raise ValueError("Need a distance >= 2")

    chosen_basis_observable = x_observables if is_memory_x else z_observables
    chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords

    p2q: Dict[complex, int] = {}
    for q in data_coords:
        p2q[q] = coord_to_index(q)
    for q in x_measure_coords:
        p2q[q] = coord_to_index(q)
    for q in z_measure_coords:
        p2q[q] = coord_to_index(q)

    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}

    data_qubits = [p2q[q] for q in data_coords]
    measurement_qubits = [p2q[q] for q in x_measure_coords]
    measurement_qubits += [p2q[q] for q in z_measure_coords]
    x_measurement_qubits = [p2q[q] for q in x_measure_coords]
    z_measurement_qubits = [p2q[q] for q in z_measure_coords]

    vertical_coords = {
        coord for coord in data_coords
        if int(coord.real) % 4 == 0 and int(coord.imag) % 4 == 2
    }
    vertical_qubits = [p2q[q] for q in vertical_coords]
    horizontal_qubits = [p2q[q] for q in data_coords if q not in vertical_coords]
    _ = horizontal_qubits  # Preserved from the original script.

    all_qubits: List[int] = []
    all_qubits += data_qubits + measurement_qubits

    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()

    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}

    for q in data_qubits:
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)

    cnotx_targets: List[List[int]] = [[], [], [], [], [], []]
    czx_targets: List[List[int]] = [[], [], [], [], [], []]
    cnotz_targets: List[List[int]] = [[], [], [], [], [], []]
    czz_targets: List[List[int]] = [[], [], [], [], [], []]

    for k in range(6):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                ctrl = p2q[measure]
                target = p2q[data]
                if k in {1, 3, 5}:
                    czx_targets[k].extend([ctrl, target])
                else:
                    cnotx_targets[k].extend([ctrl, target])

        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                ctrl = p2q[measure]
                target = p2q[data]
                if k in {0, 2, 4}:
                    cnotz_targets[k].extend([ctrl, target])
                else:
                    czz_targets[k].extend([ctrl, target])

    cycle_actions = stim.Circuit()
    params.append_begin_round_tick(cycle_actions, data_qubits)

    for k in range(6):
        if cnotx_targets[k]:
            params.append_unitary_2(cycle_actions, "CNOT", cnotx_targets[k])
        if czx_targets[k]:
            params.append_unitary_2(cycle_actions, "CZ", czx_targets[k])
        cycle_actions.append_operation("TICK", [])

    for k in range(6):
        if cnotz_targets[k]:
            params.append_unitary_2(cycle_actions, "CNOT", cnotz_targets[k])
        if czz_targets[k]:
            params.append_unitary_2(cycle_actions, "CZ", czz_targets[k])
        cycle_actions.append_operation("TICK", [])

    params.append_measure(cycle_actions, measurement_qubits, "X")
    cycle_actions.append_operation("TICK", [])
    params.append_reset(cycle_actions, measurement_qubits, "X")
    cycle_actions.append_operation("TICK", [])

    head = stim.Circuit()
    for q, coord in sorted(q2p.items()):
        head.append_operation("QUBIT_COORDS", [q], [coord.real, coord.imag])

    params.append_reset(head, data_qubits, "ZX"[is_memory_x])
    params.append_reset(head, measurement_qubits, "X")
    head.append_operation("TICK", [])
    params.append_unitary_3(head, "H", vertical_qubits)

    head += cycle_actions

    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[measure])],
            [measure.real, measure.imag, 0.0],
        )

    body = cycle_actions.copy()
    m = len(measurement_qubits)
    body.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

    for m_index in sorted(measurement_qubits, key=lambda c: (c.real, c.imag)):
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            body.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0],
            )

    tail = stim.Circuit()
    tail.append_operation("TICK", [])
    params.append_unitary_3(tail, "H", vertical_qubits)
    tail.append_operation("TICK", [])
    params.append_measure(tail, data_qubits, "X")

    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detectors: List[int] = []
        for delta in x_order:
            data = measure + delta
            if data in p2q:
                detectors.append(-len(data_qubits) + data_coord_to_order[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                detectors.append(-len(data_qubits) + data_coord_to_order[data_wrapped])

        detectors.append(-len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure])
        detectors.sort(reverse=True)

        tail.append_operation(
            "DETECTOR",
            [stim.target_rec(x) for x in detectors],
            [measure.real, measure.imag, 1.0],
        )

    for obs_id, logical in enumerate(chosen_basis_observable):
        obs_inc: List[int] = []
        for q in logical:
            obs_inc.append(-len(data_qubits) + data_coord_to_order[q])
        obs_inc.sort(reverse=True)
        tail.append_operation(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(x) for x in obs_inc],
            obs_id,
        )

    return head + body * (params.rounds - 1) + tail


def generate_rotated_tile_code_circuit(
    params: CircuitGenParameters,
    is_memory_x: bool,
) -> stim.Circuit:
    """Generate the rotated tile-code circuit used in this project."""
    if params.distance is not None:
        x_distance = params.distance
        z_distance = params.distance
    else:
        x_distance = params.x_distance
        z_distance = params.z_distance

    l = x_distance
    m = z_distance
    B = 3

    def get_edge_indices(lx: int, my: int) -> List[Tuple[Tuple[int, int], str]]:
        h_edges = [((x, y), "h") for y in range(my) for x in range(lx)]
        v_edges = [((x, y), "v") for y in range(my) for x in range(lx)]
        return h_edges + v_edges

    edges = get_edge_indices(l, m)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    idx_to_edge = {i: e for e, i in edge_to_idx.items()}
    num_edges = len(edges)

    red_h_offsets = [(0, 0), (2, 1), (2, 2)]
    red_v_offsets = [(0, 2), (1, 2), (2, 0)]
    blue_h_offsets = [(0, 2), (1, 0), (2, 0)]
    blue_v_offsets = [(0, 0), (0, 1), (2, 2)]

    def get_stabilizer_support(
        anchor: Tuple[int, int],
        h_offsets: List[Tuple[int, int]],
        v_offsets: List[Tuple[int, int]],
        lx: int,
        my: int,
    ) -> List[int]:
        x0, y0 = anchor
        support: List[int] = []

        for dx, dy in h_offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < lx and 0 <= y < my:
                idx = edge_to_idx.get(((x, y), "h"))
                if idx is not None:
                    support.append(idx)

        for dx, dy in v_offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < lx and 0 <= y < my:
                idx = edge_to_idx.get(((x, y), "v"))
                if idx is not None:
                    support.append(idx)

        return sorted(support)

    bulk_anchors = [(x, y) for x in range(l - B + 1) for y in range(m - B + 1)]
    x_boundary_anchors = [(x, y) for x in range(l - B + 1) for y in [-2, -1, m - B + 1, m - B + 2]]
    z_boundary_anchors = [(x, y) for x in [-2, -1, l - B + 1, l - B + 2] for y in range(m - B + 1)]

    red_stabilizers = [
        get_stabilizer_support(anchor, red_h_offsets, red_v_offsets, l, m)
        for anchor in bulk_anchors + x_boundary_anchors
    ]
    blue_stabilizers = [
        get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets, l, m)
        for anchor in bulk_anchors + z_boundary_anchors
    ]

    qubit_touched = np.zeros(num_edges, dtype=bool)
    for stab in red_stabilizers + blue_stabilizers:
        for q in stab:
            qubit_touched[q] = True

    old_to_new: Dict[int, int] = {}
    data_qubit_coords: Dict[int, Tuple[int, int]] = {}
    new_idx = 0
    for i, touched in enumerate(qubit_touched):
        if touched:
            old_to_new[i] = new_idx
            (x, y), orientation = idx_to_edge[i]
            data_qubit_coords[new_idx] = (4 * x + 2, 4 * y) if orientation == "h" else (4 * x, 4 * y + 2)
            new_idx += 1

    bulk_limit = (l - 2) ** 2

    hhx_ancilla_coords: Dict[int, complex] = {}
    hhz_ancilla_coords: Dict[int, complex] = {}
    _ = hhx_ancilla_coords, hhz_ancilla_coords  # Preserved names from the original script.

    hx_ancilla_coords: Dict[int, complex] = {}
    for i in range(len(red_stabilizers)):
        if i < bulk_limit:
            real_part = 4 * (i // (l - 2))
            imag_part = 4 * (i % (l - 2))
        else:
            offset = i - bulk_limit
            col = offset // 4
            row = offset % 4
            real_part = 4 * col
            imag_part = [-8, -4, 4 * (l - 2), 4 * (l - 1)][row]
        hx_ancilla_coords[i] = complex(real_part, imag_part)

    hz_ancilla_coords: Dict[int, complex] = {}
    for i in range(len(blue_stabilizers)):
        if i < bulk_limit:
            real_part = 4 * (i // (l - 2))
            imag_part = 4 * (i % (l - 2))
        else:
            offset = i - bulk_limit
            col = offset // (l - 2)
            row = offset % (l - 2)
            imag_part = 4 * row
            real_part = [-8, -4, 4 * (l - 2), 4 * (l - 1)][col]
        hz_ancilla_coords[i] = complex(real_part, imag_part)

    hz_ancilla_coords = {i: c.real + 1 + 1j * c.imag for i, c in hz_ancilla_coords.items()}
    hx_ancilla_coords = {i: c.real - 1 + 1j * c.imag for i, c in hx_ancilla_coords.items()}

    index_to_coord_data = {i: x + 1j * y for i, (x, y) in data_qubit_coords.items()}
    data_coords: Set[complex] = set(index_to_coord_data.values())
    x_measure_coords: Set[complex] = set(hx_ancilla_coords.values())
    z_measure_coords: Set[complex] = set(hz_ancilla_coords.values())

    data = np.load(f"code_data/tilecode_l{l}.npz")
    H_in = data["H_in"]
    lx = data["lx"]
    lz = data["lz"]

    x_observables: List[List[complex]] = []
    z_observables: List[List[complex]] = []

    for row in lx:
        support = np.flatnonzero(row)
        coords = [index_to_coord_data[i] for i in support]
        x_observables.append(coords)

    for row in lz:
        support = np.flatnonzero(row)
        coords = [index_to_coord_data[i] for i in support]
        z_observables.append(coords)

    _ = H_in  # Preserved from the original script.

    x_observable = x_observables[0] if x_observables else []
    z_observable = z_observables[0] if z_observables else []
    _ = x_observable, z_observable  # Preserved from the original script.

    all_coords = list(data_coords | x_measure_coords | z_measure_coords)
    min_x = int(min(c.real for c in all_coords))
    min_y = int(min(c.imag for c in all_coords))
    max_x = int(max(c.real for c in all_coords))
    max_y = int(max(c.imag for c in all_coords))
    width = max_x - min_x + 1
    _ = max_y  # Preserved computation structure from the original script.

    def coord_to_idx(q: complex) -> int:
        x = int(round(q.real)) - min_x
        y = int(round(q.imag)) - min_y
        return x + y * width

    x_order: List[complex] = [
        1 + 2 + 0j,
        1 + 10j,
        1 + 10 + 4j,
        1 + 4 + 10j,
        1 + 10 + 8j,
        1 + 8 + 2j,
    ]
    z_order: List[complex] = [
        -1 + 2j,
        -1 + 2 + 8j,
        -1 + 8 + 10j,
        -1 + 10 + 0j,
        -1 + 0 + 6j,
        -1 + 6 + 0j,
    ]

    return finish_tile_code_circuit(
        coord_to_idx,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        x_order,
        z_order,
        x_observables,
        z_observables,
        is_memory_x,
        exclude_other_basis_detectors=params.exclude_other_basis_detectors,
    )


def generate_unrotated_surface_or_toric_code_circuit(
    params: CircuitGenParameters,
    is_memory_x: bool,
    is_toric: bool,
) -> stim.Circuit:
    """Generate the inherited unrotated surface-code or toric-code circuit."""
    d = params.distance
    assert params.rounds > 0

    data_coords: Set[complex] = set()
    x_measure_coords: Set[complex] = set()
    z_measure_coords: Set[complex] = set()
    x_observable: List[complex] = []
    z_observable: List[complex] = []

    length = 2 * d if is_toric else 2 * d - 1
    for x in range(length):
        for y in range(length):
            q = x + y * 1j
            parity = (x % 2) != (y % 2)
            if parity:
                if x % 2 == 0:
                    z_measure_coords.add(q)
                else:
                    x_measure_coords.add(q)
            else:
                data_coords.add(q)
                if x == 0:
                    x_observable.append(q)
                if y == 0:
                    z_observable.append(q)

    order: List[complex] = [1, 1j, -1j, -1]

    def coord_to_idx(q: complex) -> int:
        return int(q.real + q.imag * length)

    return finish_tile_code_circuit(
        coord_to_idx,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        order,
        order,
        x_observable,
        z_observable,
        is_memory_x,
        exclude_other_basis_detectors=params.exclude_other_basis_detectors,
        wraparound_length=2 * d if is_toric else None,
    )


def generate_circuit_from_params(params: CircuitGenParameters) -> stim.Circuit:
    """Dispatch circuit generation from a parameter object."""
    if params.code_name == "surface_code":
        if params.task == "rotated_memory_x":
            return generate_rotated_tile_code_circuit(params, True)
        if params.task == "rotated_memory_z":
            return generate_rotated_tile_code_circuit(params, False)
        if params.task == "unrotated_memory_x":
            if params.distance is None:
                raise NotImplementedError("Rectangular unrotated memories are not currently supported")
            return generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=True,
                is_toric=False,
            )
        if params.task == "unrotated_memory_z":
            if params.distance is None:
                raise NotImplementedError("Rectangular unrotated memories are not currently supported")
            return generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=False,
                is_toric=False,
            )

    if params.code_name == "toric_code":
        if params.distance is None:
            raise NotImplementedError("Rectangular toric codes are not currently supported")
        if params.task == "unrotated_memory_x":
            return generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=True,
                is_toric=True,
            )
        if params.task == "unrotated_memory_z":
            return generate_unrotated_surface_or_toric_code_circuit(
                params=params,
                is_memory_x=False,
                is_toric=True,
            )

    raise ValueError(f"Unrecognised task: {params.task}")


def generate_circuit(
    code_task: str,
    *,
    rounds: int,
    distance: Optional[int] = None,
    x_distance: Optional[int] = None,
    z_distance: Optional[int] = None,
    after_clifford_depolarization: float = 0.0,
    before_round_data_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    after_reset_flip_probability: float = 0.0,
    after_single_clifford_probability: float = 0.0,
    exclude_other_basis_detectors: bool = False,
) -> stim.Circuit:
    """
    Generate common circuits.

    Supported inherited task labels:
        - surface_code:rotated_memory_x
        - surface_code:rotated_memory_z
        - surface_code:unrotated_memory_x
        - surface_code:unrotated_memory_z
        - toric_code:unrotated_memory_x
        - toric_code:unrotated_memory_z
    """
    if distance is not None:
        pass
    elif x_distance is not None and z_distance is not None:
        pass
    else:
        raise ValueError("Either the distance parameter or x_distance and z_distance parameters must be specified")

    code_name, task = code_task.split(":")
    if code_name in ["surface_code", "toric_code"]:
        params = CircuitGenParameters(
            code_name=code_name,
            task=task,
            rounds=rounds,
            distance=distance,
            x_distance=x_distance,
            z_distance=z_distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            after_single_clifford_probability=after_single_clifford_probability,
            exclude_other_basis_detectors=exclude_other_basis_detectors,
        )
        return generate_circuit_from_params(params)

    raise ValueError(f"Code name {code_name} not recognised")


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")

    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("l", type=int)
    parser.add_argument("m", type=int)
    parser.add_argument("--bias", type=float, required=True)
    args = parser.parse_args()

    l = args.l
    m = args.m
    bias = args.bias

    print(f"[INFO] Running for l = {l}, m = {m}, Z-bias = {bias}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"[INFO] Using {num_workers} sinter workers")

    num_trials = 100000
    error_rates = np.linspace(0.001, 0.03, 15)

    if not np.any(np.isclose(error_rates, 0.005, atol=1e-12, rtol=0.0)):
        error_rates = np.sort(np.append(error_rates, 0.005))

    error_rates = [float(x) for x in error_rates]

    tasks = []
    for p in error_rates:
        params = CircuitGenParameters(
            code_name="surface_code",
            task="rotated_memory_x",
            rounds=8,
            x_distance=l,
            z_distance=m,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            after_clifford_depolarization=p,
            after_single_clifford_probability=p,
            bias=bias,
        )

        circuit = generate_rotated_tile_code_circuit(params, is_memory_x=True)

        tasks.append(
            sinter.Task(
                circuit=circuit,
                json_metadata={
                    "l": int(l),
                    "m": int(m),
                    "p": float(p),
                    "bias": float(bias),
                },
            )
        )

    samples = list(
        sinter.collect(
            tasks=tasks,
            decoders=["bposd"],
            custom_decoders=sinter_decoders(),
            num_workers=num_workers,
            max_shots=num_trials,
            max_errors=25000,
        )
    )

    out_rows = []
    for s in samples:
        p = float(s.json_metadata["p"])
        n = int(s.shots)
        k = int(s.errors)
        ler = k / n if n else float("nan")
        lo, hi = wilson_interval(k, n)
        out_rows.append((p, ler, n, k, lo, hi))
        print(f"p={p:.6f} ler={ler:.6f} shots={n} errors={k} 95%CI=[{lo:.6f},{hi:.6f}]")

    filename = f"special_TILE_linear_Z_logical_error_round12_open_rates_l{l}_m{m}_bias{int(bias)}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["PhysicalErrorRate", "LogicalErrorRate", "Shots", "Errors", "CI_lower", "CI_upper"])
        for row in sorted(out_rows):
            writer.writerow(row)

    print(f"✔️  Saved results to '{filename}'")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()