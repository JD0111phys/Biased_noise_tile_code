"""
Tile-code circuit-level simulation with biased Pauli noise.

This code is inspired by the circuit-level surface-code study framework of
Oscar Higgott, and adapts that style of circuit construction to a tile-code
memory experiment with Stim + sinter + BP-OSD decoding.

The script:
    1. Builds a tile-code memory-X circuit with circuit-level noise.
    2. Runs sinter in parallel using the BP-OSD decoder.
    3. Saves logical error rates and confidence intervals to CSV.

Required external files:
    - code_data/tilecode_l{l}.npz
      containing arrays:
          H_in, lx, lz

Example usage:
    python run_tile_code.py 9 9 --bias 10000
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


def xyz_from_bias(p_total: float, z_bias: float) -> Tuple[float, float, float]:
    """
    Split a total single-qubit error probability into X/Y/Z components
    using Z-bias = pZ / pX = pZ / pY, with pX = pY.

    Args:
        p_total: Total error probability.
        z_bias: Relative Z bias.

    Returns:
        (pX, pY, pZ)
    """
    r = float(z_bias)
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
    hadamard_coords: Set[complex],
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
    """
    Final assembly of the tile-code circuit:
        - head
        - repeated stabilizer-measurement body
        - tail with final data measurement and logical observables
    """
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")
    if params.x_distance is not None and (
        params.x_distance < 2 or params.z_distance < 2
    ):
        raise ValueError("Need x_distance, z_distance >= 2")

    chosen_basis_observables = x_observables if is_memory_x else z_observables
    chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords

    # Coordinate ↔ qubit maps
    p2q: Dict[complex, int] = {}
    for q in data_coords | x_measure_coords | z_measure_coords:
        p2q[q] = coord_to_index(q)

    q2p: Dict[int, complex] = {v: k for k, v in p2q.items()}

    data_qubits = sorted(p2q[q] for q in data_coords)
    x_measurement_qubits = sorted(p2q[q] for q in x_measure_coords)
    z_measurement_qubits = sorted(p2q[q] for q in z_measure_coords)
    measurement_qubits = sorted(x_measurement_qubits + z_measurement_qubits)
    hadamard_qubits = sorted(p2q[q] for q in hadamard_coords)

    # Reverse order maps used when defining detectors
    data_coord_to_order: Dict[complex, int] = {}
    measure_coord_to_order: Dict[complex, int] = {}

    for q in data_qubits:
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)

    # Gate target lists per sub-round
    cnotx_targets: List[List[int]] = [[] for _ in range(6)]
    czx_targets: List[List[int]] = [[] for _ in range(6)]
    cnotz_targets: List[List[int]] = [[] for _ in range(6)]
    czz_targets: List[List[int]] = [[] for _ in range(6)]
    cyz_targets: List[List[int]] = [[] for _ in range(6)]

    # X-check interactions
    for k in range(6):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                ctrl = p2q[measure]
                target = p2q[data]
                if target in hadamard_qubits:
                    czx_targets[k].extend([ctrl, target])
                else:
                    cnotx_targets[k].extend([ctrl, target])

    # Z-check interactions
    for k in range(6):
        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                ctrl = p2q[measure]
                target = p2q[data]
                if target in hadamard_qubits:
                    cnotz_targets[k].extend([ctrl, target])
                elif k in [0, 2, 4]:
                    cyz_targets[k].extend([ctrl, target])
                else:
                    czz_targets[k].extend([ctrl, target])

    cycle_actions = stim.Circuit()
    params.append_begin_round_tick(cycle_actions, data_qubits)

    for k in range(6):
        if czx_targets[k]:
            params.append_unitary_2(cycle_actions, "CZ", czx_targets[k])
        if cnotx_targets[k]:
            params.append_unitary_2(cycle_actions, "CNOT", cnotx_targets[k])
        cycle_actions.append_operation("TICK", [])

    for k in range(6):
        if cnotz_targets[k]:
            params.append_unitary_2(cycle_actions, "CNOT", cnotz_targets[k])
        if czz_targets[k]:
            params.append_unitary_2(cycle_actions, "CZ", czz_targets[k])
        if cyz_targets[k]:
            params.append_unitary_2(cycle_actions, "CY", cyz_targets[k])
        cycle_actions.append_operation("TICK", [])

    params.append_measure(cycle_actions, measurement_qubits, "X")
    cycle_actions.append_operation("TICK", [])
    params.append_reset(cycle_actions, measurement_qubits, "X")
    cycle_actions.append_operation("TICK", [])

    # Head
    head = stim.Circuit()
    for q, coord in sorted(q2p.items()):
        head.append_operation("QUBIT_COORDS", [q], [coord.real, coord.imag])

    params.append_reset(head, data_qubits, "X" if is_memory_x else "Z")
    params.append_reset(head, measurement_qubits, "X")
    head.append_operation("TICK", [])
    params.append_unitary_3(head, "H", hadamard_qubits)

    head += cycle_actions

    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[measure])],
            [measure.real, measure.imag, 0.0],
        )

    # Body
    body = cycle_actions.copy()
    m = len(measurement_qubits)
    body.append_operation("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

    for m_index in sorted(measurement_qubits):
        m_coord = q2p[m_index]
        k = len(measurement_qubits) - measure_coord_to_order[m_coord] - 1
        if not exclude_other_basis_detectors or m_coord in chosen_basis_measure_coords:
            body.append_operation(
                "DETECTOR",
                [stim.target_rec(-k - 1), stim.target_rec(-k - 1 - m)],
                [m_coord.real, m_coord.imag, 0.0],
            )

    # Tail
    tail = stim.Circuit()
    tail.append_operation("TICK", [])
    params.append_unitary_3(tail, "H", hadamard_qubits)
    tail.append_operation("TICK", [])
    params.append_measure(tail, data_qubits, "X")

    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detector_targets: List[int] = []
        for delta in x_order:
            data = measure + delta
            if data in p2q:
                detector_targets.append(-len(data_qubits) + data_coord_to_order[data])
            elif wraparound_length is not None:
                data_wrapped = (data.real % wraparound_length) + (data.imag % wraparound_length) * 1j
                detector_targets.append(-len(data_qubits) + data_coord_to_order[data_wrapped])

        detector_targets.append(
            -len(data_qubits) - len(measurement_qubits) + measure_coord_to_order[measure]
        )
        detector_targets.sort(reverse=True)

        tail.append_operation(
            "DETECTOR",
            [stim.target_rec(x) for x in detector_targets],
            [measure.real, measure.imag, 1.0],
        )

    for obs_id, logical in enumerate(chosen_basis_observables):
        obs_inc = [-len(data_qubits) + data_coord_to_order[q] for q in logical]
        obs_inc.sort(reverse=True)
        tail.append_operation(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(x) for x in obs_inc],
            obs_id,
        )

    return head + body * (params.rounds - 1) + tail


def generate_hadamard_coords(m: int) -> Set[complex]:
    """
    Generate the subset of data-qubit coordinates receiving Hadamards.
    """
    coords: Set[complex] = set()

    for x in range(2, 4 * m, 12):
        for y in range(0, 4 * m, 4):
            coords.add(x + 1j * y)

    for x in range(6, 4 * m, 12):
        for y in range(4, 4 * m, 8):
            coords.add(x + 1j * y)

    return coords


def generate_tile_code_memory_x_circuit(params: CircuitGenParameters) -> stim.Circuit:
    """
    Generate the tile-code memory-X circuit used in this project.
    """
    if params.distance is not None:
        l = params.distance
        m = params.distance
    else:
        l = params.x_distance
        m = params.z_distance

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
            if orientation == "h":
                data_qubit_coords[new_idx] = (4 * x + 2, 4 * y)
            else:
                data_qubit_coords[new_idx] = (4 * x, 4 * y + 2)
            new_idx += 1

    bulk_limit = (l - 2) ** 2

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

    # Shift ancillas to avoid overlap
    hz_ancilla_coords = {i: c.real + 1 + 1j * c.imag for i, c in hz_ancilla_coords.items()}
    hx_ancilla_coords = {i: c.real - 1 + 1j * c.imag for i, c in hx_ancilla_coords.items()}

    index_to_coord_data = {i: x + 1j * y for i, (x, y) in data_qubit_coords.items()}
    data_coords: Set[complex] = set(index_to_coord_data.values())
    x_measure_coords: Set[complex] = set(hx_ancilla_coords.values())
    z_measure_coords: Set[complex] = set(hz_ancilla_coords.values())

    hadamard_coords = generate_hadamard_coords(m)

    data = np.load(f"code_data/tilecode_l{l}.npz")
    lx = data["lx"]
    lz = data["lz"]

    x_observables: List[List[complex]] = []
    z_observables: List[List[complex]] = []

    for row in lx:
        support = np.flatnonzero(row)
        x_observables.append([index_to_coord_data[i] for i in support])

    for row in lz:
        support = np.flatnonzero(row)
        z_observables.append([index_to_coord_data[i] for i in support])

    all_coords = list(data_coords | x_measure_coords | z_measure_coords)
    min_x = int(min(c.real for c in all_coords))
    min_y = int(min(c.imag for c in all_coords))
    max_x = int(max(c.real for c in all_coords))
    width = max_x - min_x + 1

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
        coord_to_index=coord_to_idx,
        data_coords=data_coords,
        x_measure_coords=x_measure_coords,
        z_measure_coords=z_measure_coords,
        hadamard_coords=hadamard_coords,
        params=params,
        x_order=x_order,
        z_order=z_order,
        x_observables=x_observables,
        z_observables=z_observables,
        is_memory_x=True,
        exclude_other_basis_detectors=params.exclude_other_basis_detectors,
    )


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
    bias: float = 10000.0,
) -> stim.Circuit:
    """
    Dispatch circuit generation.

    Currently this cleaned script only supports:
        surface_code:rotated_memory_x
    where "surface_code" here refers to your tile-code generator path.
    """
    if distance is None and (x_distance is None or z_distance is None):
        raise ValueError(
            "Either provide distance, or provide both x_distance and z_distance."
        )

    code_name, task = code_task.split(":")
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
        bias=bias,
    )

    if code_name == "surface_code" and task == "rotated_memory_x":
        return generate_tile_code_memory_x_circuit(params)

    raise ValueError(f"Unsupported code task in this cleaned script: {code_task}")


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson confidence interval for a binomial proportion.
    """
    if n == 0:
        return float("nan"), float("nan")

    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def build_tasks(
    l: int,
    m: int,
    bias: float,
    rounds: int,
    error_rates: List[float],
) -> List[sinter.Task]:
    """
    Build sinter tasks over the requested physical error rates.
    """
    tasks: List[sinter.Task] = []

    for p in error_rates:
        circuit = generate_circuit(
            "surface_code:rotated_memory_x",
            rounds=rounds,
            x_distance=l,
            z_distance=m,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
            after_clifford_depolarization=p,
            after_single_clifford_probability=p,
            bias=bias,
        )

        tasks.append(
            sinter.Task(
                circuit=circuit,
                json_metadata={
                    "l": int(l),
                    "m": int(m),
                    "p": float(p),
                    "bias": float(bias),
                    "rounds": int(rounds),
                },
            )
        )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tile-code circuit-level simulations with biased noise.")
    parser.add_argument("l", type=int, help="X-distance / first lattice size parameter")
    parser.add_argument("m", type=int, help="Z-distance / second lattice size parameter")
    parser.add_argument("--bias", type=float, required=True, help="Z-bias for the Pauli channel")
    parser.add_argument("--rounds", type=int, default=8, help="Number of syndrome-extraction rounds")
    parser.add_argument("--max_shots", type=int, default=100000, help="Maximum shots per physical error rate")
    parser.add_argument("--max_errors", type=int, default=25000, help="Stop early after this many logical errors")
    args = parser.parse_args()

    l = args.l
    m = args.m
    bias = args.bias
    rounds = args.rounds
    max_shots = args.max_shots
    max_errors = args.max_errors

    print(f"[INFO] Running for l={l}, m={m}, bias={bias}, rounds={rounds}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"[INFO] Using {num_workers} sinter workers")

    error_rates = np.linspace(0.001, 0.02, 15)
    if not np.any(np.isclose(error_rates, 0.005, atol=1e-12, rtol=0.0)):
        error_rates = np.sort(np.append(error_rates, 0.005))
    error_rates = [float(x) for x in error_rates]

    tasks = build_tasks(
        l=l,
        m=m,
        bias=bias,
        rounds=rounds,
        error_rates=error_rates,
    )

    samples = list(
        sinter.collect(
            tasks=tasks,
            decoders=["bposd"],
            custom_decoders=sinter_decoders(),
            num_workers=num_workers,
            max_shots=max_shots,
            max_errors=max_errors,
        )
    )

    out_rows = []
    for sample in samples:
        p = float(sample.json_metadata["p"])
        n = int(sample.shots)
        k = int(sample.errors)
        ler = k / n if n else float("nan")
        lo, hi = wilson_interval(k, n)

        out_rows.append((p, ler, n, k, lo, hi))
        print(
            f"p={p:.6f}  ler={ler:.6f}  shots={n}  errors={k}  "
            f"95% CI=[{lo:.6f}, {hi:.6f}]"
        )

    filename = (
        f"special_TILE_middle_Z_logical_error_round{rounds}"
        f"_open_rates_l{l}_m{m}_bias{int(bias)}.csv"
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["PhysicalErrorRate", "LogicalErrorRate", "Shots", "Errors", "CI_lower", "CI_upper"]
        )
        for row in sorted(out_rows):
            writer.writerow(row)

    print(f"[INFO] Saved results to '{filename}'")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()