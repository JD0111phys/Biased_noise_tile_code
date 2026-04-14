import itertools
import stim
import sys
import copy
from pathlib import Path

import numpy as np

# Dynamically resolve the repo root (two levels up from this file) so that
# the import works for any user who clones the repository.
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))
from finite_bias_circuit_level_simulation.circuit_level_css import *

# Override finish_tile_code_circuit to interleave the loops!
def finish_tile_code_circuit_interleaved(
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
    *,
    exclude_other_basis_detectors=False,
    wraparound_length=None,
):
    
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.distance is not None and params.distance < 2:
        raise ValueError("Need a distance >= 2")

    chosen_basis_observable = x_observables if is_memory_x else z_observables
    chosen_basis_measure_coords = x_measure_coords if is_memory_x else z_measure_coords

    p2q = {}
    for q in data_coords:
        p2q[q] = coord_to_idx(q)
    for q in x_measure_coords:
        p2q[q] = coord_to_idx(q)
    for q in z_measure_coords:
        p2q[q] = coord_to_idx(q)

    q2p = {v: k for k, v in p2q.items()}

    data_qubits = [p2q[q] for q in data_coords]
    measurement_qubits = [p2q[q] for q in x_measure_coords]
    measurement_qubits += [p2q[q] for q in z_measure_coords]
    x_measurement_qubits = [p2q[q] for q in x_measure_coords]
    z_measurement_qubits = [p2q[q] for q in z_measure_coords]

    all_qubits = data_qubits + measurement_qubits
    all_qubits.sort()
    data_qubits.sort()
    measurement_qubits.sort()
    x_measurement_qubits.sort()
    z_measurement_qubits.sort()

    data_coord_to_order = {}
    measure_coord_to_order = {}

    for q in data_qubits:
        data_coord_to_order[q2p[q]] = len(data_coord_to_order)
    for q in measurement_qubits:
        measure_coord_to_order[q2p[q]] = len(measure_coord_to_order)

    cnot_targets = [[], [], [], [], [], []]
    cz_targets = [[], [], [], [], [], []]

    for k in range(6):
        for measure in sorted(x_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + x_order[k]
            if data in p2q:
                cnot_targets[k].append(p2q[measure])
                cnot_targets[k].append(p2q[data])

        for measure in sorted(z_measure_coords, key=lambda c: (c.real, c.imag)):
            data = measure + z_order[k]
            if data in p2q:
                cz_targets[k].append(p2q[measure])
                cz_targets[k].append(p2q[data])

    cycle_actions = stim.Circuit()
    params.append_begin_round_tick(cycle_actions, data_qubits)

    # # =============== INTERLEAVED LOOP =================
    # for k in range(6):
    #     if cnot_targets[k]:
    #         params.append_unitary_2(cycle_actions, "CNOT", cnot_targets[k])
    #     if cz_targets[k]:
    #         params.append_unitary_2(cycle_actions, "CZ", cz_targets[k])
    #     cycle_actions.append_operation("TICK", [])
    # # ==================================================
        # Step 1: only CNOT at k = 0
    if cnot_targets[0]:
        params.append_unitary_2(cycle_actions, "CNOT", cnot_targets[0])
    cycle_actions.append_operation("TICK", [])

    # Step 2: for k = 1..5, do CNOT[k] and CZ[k-1] in the same tick layer
    for k in range(1, 6):
        if cnot_targets[k]:
            params.append_unitary_2(cycle_actions, "CNOT", cnot_targets[k])

        if cz_targets[k - 1]:
            params.append_unitary_2(cycle_actions, "CZ", cz_targets[k - 1])

        cycle_actions.append_operation("TICK", [])

    # Step 3: only final CZ at k = 5
    if cz_targets[5]:
        params.append_unitary_2(cycle_actions, "CZ", cz_targets[5])
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

    head += cycle_actions

    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        head.append_operation(
            "DETECTOR",
            [stim.target_rec(-len(measurement_qubits) + measure_coord_to_order[measure])],
            [measure.real, measure.imag, 0.0, 0.0 if measure in x_measure_coords else 3.0],
        )

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
                [m_coord.real, m_coord.imag, 0.0, 0.0 if m_coord in x_measure_coords else 3.0],
            )

    tail = stim.Circuit()
    params.append_measure(tail, data_qubits, "X")

    for measure in sorted(chosen_basis_measure_coords, key=lambda c: (c.real, c.imag)):
        detectors = []
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
            [measure.real, measure.imag, 1.0, 0.0 if measure in x_measure_coords else 3.0],
        )

    for obs_id, logical in enumerate(chosen_basis_observable):
        obs_inc = []
        for q in logical:
            obs_inc.append(-len(data_qubits) + data_coord_to_order[q])
        obs_inc.sort(reverse=True)
        tail.append_operation(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(x) for x in obs_inc],
            obs_id,
        )

    return head + body * (params.rounds - 1) + tail

def generate_tile_code_circuit_search(
    params,
    is_memory_x,
    custom_z_order
):
    l = params.distance if params.distance is not None else params.x_distance
    m = params.distance if params.distance is not None else params.z_distance
    B = 3

    def get_edge_indices(lx, my):
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

    def get_stabilizer_support(anchor, h_offsets, v_offsets, lx, my):
        x0, y0 = anchor
        support = []
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

    red_stabilizers = [get_stabilizer_support(a, red_h_offsets, red_v_offsets, l, m) for a in bulk_anchors + x_boundary_anchors]
    blue_stabilizers = [get_stabilizer_support(a, blue_h_offsets, blue_v_offsets, l, m) for a in bulk_anchors + z_boundary_anchors]

    qubit_touched = np.zeros(num_edges, dtype=bool)
    for stab in red_stabilizers + blue_stabilizers:
        for q in stab:
            qubit_touched[q] = True

    data_qubit_coords = {}
    new_idx = 0
    for i, touched in enumerate(qubit_touched):
        if touched:
            (x, y), orientation = idx_to_edge[i]
            if orientation == "h":
                data_qubit_coords[new_idx] = (4 * x + 2, 4 * y)
            else:
                data_qubit_coords[new_idx] = (4 * x, 4 * y + 2)
            new_idx += 1

    bulk_limit = (l - 2) ** 2

    hx_ancilla_coords = {}
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

    hz_ancilla_coords = {}
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
    data_coords = set(index_to_coord_data.values())
    x_measure_coords = set(hx_ancilla_coords.values())
    z_measure_coords = set(hz_ancilla_coords.values())

    data = np.load(f"code_data/tilecode_l{l}.npz")
    lx = data["lx"]
    lz = data["lz"]

    x_observables = []
    z_observables = []

    for row in lx:
        support = np.flatnonzero(row)
        coords = [index_to_coord_data[i] for i in support]
        x_observables.append(coords)

    for row in lz:
        support = np.flatnonzero(row)
        coords = [index_to_coord_data[i] for i in support]
        z_observables.append(coords)

    all_coords = list(data_coords | x_measure_coords | z_measure_coords)
    min_x = int(min(c.real for c in all_coords))
    min_y = int(min(c.imag for c in all_coords))
    max_x = int(max(c.real for c in all_coords))
    width = max_x - min_x + 1

    def coord_to_idx(q):
        x = int(round(q.real)) - min_x
        y = int(round(q.imag)) - min_y
        return x + y * width

    x_order = [
        1 + 2 + 0j,
        1 + 10j,
        1 + 10 + 4j,
        1 + 4 + 10j,
        1 + 10 + 8j,
        1 + 8 + 2j,
    ]

    return finish_tile_code_circuit_interleaved(
        coord_to_idx,
        data_coords,
        x_measure_coords,
        z_measure_coords,
        params,
        x_order,
        custom_z_order,
        x_observables,
        z_observables,
        is_memory_x,
        exclude_other_basis_detectors=params.exclude_other_basis_detectors,
    )

def main():
    original_z_order = [
        -1 + 2j,
        -1 + 2 + 8j,
        -1 + 8 + 10j,
        -1 + 10 + 0j,
        -1 + 0 + 6j,
        -1 + 6 + 0j,
    ]
    
    # 6! = 720 configurations
    p = CircuitGenParameters(
        code_name="tile_code",
        task="memory_x",
        rounds=4, # reduce rounds just for faster stim compilation testing
        x_distance=6,
        z_distance=6,
        bias=1.0, # no noise just for structural test
        exclude_other_basis_detectors=False
    )
    
    success_count = 0
    valid_permutations = []
    
    print("Testing 720 permutations for Z-check ordering...")
    for idx, perm in enumerate(itertools.permutations(original_z_order)):
        if idx % 100 == 0 and idx > 0:
            print(f"Tested {idx} / 720...")
        
        try:
            # 1) Generate the interleaved circuit using the specific z_order permutation
            circ = generate_tile_code_circuit_search(p, is_memory_x=True, custom_z_order=list(perm))
            
            # 2) Immediately check for non-deterministic detectors/commutativity failure!
            # If the circuit has anti-commuting measurements in the DEM generation phase, stim catches it identically.
            # We allow approximate disjoint errors and ignore decomposition failures so that IF noise is ever added,
            # decomposition errors aren't incorrectly caught as commutativity failures.
            dem = circ.detector_error_model(
                allow_gauge_detectors=False,
                decompose_errors=True,
                approximate_disjoint_errors=True,
                ignore_decomposition_failures=True,
            )            
            # If we get here, it succeeded!
            success_count += 1
            valid_permutations.append(list(perm))
            print(f"[FOUND] Permutation {idx} creates a valid deterministic schedule!")
            
        except Exception as e:
            # Most will fail with 'non-deterministic detector' because of commutativity rules
            pass

    print(f"Total valid schedules found: {success_count} / 720")
    if success_count > 0:
        print("All valid schedules:")
        for idx, sched in enumerate(valid_permutations):
            print(f"Schedule {idx + 1}: {sched}")
        
if __name__ == "__main__":
    main()
