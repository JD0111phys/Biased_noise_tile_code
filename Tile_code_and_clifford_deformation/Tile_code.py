"""
Open and periodic tile-code constructions.

This file provides two lattice constructions for the tile code:

1. Periodic tile code
2. Open tile code

Both constructions return the X-check matrix Hx, the Z-check matrix Hz,
and the full CSS parity-check matrix

    H_parity = [[Hx, 0],
                [0,  Hz]]

Usage
-----
Hx, Hz, H_parity = build_periodic_tile_code(l, m)

or

Hx, Hz, H_parity = build_open_tile_code(l, m)
"""

import numpy as np


# ============================================================
# Common helper functions
# ============================================================
def get_edge_indices(l, m):
    """
    Return the ordered list of horizontal and vertical edges
    on an l x m lattice.

    Ordering:
      1. all horizontal edges
      2. all vertical edges
    """
    h_edges = [((x, y), "h") for y in range(m) for x in range(l)]
    v_edges = [((x, y), "v") for y in range(m) for x in range(l)]
    return h_edges + v_edges


def stabilizer_to_vector(stab, length):
    """
    Convert a stabilizer support list to a binary parity-check row vector.
    """
    vec = np.zeros(length, dtype=int)
    for q in stab:
        vec[q] = 1
    return vec


def remap_stabilizers(red_stabilizers, blue_stabilizers, num_edges):
    """
    Remove unsupported qubits, remap qubit indices, and remove empty stabilizers.
    """
    qubit_touched = np.zeros(num_edges, dtype=bool)

    for stab in red_stabilizers + blue_stabilizers:
        for q in stab:
            qubit_touched[q] = True

    old_to_new = {}
    new_idx = 0
    for i, touched in enumerate(qubit_touched):
        if touched:
            old_to_new[i] = new_idx
            new_idx += 1

    num_qubits_final = new_idx

    def remap_stabilizer(stab):
        return [old_to_new[q] for q in stab if q in old_to_new]

    red_stabilizers = [remap_stabilizer(stab) for stab in red_stabilizers]
    blue_stabilizers = [remap_stabilizer(stab) for stab in blue_stabilizers]

    red_stabilizers = [stab for stab in red_stabilizers if len(stab) > 0]
    blue_stabilizers = [stab for stab in blue_stabilizers if len(stab) > 0]

    return red_stabilizers, blue_stabilizers, num_qubits_final


def build_parity_matrices(red_stabilizers, blue_stabilizers, num_qubits_final):
    """
    Build Hx, Hz, and the full CSS parity-check matrix H_parity.
    """
    Hx = np.array(
        [stabilizer_to_vector(stab, num_qubits_final) for stab in red_stabilizers],
        dtype=int,
    )
    Hz = np.array(
        [stabilizer_to_vector(stab, num_qubits_final) for stab in blue_stabilizers],
        dtype=int,
    )

    zeros_x = np.zeros((Hx.shape[0], Hz.shape[1]), dtype=int)
    zeros_z = np.zeros((Hz.shape[0], Hx.shape[1]), dtype=int)

    H_parity = np.block([[Hx, zeros_x], [zeros_z, Hz]])

    return Hx, Hz, H_parity


# ============================================================
# Periodic tile code
# ============================================================
def build_periodic_tile_code(l, m):
    """
    Construct the periodic tile code on an l x m lattice.

    In the periodic construction:
      - all anchors lie inside the lattice
      - stabilizer supports wrap around using periodic boundary conditions
    """
    edges = get_edge_indices(l, m)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)

    # Offsets relative to anchor (x0, y0)
    # Red (X-type)
    red_h_offsets = [(0, 0), (2, 1), (2, 2)]
    red_v_offsets = [(0, 2), (1, 2), (2, 0)]

    # Blue (Z-type)
    blue_h_offsets = [(0, 2), (1, 0), (2, 0)]
    blue_v_offsets = [(0, 0), (0, 1), (2, 2)]

    def get_stabilizer_support(anchor, h_offsets, v_offsets):
        x0, y0 = anchor
        support = []

        # Horizontal edges with periodic wrapping
        for dx, dy in h_offsets:
            x, y = (x0 + dx) % l, (y0 + dy) % m
            idx = edge_to_idx.get(((x, y), "h"))
            if idx is not None:
                support.append(idx)

        # Vertical edges with periodic wrapping
        for dx, dy in v_offsets:
            x, y = (x0 + dx) % l, (y0 + dy) % m
            idx = edge_to_idx.get(((x, y), "v"))
            if idx is not None:
                support.append(idx)

        return sorted(support)

    anchors = [(x, y) for x in range(l) for y in range(m)]

    red_stabilizers = []
    blue_stabilizers = []

    for anchor in anchors:
        red_stabilizers.append(
            get_stabilizer_support(anchor, red_h_offsets, red_v_offsets)
        )
        blue_stabilizers.append(
            get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets)
        )

    red_stabilizers, blue_stabilizers, num_qubits_final = remap_stabilizers(
        red_stabilizers, blue_stabilizers, num_edges
    )

    Hx, Hz, H_parity = build_parity_matrices(
        red_stabilizers, blue_stabilizers, num_qubits_final
    )

    return Hx, Hz, H_parity


# ============================================================
# Open tile code
# ============================================================
def build_open_tile_code(l, m, B=3):
    """
    Construct the open tile code on an l x m lattice.

    In the open construction:
      - bulk stabilizers are placed on interior anchors
      - boundary stabilizers are added from anchors outside the lattice
      - supports are truncated at the boundary rather than wrapped
    """
    edges = get_edge_indices(l, m)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)

    # Offsets relative to anchor (x0, y0)
    # Red (X-type)
    red_h_offsets = [(0, 0), (2, 1), (2, 2)]
    red_v_offsets = [(0, 2), (1, 2), (2, 0)]

    # Blue (Z-type)
    blue_h_offsets = [(0, 2), (1, 0), (2, 0)]
    blue_v_offsets = [(0, 0), (0, 1), (2, 2)]

    def get_stabilizer_support(anchor, h_offsets, v_offsets):
        x0, y0 = anchor
        support = []

        # Horizontal edges without wrapping
        for dx, dy in h_offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < l and 0 <= y < m:
                idx = edge_to_idx.get(((x, y), "h"))
                if idx is not None:
                    support.append(idx)

        # Vertical edges without wrapping
        for dx, dy in v_offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < l and 0 <= y < m:
                idx = edge_to_idx.get(((x, y), "v"))
                if idx is not None:
                    support.append(idx)

        return sorted(support)

    # Bulk anchors
    bulk_anchors = [(x, y) for x in range(l - B + 1) for y in range(m - B + 1)]

    # X-boundary anchors (top and bottom, outside the lattice)
    x_boundary_anchors = [
        (x, y) for x in range(l - B + 1) for y in [-2, -1, m - B + 1, m - B + 2]
    ]

    # Z-boundary anchors (left and right, outside the lattice)
    z_boundary_anchors = [
        (x, y) for x in [-2, -1, l - B + 1, l - B + 2] for y in range(m - B + 1)
    ]

    red_stabilizers = []
    blue_stabilizers = []

    # Add bulk stabilizers
    for anchor in bulk_anchors:
        red_stabilizers.append(
            get_stabilizer_support(anchor, red_h_offsets, red_v_offsets)
        )
        blue_stabilizers.append(
            get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets)
        )

    # Add X-boundary stabilizers
    for anchor in x_boundary_anchors:
        red_stabilizers.append(
            get_stabilizer_support(anchor, red_h_offsets, red_v_offsets)
        )

    # Add Z-boundary stabilizers
    for anchor in z_boundary_anchors:
        blue_stabilizers.append(
            get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets)
        )

    red_stabilizers, blue_stabilizers, num_qubits_final = remap_stabilizers(
        red_stabilizers, blue_stabilizers, num_edges
    )

    Hx, Hz, H_parity = build_parity_matrices(
        red_stabilizers, blue_stabilizers, num_qubits_final
    )

    return Hx, Hz, H_parity
