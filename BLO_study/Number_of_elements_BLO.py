"""
BLO study for one sample deformation.

This script constructs the periodic tile code on an l x m lattice, applies
a probabilistic Clifford deformation to the full CSS parity-check matrix,
and estimates the average number of independent pure-Z logical operators
(BLO-related quantity) over many random samples.

This file is intended as one sample BLO study at the deformation point

    p = 0.25
    q = 0.5

where
    p = Hadamard deformation probability
    q = YZ deformation probability

Usage
-----
python performance.py <l> <m>

Example
-------
python performance.py 12 12
"""

import os
import sys
import numpy as np
import pandas as pd

from ldpc.mod2 import nullspace, rank


# ============================================================
# Lattice dimensions from command line
# ============================================================
if len(sys.argv) < 3:
    print("Usage: python performance.py <l> <m>")
    sys.exit(1)

l = int(sys.argv[1])
m = int(sys.argv[2])

print(f"[INFO] Running for l = {l}, m = {m}")


# ============================================================
# Periodic tile-code construction
# ============================================================
def get_edge_indices(l, m):
    """
    Return the ordered list of horizontal and vertical edges
    on an l x m periodic lattice.
    """
    h_edges = [((x, y), "h") for y in range(m) for x in range(l)]
    v_edges = [((x, y), "v") for y in range(m) for x in range(l)]
    return h_edges + v_edges


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


def get_stabilizer_support(anchor, h_offsets, v_offsets, l, m):
    """
    Construct the support of a stabilizer anchored at `anchor`
    using periodic boundary conditions.
    """
    x0, y0 = anchor
    support = []

    # Horizontal edges
    for dx, dy in h_offsets:
        x, y = (x0 + dx) % l, (y0 + dy) % m
        idx = edge_to_idx.get(((x, y), "h"))
        if idx is not None:
            support.append(idx)

    # Vertical edges
    for dx, dy in v_offsets:
        x, y = (x0 + dx) % l, (y0 + dy) % m
        idx = edge_to_idx.get(((x, y), "v"))
        if idx is not None:
            support.append(idx)

    return sorted(support)


# Bulk anchors
bulk_anchors = [(x, y) for x in range(l) for y in range(m)]

red_stabilizers = []
blue_stabilizers = []

for anchor in bulk_anchors:
    red_stabilizers.append(
        get_stabilizer_support(anchor, red_h_offsets, red_v_offsets, l, m)
    )
    blue_stabilizers.append(
        get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets, l, m)
    )


# ============================================================
# Remove unsupported qubits
# ============================================================
qubit_touched = np.zeros(num_edges, dtype=bool)

for stab in red_stabilizers + blue_stabilizers:
    for qbit in stab:
        qubit_touched[qbit] = True

old_to_new = {}
new_idx = 0

for i, touched in enumerate(qubit_touched):
    if touched:
        old_to_new[i] = new_idx
        new_idx += 1

num_qubits_final = new_idx


def remap_stabilizer(stab):
    return [old_to_new[qbit] for qbit in stab if qbit in old_to_new]


red_stabilizers = [remap_stabilizer(stab) for stab in red_stabilizers]
blue_stabilizers = [remap_stabilizer(stab) for stab in blue_stabilizers]

# Remove empty stabilizers
red_stabilizers = [stab for stab in red_stabilizers if len(stab) > 0]
blue_stabilizers = [stab for stab in blue_stabilizers if len(stab) > 0]


def stabilizer_to_vector(stab, length):
    vec = np.zeros(length, dtype=int)
    for qbit in stab:
        vec[qbit] = 1
    return vec


Hx = np.array(
    [stabilizer_to_vector(stab, num_qubits_final) for stab in red_stabilizers],
    dtype=int,
)
Hz = np.array(
    [stabilizer_to_vector(stab, num_qubits_final) for stab in blue_stabilizers],
    dtype=int,
)

# Full CSS parity-check matrix
zeros_x = np.zeros((Hx.shape[0], Hz.shape[1]), dtype=int)
zeros_z = np.zeros((Hz.shape[0], Hx.shape[1]), dtype=int)
H_parity = np.block([[Hx, zeros_x], [zeros_z, Hz]])

print(f"[INFO] Number of physical qubits after pruning: N = {num_qubits_final}")
print(f"[INFO] Number of X stabilizers: {Hx.shape[0]}")
print(f"[INFO] Number of Z stabilizers: {Hz.shape[0]}")
print(f"[INFO] Full parity-check matrix shape: {H_parity.shape}")


# ============================================================
# Probabilistic Clifford deformation
# ============================================================
def apply_probabilistic_deformation(H, n, p, q, rng):
    """
    Apply probabilistic deformation on each qubit using the provided RNG.

    On each qubit:
      - with probability p: apply Hadamard deformation
      - with probability q: apply XY deformation
      - else: do nothing

    The action on columns is:
      Hadamard: (X, Z) -> (Z, X)
      XY:       (X, Z) -> (X + Z, Z)

    Parameters
    ----------
    H : np.ndarray
        Stabilizer matrix of shape (r, 2n).
    n : int
        Number of physical qubits.
    p : float
        Probability of Hadamard deformation.
    q : float
        Probability of XY deformation.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    H : np.ndarray
        Deformed stabilizer matrix.
    """
    for i in range(n):
        r = rng.random()

        x_col = H[:, i].copy()
        z_col = H[:, i + n].copy()

        if r < p:
            # Hadamard: (x, z) -> (z, x)
            H[:, i] = z_col
            H[:, i + n] = x_col

        elif r < p + q:
            # XY deformation: (x, z) -> (x + z, z)
            H[:, i] = (x_col + z_col) % 2
            H[:, i + n] = z_col

        # Else: do nothing

    return H


# One sample point for the BLO study
p = 0.25
q = 0.5


# ============================================================
# Seed logging
# ============================================================
filename = f"used_seeds_l_{l}_m_{m}.txt"
save_seeds = True

if save_seeds:
    with open(filename, "w") as f:
        pass


# ============================================================
# Monte Carlo sampling for BLO-related quantities
# ============================================================
num_samples = 10000

independent_logicals_counts = []
from_rank_counts = []

for trial_index in range(num_samples):
    seed = trial_index
    rng = np.random.default_rng(seed=seed)

    if save_seeds:
        with open(filename, "a") as f:
            f.write(f"{seed}\n")

    H_input = H_parity.copy()
    H_deformed = apply_probabilistic_deformation(
        H_input,
        H_input.shape[1] // 2,
        p,
        q,
        rng,
    )

    pure_z_nullspace = nullspace(H_deformed[:, :(H_deformed.shape[1] // 2)]).toarray() % 2

    independent_logicals = len(pure_z_nullspace)
    nullspace_rank = rank(pure_z_nullspace)

    independent_logicals_counts.append(independent_logicals)
    from_rank_counts.append(nullspace_rank)


# ============================================================
# Save summary results
# ============================================================
results = [
    {
        "system_size": (l, m),
        "average_inequivalent_pureZ_logicals": np.mean(independent_logicals_counts),
        "average_ranks": np.mean(from_rank_counts),
    }
]

df = pd.DataFrame(results)

output_csv = f"final_periodic_pureZ_logicals_summary_size_l_{l}_m_{m}_p_{p}_q_{q}.csv"
df.to_csv(output_csv, index=False)

print(f"[INFO] Saved results to {output_csv}")
