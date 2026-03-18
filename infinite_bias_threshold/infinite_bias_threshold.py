"""
Periodic lattice infinite-bias threshold study.

This script constructs a periodic tile code on an l x m lattice, applies
probabilistic single-qubit Clifford deformations, and estimates the logical
error rate under pure Z noise using BP+OSD decoding.

This file provides **one sample simulation code** using the probabilistic
Clifford deformation channel with parameters

    p = 0.25   (Hadamard deformation probability)
    q = 0.5    (YZ deformation probability)

Usage
-----
python performance.py <l> <m>

Example
-------
python performance.py 12 12
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from numpy.random import SeedSequence
from scipy.stats import bootstrap
from ldpc import BpOsdDecoder
from bposd.css import css_code


# ============================================================
# Lattice dimensions from command line
# ============================================================
if len(sys.argv) < 3:
    print("Usage: python performance.py <l> <m>")
    sys.exit(1)

l = int(sys.argv[1])
m = int(sys.argv[2])

print(f"[INFO] Lattice dimensions: l = {l}, m = {m}")


# ============================================================
# Helper functions for periodic tile-code construction
# ============================================================
def get_edge_indices(l: int, m: int):
    """
    Return the ordered list of horizontal and vertical edges
    on an l x m periodic lattice.

    Ordering:
      1. all horizontal edges
      2. all vertical edges
    """
    h_edges = [((x, y), "h") for y in range(m) for x in range(l)]
    v_edges = [((x, y), "v") for y in range(m) for x in range(l)]
    return h_edges + v_edges


def get_stabilizer_support(anchor, h_offsets, v_offsets, l, m, edge_to_idx):
    """
    Construct the support of a stabilizer anchored at `anchor`,
    using periodic boundary conditions.
    """
    x0, y0 = anchor
    support = []

    for dx, dy in h_offsets:
        x, y = (x0 + dx) % l, (y0 + dy) % m
        idx = edge_to_idx.get(((x, y), "h"))
        if idx is not None:
            support.append(idx)

    for dx, dy in v_offsets:
        x, y = (x0 + dx) % l, (y0 + dy) % m
        idx = edge_to_idx.get(((x, y), "v"))
        if idx is not None:
            support.append(idx)

    return sorted(support)


def stabilizer_to_vector(stab, length):
    vec = np.zeros(length, dtype=int)
    for q in stab:
        vec[q] = 1
    return vec


def remap_stabilizer(stab, old_to_new):
    return [old_to_new[q] for q in stab if q in old_to_new]


def apply_probabilistic_deformation(H, lx, lz, p, q, rng):
    """
    Apply probabilistic single-qubit Clifford deformation independently.

    Channel used in this example:

        Hadamard with probability p = 0.25
        XY deformation with probability q = 0.5
        Identity otherwise

    Transformations:
        Hadamard : (X,Z) -> (Z,X)
        XY       : (X,Z) -> (X+Z, Z)
    """

    n = lx.shape[1]
    L = np.hstack((lx, lz))

    for i in range(n):

        r = rng.random()

        x_col = H[:, i].copy()
        z_col = H[:, i + n].copy()

        lx_col = L[:, i].copy()
        lz_col = L[:, i + n].copy()

        if r < p:

            H[:, i] = z_col
            H[:, i + n] = x_col

            L[:, i] = lz_col
            L[:, i + n] = lx_col

        elif r < p + q:

            H[:, i] = (x_col + z_col) % 2
            H[:, i + n] = z_col

            L[:, i] = (lx_col + lz_col) % 2
            L[:, i + n] = lz_col

    lx_new = L[:, :n]
    lz_new = L[:, n:]

    return H, lx_new, lz_new


# ============================================================
# Build periodic tile code
# ============================================================
edges = get_edge_indices(l, m)
edge_to_idx = {edge: i for i, edge in enumerate(edges)}
num_edges = len(edges)

# Red (X-type)
red_h_offsets = [(0, 0), (2, 1), (2, 2)]
red_v_offsets = [(0, 2), (1, 2), (2, 0)]

# Blue (Z-type)
blue_h_offsets = [(0, 2), (1, 0), (2, 0)]
blue_v_offsets = [(0, 0), (0, 1), (2, 2)]

anchors = [(x, y) for x in range(l) for y in range(m)]

red_stabilizers = []
blue_stabilizers = []

for anchor in anchors:

    red_stabilizers.append(
        get_stabilizer_support(anchor, red_h_offsets, red_v_offsets, l, m, edge_to_idx)
    )

    blue_stabilizers.append(
        get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets, l, m, edge_to_idx)
    )


# ============================================================
# Remove unsupported qubits
# ============================================================
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

red_stabilizers = [remap_stabilizer(stab, old_to_new) for stab in red_stabilizers]
blue_stabilizers = [remap_stabilizer(stab, old_to_new) for stab in blue_stabilizers]

red_stabilizers = [stab for stab in red_stabilizers if len(stab) > 0]
blue_stabilizers = [stab for stab in blue_stabilizers if len(stab) > 0]

Hx = np.array(
    [stabilizer_to_vector(stab, num_qubits_final) for stab in red_stabilizers],
    dtype=int,
)

Hz = np.array(
    [stabilizer_to_vector(stab, num_qubits_final) for stab in blue_stabilizers],
    dtype=int,
)

print(f"[INFO] Number of physical qubits after pruning: N = {num_qubits_final}")
print(f"[INFO] Number of X stabilizers: {Hx.shape[0]}")
print(f"[INFO] Number of Z stabilizers: {Hz.shape[0]}")


# ============================================================
# Build CSS code
# ============================================================
qcode = css_code(Hx, Hz)
qcode.test()

lx = qcode.lx.toarray()
lz = qcode.lz.toarray()

H_in = qcode.h.toarray()

print(f"[INFO] CSS code constructed successfully.")
print(f"[INFO] qcode.N = {qcode.N}")
print(f"[INFO] Number of logical qubits = {lx.shape[0]}")


# ============================================================
# Deformation parameters (example values)
# ============================================================
p = 0.25
q = 0.5

physical_error_rates = np.linspace(0.01, 0.5, 20)


# ============================================================
# Seed handling
# ============================================================
seed_file = "master_seed_single.txt"

if not os.path.exists(seed_file):

    seed_val = int(SeedSequence(12345).generate_state(1)[0])

    with open(seed_file, "w") as f:
        f.write(f"{seed_val}\n")

else:

    with open(seed_file, "r") as f:
        seed_val = int(f.readline().strip())

print(f"[INFO] Using master seed = {seed_val}")


trials_per_error_rate = [
    int(20000 + max(0, (0.25 - e)) * 80000) if e <= 0.25 else 20000
    for e in physical_error_rates
]

master_seq = SeedSequence(seed_val)
child_seeds = master_seq.spawn(sum(trials_per_error_rate))


# ============================================================
# Infinite-bias decoding simulation
# ============================================================
logical_error_rates = []
bootstrap_standard_errors = []

trial_seed_index = 0

for error_rate_ins in physical_error_rates:

    num_trials = (
        int(20000 + max(0, (0.25 - error_rate_ins)) * 80000)
        if error_rate_ins <= 0.25
        else 20000
    )

    print(f"[INFO] Physical Z error rate = {error_rate_ins:.5f}, trials = {num_trials}")

    trial_outcomes = []

    for _ in range(num_trials):

        trial_seq = child_seeds[trial_seed_index]
        rng = np.random.default_rng(trial_seq)

        trial_seed_index += 1

        H_def, lx_def, lz_def = apply_probabilistic_deformation(
            H_in.copy(), lx.copy(), lz.copy(), p, q, rng
        )

        Hx_def = H_def[:, :qcode.N]

        z_error_pure = (rng.random(qcode.N) < error_rate_ins).astype(np.uint8)

        syndrome = (Hx_def @ z_error_pure) % 2

        decoder = BpOsdDecoder(
            Hx_def,
            error_rate=float(error_rate_ins),
            bp_method="product_sum",
            max_iter=100,
            schedule="serial",
            osd_method="osd_e",
            osd_order=8,
        )

        correction = decoder.decode(syndrome)

        residual = (correction + z_error_pure) % 2

        logical_fail = (lx_def @ residual % 2).any()

        trial_outcomes.append(int(logical_fail))

    mean_ler = np.mean(trial_outcomes)
    logical_error_rates.append(mean_ler)

    try:

        bs = bootstrap(
            (np.array(trial_outcomes),),
            np.mean,
            n_resamples=499,
            confidence_level=0.95,
            method="BCa",
        )

        bootstrap_standard_errors.append(bs.standard_error)

    except Exception:

        bootstrap_standard_errors.append(0.0)


# ============================================================
# Save results
# ============================================================
df = pd.DataFrame(
    {
        "Physical Z Error Rates": physical_error_rates,
        "Logical Error Rates": logical_error_rates,
        "Bootstrap Standard Error": bootstrap_standard_errors,
    }
)

csv_filename = (
    f"periodic_tile_code_infinite_bias_threshold_l{l}_m{m}_"
    f"N_{qcode.N}_p{p}_q{q}.csv"
)

df.to_csv(csv_filename, index=False)

print(f"[INFO] Results saved to {csv_filename}")
