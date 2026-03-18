"""
MWBLO detail study for one sample deformation point.

This script constructs a tile-code parity-check matrix, applies a Clifford
deformation channel, and studies detailed MWBLO-related quantities,
including:
    - number of nullspace pure-Z logicals
    - minimum nullspace-logical weight
    - overlap measures among nullspace logicals

This file is one sample example with random deformation probabilities

    p = 0.25
    q = 0.5

where
    p = Hadamard deformation probability
    q = Y->Z deformation probability

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

from itertools import product, combinations
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
# Tile-code construction
# ============================================================
def get_edge_indices(l, m):
    h_edges = [((x, y), "h") for y in range(m) for x in range(l)]
    v_edges = [((x, y), "v") for y in range(m) for x in range(l)]
    return h_edges + v_edges


edges = get_edge_indices(l, m)
edge_to_idx = {e: i for i, e in enumerate(edges)}
num_edges = len(edges)

# Offsets from the tile pattern (relative to anchor at (x0, y0))
# Red (X-type)
red_h_offsets = [(0, 0), (2, 1), (2, 2)]
red_v_offsets = [(0, 2), (1, 2), (2, 0)]

# Blue (Z-type)
blue_h_offsets = [(0, 2), (1, 0), (2, 0)]
blue_v_offsets = [(0, 0), (0, 1), (2, 2)]


def get_stabilizer_support(anchor, h_offsets, v_offsets, l, m):
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

# Remove unsupported qubits
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

# Remove empty stabilizers
red_stabilizers = [stab for stab in red_stabilizers if len(stab) > 0]
blue_stabilizers = [stab for stab in blue_stabilizers if len(stab) > 0]


def stabilizer_to_vector(stab, length):
    vec = np.zeros(length, dtype=int)
    for q in stab:
        vec[q] = 1
    return vec


Hx = np.array(
    [stabilizer_to_vector(stab, num_qubits_final) for stab in red_stabilizers],
    dtype=int,
)
Hz = np.array(
    [stabilizer_to_vector(stab, num_qubits_final) for stab in blue_stabilizers],
    dtype=int,
)

# Create the zero matrices of appropriate size
zeros_x = np.zeros((Hx.shape[0], Hz.shape[1]), dtype=int)
zeros_z = np.zeros((Hz.shape[0], Hx.shape[1]), dtype=int)

# Construct the full parity-check matrix
H_parity = np.block([[Hx, zeros_x], [zeros_z, Hz]])


# ============================================================
# Deformation routines
# ============================================================
def hadamard_on_quarters(H, n):
    """
    Swap the 2nd quarter of columns with the 4th quarter of columns
    in the stabilizer matrix H.

    H is of shape (2r, 2n) with 2n divisible by 4.
    """
    H_new = H.copy()

    rows, total_cols = H_new.shape
    assert total_cols % 4 == 0, "2n must be divisible by 4"
    q = total_cols // 4

    for offset in range(q):
        c2 = q + offset
        c4 = 3 * q + offset

        tmp = H_new[:, c2].copy()
        H_new[:, c2] = H_new[:, c4]
        H_new[:, c4] = tmp

    return H_new


def Deformation_XY_Translational_invariant(H, n):
    """
    Translationally invariant XY deformation on H.

    H is of shape (2r, 2n) with 2n divisible by 4.
    """
    H_new = H.copy()

    rows, total_cols = H_new.shape
    assert total_cols % 4 == 0, "2n must be divisible by 4"
    n = total_cols // 2

    for offset in range(n):
        c1 = offset
        c2 = n + offset

        tmpHx = H_new[:, c1].copy()
        tmpHz = H_new[:, c2].copy()

        H_new[:, c1] = (tmpHx + tmpHz) % 2
        H_new[:, c2] = tmpHz

    return H_new


def Deformation_on_Translational_invariant(H):
    """
    Translationally invariant deformation on H.
    """
    H_new = H.copy()

    rows, total_cols = H_new.shape
    assert total_cols % 4 == 0, "2n must be divisible by 4"
    q = total_cols // 4
    n = total_cols // 2

    for offset in range(q):
        c2 = q + offset
        c4 = 3 * q + offset
        c1 = offset
        c3 = 2 * q + offset

        tmpH2 = H_new[:, c2].copy()
        tmpH4 = H_new[:, c4].copy()

        H_new[:, c2] = (tmpH2 + tmpH4) % 2
        H_new[:, c4] = tmpH4

    for i in range(l):
        for j in range(l // 3):
            tempH = H_new[:, l * i + 3 * j].copy()
            H_new[:, l * i + 3 * j] = H_new[:, l * i + 3 * j + n]
            H_new[:, l * i + 3 * j + n] = tempH

    for i in range(l // 2):
        for j in range(l // 3):
            tmpH = H_new[:, l * (2 * i + 1) + 3 * j + 1].copy()
            H_new[:, l * (2 * i + 1) + 3 * j + 1] = H_new[:, l * (2 * i + 1) + 3 * j + 1 + n]
            H_new[:, l * (2 * i + 1) + 3 * j + 1 + n] = tmpH

    return H_new


def apply_probabilistic_deformation(H, n, p, q, rng):
    """
    Apply probabilistic deformation on each qubit using provided rng:
    - with probability p: apply Hadamard
    - with probability q: apply XY deformation
    - else: do nothing
    """
    for i in range(n):
        r = rng.random()

        x_col = H[:, i].copy()
        z_col = H[:, i + n].copy()

        if r < p:
            H[:, i] = z_col
            H[:, i + n] = x_col

        elif r < p + q:
            H[:, i] = (x_col + z_col) % 2
            H[:, i + n] = z_col

    return H


def apply_random_hadamard(H, num_qubits, prob):
    """
    Apply random Hadamard gates to the stabilizer matrix in binary symplectic form.
    """
    n = num_qubits
    hadamard_applied = np.random.choice([True, False], size=n, p=[prob, 1 - prob])

    for i in range(n):
        if hadamard_applied[i]:
            temp = H[:, i].copy()
            H[:, i] = H[:, i + n]
            H[:, i + n] = temp

    return H


# One sample point
p = 0.25
q = 0.5


# ============================================================
# Linear algebra helpers
# ============================================================
def is_in_row_space(vector, basis):
    augmented_matrix = np.vstack([basis, vector])
    rank_before = rank(basis)
    rank_after = rank(augmented_matrix)
    return rank_before == rank_after


def compute_pairwise_overlap(pureZ_logicals):
    """
    Overlap between first two logicals.
    Returns np.nan if fewer than two logicals.
    """
    pureZ_logicals = np.array(pureZ_logicals) % 2

    if pureZ_logicals.shape[0] < 2:
        return np.nan

    L1 = pureZ_logicals[0]
    L2 = pureZ_logicals[1]
    return int(np.sum(L1 & L2))


def compute_overlap(pureZ_logicals):
    """
    Total number of qubits acted on by more than one logical.
    """
    support_per_qubit = np.sum(pureZ_logicals, axis=0)
    support_per_qubit = np.array(support_per_qubit)
    overlap_indices = np.where(support_per_qubit > 1)[0]
    overlap_count = len(overlap_indices)
    return overlap_count


def compute_overlap_one_logical(pureZ_logicals):
    """
    How many qubits the first logical overlaps with any other logical.
    Returns np.nan if there are no logicals.
    """
    pureZ_logicals = np.array(pureZ_logicals) % 2

    if pureZ_logicals.shape[0] == 0:
        return np.nan

    target = pureZ_logicals[0]
    others = pureZ_logicals[1:]

    if others.size == 0:
        return 0

    support_by_others = np.any(others, axis=0)
    shared_support = target & support_by_others
    return int(np.sum(shared_support))


def min_weight_logicals(pure_z_logicals, H):
    pure_z_logicals = np.array(pure_z_logicals) % 2
    weights = np.sum(pure_z_logicals, axis=1)
    sorted_indices = np.argsort(weights)
    pure_z_logicals = pure_z_logicals[sorted_indices]

    basis_matrix = np.zeros((0, pure_z_logicals.shape[1]), dtype=int)
    basis_list = []

    for v in pure_z_logicals:
        trial_matrix = np.vstack([basis_matrix, v])
        if rank(trial_matrix) > rank(basis_matrix):
            basis_list.append(v)
            basis_matrix = trial_matrix
        if rank(basis_matrix) == rank(pure_z_logicals):
            break

    basis_list = np.array(basis_list) % 2
    final_list = [v for v in basis_list if not is_in_row_space(v, H)]

    return np.array(final_list) % 2


def generate_all_pureZ_combinations(pureZ_stabiliser):
    N = pureZ_stabiliser.shape[0]
    if N == 0:
        return np.empty((0, pureZ_stabiliser.shape[1]), dtype=int)

    all_coeffs = np.array(list(product([0, 1], repeat=N)), dtype=int)
    all_combinations = (all_coeffs @ pureZ_stabiliser) % 2
    return all_combinations


def min_weight_logicals_Z_nearby(pure_z_logicals, stabs, H):
    """
    Minimize logical weights using nearby pure-Z stabilizers.
    """
    pure_z_logicals = (np.array(pure_z_logicals, dtype=np.uint8) & 1)
    stabs = (np.array(stabs, dtype=np.uint8) & 1)
    H = (np.array(H, dtype=np.uint8) & 1)

    logicals_list = [v for v in pure_z_logicals if not is_in_row_space(v, H)]
    if len(logicals_list) == 0:
        ncols = pure_z_logicals.shape[1] if pure_z_logicals.ndim == 2 else 0
        return np.zeros((0, ncols), dtype=np.uint8)

    logicals = (np.array(logicals_list, dtype=np.uint8) & 1)

    weights = np.sum(logicals, axis=1)
    sorted_indices = np.argsort(weights)
    logicals = logicals[sorted_indices]

    out = []
    MAX_NEAR_STABS = 50

    for L in logicals:
        L = (L.astype(np.uint8) & 1)
        support = L.astype(bool)

        near_stabs = [S for S in stabs if (S.astype(bool) & support).any()]
        num = len(near_stabs)

        if num == 0:
            out.append([L.copy()])
            continue

        if num > MAX_NEAR_STABS:
            near_stabs = near_stabs[:MAX_NEAR_STABS]
            num = MAX_NEAR_STABS

        best_w = int(L.sum())
        best_set = [L.copy()]

        for combo in product([0, 1], repeat=num):
            if not any(combo):
                continue

            Scombo = np.zeros_like(L, dtype=np.uint8)
            for coeff, S in zip(combo, near_stabs):
                if coeff:
                    Scombo ^= S

            cand = (L ^ Scombo)
            w = int(cand.sum())

            if w < best_w:
                best_w = w
                best_set = [cand]
            elif w == best_w and not any(np.array_equal(cand, x) for x in best_set):
                best_set.append(cand)

        out.append(best_set)

    flat = np.vstack([family[0] for family in out]).astype(np.uint8) & 1
    weight = np.sum(flat, axis=1)
    sorted_ind = np.argsort(weight)
    flat = flat[sorted_ind]

    return flat


def extract_pureZ_stabilisers(pureZ_logicals, H_def):
    """
    Extract pure-Z stabilizers from pure-Z logical candidates in reduced length-n form.
    """
    pureZ_logicals = np.array(pureZ_logicals, dtype=int) % 2
    H_x = np.array(H_def, dtype=int) % 2

    pureZ_stabiliser = []
    for v in pureZ_logicals:
        v = v % 2
        if is_in_row_space(v, H_x) and np.any(v):
            pureZ_stabiliser.append(v)

    return np.array(pureZ_stabiliser, dtype=int) % 2


def extract_pureZ_logicals(H_def, n):
    Omega = np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])

    H_x_nullspace = nullspace(H_def[:, :n]).toarray() % 2
    pureZ_logicals = np.array(H_x_nullspace)

    logical = [v for v in pureZ_logicals if not is_in_row_space(v, H_def[:, n:])]
    logical = np.array(logical) % 2

    pureZ_stabiliser = extract_pureZ_stabilisers(pureZ_logicals, H_def[:, n:])
    pureZ_stabiliser = np.array(pureZ_stabiliser) % 2

    nullspace_logicals = min_weight_logicals_Z_nearby(logical, pureZ_stabiliser, H_def[:, n:])
    nullspace_logicals = np.array(nullspace_logicals)

    min_weights_nullspace = []
    for logical_op in nullspace_logicals:
        weight = np.sum(logical_op)
        min_weights_nullspace.append(weight)

    if len(min_weights_nullspace) == 0:
        min_weight_nullspace = None
    else:
        min_weight_nullspace = np.min(min_weights_nullspace)

    return (
        min_weight_nullspace,
        np.array(nullspace_logicals),
        pureZ_stabiliser,
        pureZ_logicals,
    )


# ============================================================
# Seed logging
# ============================================================
filename = f"used_seeds_l_{l}_m_{m}.txt"
save_seeds = True

if save_seeds:
    with open(filename, "w") as f:
        pass


# ============================================================
# Monte Carlo sampling for MWBLO details
# ============================================================
num_samples = 10000

nullspace_logicals_counts = []
nullspace_min_weights_per_sample = []
num_nullspace_min_weight_logicals = []
overlap_count = []
pairwise_overlap_count = []
overlap_per_logical_count = []
stabiliser_count = []
before = []

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

    (
        min_weight_nullspace,
        nullspace_logicals,
        pureZ_stabiliser,
        pureZ_logicals,
    ) = extract_pureZ_logicals(H_deformed, H_deformed.shape[1] // 2)

    overlap = compute_overlap(nullspace_logicals)
    pairwise_overlap = compute_pairwise_overlap(nullspace_logicals)
    overlap_per_logical = compute_overlap_one_logical(nullspace_logicals)

    nullspace_logicals_counts.append(len(nullspace_logicals))
    nullspace_min_weights_per_sample.append(min_weight_nullspace)

    overlap_count.append(overlap)
    pairwise_overlap_count.append(pairwise_overlap)
    overlap_per_logical_count.append(overlap_per_logical)
    stabiliser_count.append(len(pureZ_stabiliser))


# ============================================================
# Save summary results
# ============================================================
results = [{
    "system_size": (l, m),
    "average_nullspace_pureZ_logicals": np.mean(nullspace_logicals_counts),
    "average_min_weight_nullspace_logical": np.mean(nullspace_min_weights_per_sample),
    "average_overlap_per_logical": np.mean(overlap_per_logical_count),
    "average_total_overlap": np.mean(overlap_count),
    "average_pairwise_overlap": np.mean(pairwise_overlap_count),
}]

df = pd.DataFrame(results)

output_csv = f"latest_pureZ_logicals_summary_size_l_{l}_m_{m}_p_{p}_q_{q}.csv"
df.to_csv(output_csv, index=False)

print(f"Saved results to {output_csv}")

# Note:
# The total number of elements in the BLO is always rank(nullspace(Hx)).
# However, the present construction can return a count smaller than
# rank(nullspace(Hx)) when pure-Z stabilizers are present inside the nullspace.
# In this code, our goal is not to determine the total BLO count. Rather, we
# focus on minimizing the weights of the logical operators obtained from the
# nullspace, and then, in particular, we compute the  weight and the
# total overlap of the resulting minimum-weight logicals.