"""
Clifford deformation utilities.

This file collects several Clifford-deformation routines used in the tile-code
simulations. It includes:

1. Probabilistic single-qubit deformation
2. Quarter-swap linear deformation
3. Translationally invariant deformation corresponding to TI(0.25, 0.5)
4. Translationally invariant XY deformation

These functions act on the full symplectic stabilizer matrix H and, where
appropriate, on the logical operator matrices lx and lz.
"""

import numpy as np


# ============================================================
# Deformation parameters used in the probabilistic channel
# ============================================================
p = 0.25  # Hadamard deformation probability
q = 0.5   # XY (Z -> Y) deformation probability


# ============================================================
# 1. Probabilistic Clifford deformation
# ============================================================
def apply_probabilistic_deformation(H, lx, lz, p, q, rng):
    """
    Apply probabilistic deformation on each qubit using the provided RNG.

    On each qubit independently:
      - with probability p: apply Hadamard deformation
      - with probability q: apply XY deformation
      - else: do nothing

    Parameters
    ----------
    H : np.ndarray
        Stabilizer matrix of shape (r, 2n).
    lx : np.ndarray
        Logical X operators of shape (k, n).
    lz : np.ndarray
        Logical Z operators of shape (k, n).
    p : float
        Probability to apply Hadamard deformation.
    q : float
        Probability to apply XY deformation.
    rng : np.random.Generator
        Random number generator for the deformation.

    Returns
    -------
    H_new, lx_new, lz_new : np.ndarray
        Updated stabilizer and logical matrices after deformation.
    """
    n = lx.shape[1]
    L = np.hstack((lx, lz))  # Combined logical operators

    for i in range(n):
        r = rng.random()

        x_col = H[:, i].copy()
        z_col = H[:, i + n].copy()
        lx_col = L[:, i].copy()
        lz_col = L[:, i + n].copy()

        if r < p:
            # Apply Hadamard: (x, z) -> (z, x)
            H[:, i] = z_col
            H[:, i + n] = x_col
            L[:, i] = lz_col
            L[:, i + n] = lx_col

        elif r < p + q:
            # Apply XY deformation: (x, z) -> (x + z, z)
            H[:, i] = (x_col + z_col) % 2
            H[:, i + n] = z_col
            L[:, i] = (lx_col + lz_col) % 2
            L[:, i + n] = lz_col

        # Else: do nothing

    lx_new = L[:, :n]
    lz_new = L[:, n:]
    return H, lx_new, lz_new


# Example usage:
# H_in = qcode.h.toarray()
# lx_in = lx.toarray()
# lz_in = lz.toarray()
# H_def, lx_def, lz_def = apply_probabilistic_deformation(H_in, lx_in, lz_in, p, q, rng)


# ============================================================
# 2. Quarter-swap linear deformation
# ============================================================
def hadamard_on_quarters_with_logicals(H, lx, lz):
    """
    Swap the 2nd quarter of columns with the 4th quarter of columns
    in both the stabilizer matrix H and the logical operators (lx, lz).

    Assumes:
      - H has shape (r, 2n) with columns ordered as [X | Z]
      - 2n is divisible by 4
      - lx and lz each have shape (k, n)

    The same column permutation is applied to the concatenated logical matrix
    L = [lx | lz].

    Returns
    -------
    H_new, lx_new, lz_new : np.ndarray
        Deformed stabilizer and logical matrices.
    """
    H_new = H.copy()
    L = np.hstack([lx, lz]).copy()

    _, total_cols = H_new.shape
    if total_cols % 4 != 0:
        raise ValueError("Expected total columns = 2n to be divisible by 4.")

    q = total_cols // 4

    # Quarters:
    # [0 .. q-1], [q .. 2q-1], [2q .. 3q-1], [3q .. 4q-1]
    # swap 2nd quarter <-> 4th quarter
    for offset in range(q):
        c2 = q + offset
        c4 = 3 * q + offset

        # Swap in H
        tmp = H_new[:, c2].copy()
        H_new[:, c2] = H_new[:, c4]
        H_new[:, c4] = tmp

        # Swap in L
        tmpL = L[:, c2].copy()
        L[:, c2] = L[:, c4]
        L[:, c4] = tmpL

    n = total_cols // 2
    lx_new = L[:, :n]
    lz_new = L[:, n:]
    return H_new, lx_new, lz_new


# Example usage:
# H_in, lx_in, lz_in = qcode.h.toarray(), lx, lz
# H_XZ_output, lx_XZ_output, lz_XZ_output = hadamard_on_quarters_with_logicals(H_in, lx_in, lz_in)


# ============================================================
# 3. Translationally invariant deformation: TI(0.25, 0.5)
# ============================================================
def Deformation_on_Translational_invariant(H, lx, lz, l):
    """
    Translationally invariant deformation corresponding to TI(0.25, 0.5).

    First, modify the 2nd quarter of columns using the 4th quarter:
        c2 -> c2 + c4
        c4 -> c4

    This is applied in both the stabilizer matrix H and the logical matrix L.

    Then apply additional X/Z swaps on selected qubits determined by the
    lattice size l.

    Parameters
    ----------
    H : np.ndarray
        Stabilizer matrix of shape (2r, 2n).
    lx : np.ndarray
        Logical X operators of shape (k, n).
    lz : np.ndarray
        Logical Z operators of shape (k, n).
    l : int
        Lattice linear size used in the translationally invariant pattern.

    Returns
    -------
    H_new, lx_new, lz_new : np.ndarray
        Deformed stabilizer and logical matrices.
    """
    H_new = H.copy()
    L = np.hstack([lx, lz]).copy()

    rows, total_cols = H_new.shape
    assert total_cols % 4 == 0, "2n must be divisible by 4"

    q = total_cols // 4
    n = total_cols // 2

    # First deformation step on quartered columns
    for offset in range(q):
        c2 = q + offset
        c4 = 3 * q + offset
        c1 = offset
        c3 = 2 * q + offset

        # Stabilizer columns
        tmpH2 = H_new[:, c2].copy()
        tmpH4 = H_new[:, c4].copy()

        H_new[:, c2] = (tmpH2 + tmpH4) % 2
        H_new[:, c4] = tmpH4

        # Logical columns
        tmpL2 = L[:, c2].copy()
        tmpL4 = L[:, c4].copy()

        L[:, c2] = (tmpL2 + tmpL4) % 2
        L[:, c4] = tmpL4

    # Second deformation step
    for i in range(l):
        for j in range(l // 3):
            tempH = H_new[:, l * i + 3 * j].copy()
            H_new[:, l * i + 3 * j] = H_new[:, l * i + 3 * j + n]
            H_new[:, l * i + 3 * j + n] = tempH

            tempL = L[:, l * i + 3 * j].copy()
            L[:, l * i + 3 * j] = L[:, l * i + 3 * j + n]
            L[:, l * i + 3 * j + n] = tempL

    # Third deformation step
    for i in range(l // 2):
        for j in range(l // 3):
            tmpH = H_new[:, l * (2 * i + 1) + 3 * j + 1].copy()
            H_new[:, l * (2 * i + 1) + 3 * j + 1] = H_new[:, l * (2 * i + 1) + 3 * j + 1 + n]
            H_new[:, l * (2 * i + 1) + 3 * j + 1 + n] = tmpH

            tmpL = L[:, l * (2 * i + 1) + 3 * j + 1].copy()
            L[:, l * (2 * i + 1) + 3 * j + 1] = L[:, l * (2 * i + 1) + 3 * j + 1 + n]
            L[:, l * (2 * i + 1) + 3 * j + 1 + n] = tmpL

    lx_new = L[:, :n]
    lz_new = L[:, n:]

    return H_new, lx_new, lz_new


# Example usage:
# H_in, lx_in, lz_in = qcode.h.toarray(), lx, lz
# H_XZ_output, lx_XZ_output, lz_XZ_output = Deformation_on_Translational_invariant(H_in, lx_in, lz_in, l)


# ============================================================
# 4. Translationally invariant XY deformation
# ============================================================
def Deformation_XY_Translational_invariant(H, lx, lz):
    """
    Translationally invariant XY deformation.

    Applies the map:
        X -> X + Z
        Z -> Z

    on both the stabilizer matrix H and the logical operators (lx, lz).

    Parameters
    ----------
    H : np.ndarray
        Stabilizer matrix of shape (r, 2n).
    lx : np.ndarray
        Logical X operators of shape (k, n).
    lz : np.ndarray
        Logical Z operators of shape (k, n).

    Returns
    -------
    H_new : np.ndarray
        Deformed stabilizer matrix
    lx_new : np.ndarray
        Updated logical X operators
    lz_new : np.ndarray
        Updated logical Z operators
    """

    H_new = H.copy()

    # Combine logical operators
    L = np.hstack([lx, lz]).copy()

    rows, total_cols = H_new.shape
    assert total_cols % 2 == 0, "Expected H to have 2n columns"

    n = total_cols // 2

    for offset in range(n):

        c1 = offset        # X column
        c2 = n + offset    # Z column

        # Stabilizer update
        tmpHx = H_new[:, c1].copy()
        tmpHz = H_new[:, c2].copy()

        H_new[:, c1] = (tmpHx + tmpHz) % 2
        H_new[:, c2] = tmpHz

        # Logical update
        tmpLx = L[:, c1].copy()
        tmpLz = L[:, c2].copy()

        L[:, c1] = (tmpLx + tmpLz) % 2
        L[:, c2] = tmpLz

    lx_new = L[:, :n]
    lz_new = L[:, n:]

    return H_new, lx_new, lz_new

# Example usage:
# H_xy_output = Deformation_XY_Translational_invariant(H_in, qcode.N)
