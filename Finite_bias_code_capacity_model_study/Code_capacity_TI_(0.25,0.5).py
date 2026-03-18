# Finite-bias code-capacity model study for open tile codes.
# The implementation below is inspired by the GitHub codes associated with Ref.:
# https://quantum-journal.org/papers/q-2023-05-15-1005/

import os
import math
import json
import time
import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from ldpc import BpOsdDecoder
from bposd.css import css_code
import scipy.sparse as sparse


class css_decode_sim:
    """
    A class for simulating BP+OSD decoding of CSS codes.

    Note
    ----
    The input parameters can be entered directly or as a dictionary.

    Parameters
    ----------
    hx : numpy.ndarray
        The hx matrix of the CSS code.
    hz : numpy.ndarray
        The hz matrix of the CSS code.
    error_rate : float
        The physical error rate on each qubit.
    xyz_error_bias : list of ints
        The relative bias for X, Y and Z errors.
    seed : int
        The random number generator seed.
    target_runs : int
        The number of runs you wish to simulate.
    bp_method : str
        The BP method. Choose either: "minimum_sum" or "product_sum".
    ms_scaling_factor : float
        The minimum-sum scaling factor (if applicable).
    max_iter : int
        The maximum number of iterations for BP.
    osd_method : str
        The OSD method. Choose from: "osd_cs", "osd_e", or "osd0".
    channel_update : str
        The channel update method. Choose from: None, "x->z", or "z->x".
    output_file : str
        The output file to write to.
    save_interval : int
        The time interval (in seconds) between writing to the output file.
    check_code : bool
        Check whether the CSS code is valid.
    tqdm_disable : bool
        Enable/disable the tqdm progress bar. If running on an HPC cluster,
        it is recommended to disable tqdm.
    run_sim : bool
        If enabled (default), the simulation starts automatically.
    hadamard_rotate : bool
        Toggle Hadamard rotate. ON: 1; OFF: 0.
    hadamard_rotate_sector1_length : int
        Specifies the number of qubits in sector 1 for the Hadamard rotation.
    error_bar_precision_cutoff : float
        The simulation stops after this precision is reached.
    p : float
        Probability to apply Hadamard.
    q : float
        Probability to apply YZ deformation.
    apply_deformed_error : bool
        Whether to use a deformed error channel.
    """

    def __init__(self, hx=None, hz=None, lx=None, lz=None, **input_dict):
        default_input = {
            "error_rate": None,
            "xyz_error_bias": [1, 1, 1],
            "target_runs": 100,
            "seed": 0,
            "bp_method": "minimum_sum",
            "ms_scaling_factor": 0.625,
            "max_iter": 0,
            "osd_method": "osd_cs",
            "osd_order": 2,
            "save_interval": 2,
            "output_file": None,
            "check_code": 1,
            "tqdm_disable": 0,
            "run_sim": 1,
            "channel_update": "x->z",
            "hadamard_rotate": 0,
            "hadamard_rotate_sector1_length": 0,
            "error_bar_precision_cutoff": 1e-3,
            "p": 0.0,
            "q": 0.0,
            "apply_deformed_error": False,
            "apply_static_error_deformation": False,
            "static_deformation_map_data": None,
        }

        for key in input_dict.keys():
            self.__dict__[key] = input_dict[key]
        for key in default_input.keys():
            if key not in input_dict:
                self.__dict__[key] = default_input[key]

        output_values = {
            "K": None,
            "N": None,
            "start_date": None,
            "runtime": 0.0,
            "runtime_readable": None,
            "run_count": 0,
            "bp_converge_count_x": 0,
            "bp_converge_count_z": 0,
            "bp_success_count": 0,
            "bp_logical_error_rate": 0,
            "bp_logical_error_rate_eb": 0,
            "osd0_success_count": 0,
            "osd0_logical_error_rate": 0.0,
            "osd0_logical_error_rate_eb": 0.0,
            "osdw_success_count": 0,
            "osdw_logical_error_rate": 0.0,
            "osdw_logical_error_rate_eb": 0.0,
            "osdw_word_error_rate": 0.0,
            "osdw_word_error_rate_eb": 0.0,
            "min_logical_weight": 1e9,
        }

        for key in output_values.keys():
            if key not in self.__dict__:
                self.__dict__[key] = output_values[key]

        temp = []
        for key in self.__dict__.keys():
            if key not in [
                "channel_probs_x",
                "channel_probs_z",
                "channel_probs_y",
                "hx",
                "hz",
            ]:
                temp.append(key)
        self.output_keys = temp

        if not self.seed:
            self.seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0]
            )
        self.rng = np.random.default_rng(self.seed)
        print(f"RNG Seed: {self.seed}")

        self.hx = sparse.csr_matrix(hx).astype(np.uint8)
        self.hz = sparse.csr_matrix(hz).astype(np.uint8)
        self.N = self.hz.shape[1]

        if self.min_logical_weight == 1e9:
            self.min_logical_weight = self.N

        self.error_x = np.zeros(self.N).astype(np.uint8)
        self.error_z = np.zeros(self.N).astype(np.uint8)

        self.lx = lx
        self.lz = lz

        self._construct_code()
        self._error_channel_setup()
        self._decoder_setup()

        if self.run_sim:
            self.run_decode_sim()

    def _single_run(self):
        self.error_x, self.error_z = self._generate_error()

        s = (self.hx @ self.error_z + self.hz @ self.error_x) % 2

        self.bpd = BpOsdDecoder(
            self.H_bin,
            channel_probs=self.channel_probs_bin,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )

        d_bin = self.bpd.decode(s)
        if d_bin is None:
            raise RuntimeError("Decoder returned None; check osd_method/max_iter.")
        d_bin = np.asarray(d_bin, dtype=np.uint8).ravel()

        dec_x = d_bin[: self.N]
        dec_z = d_bin[self.N : 2 * self.N]
        self.bpd_dec_x = dec_x
        self.bpd_dec_z = dec_z

        self._encoded_error_rates_joint()

    def _channel_update(self, update_direction):
        """
        Update the channel probability vector for the second decoding component
        based on the first. The channel probability updates can be derived
        from Bayes' rule.
        """
        if update_direction == "x->z":
            decoder_probs = np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_x.osdw_decoding[i] == 1:
                    if (self.channel_probs_x[i] + self.channel_probs_y[i]) == 0:
                        decoder_probs[i] = 0
                    else:
                        decoder_probs[i] = self.channel_probs_y[i] / (
                            self.channel_probs_x[i] + self.channel_probs_y[i]
                        )
                elif self.bpd_x.osdw_decoding[i] == 0:
                    decoder_probs[i] = self.channel_probs_z[i] / (
                        1 - self.channel_probs_x[i] - self.channel_probs_y[i]
                    )

            self.bpd_z.update_channel_probs(decoder_probs)

        elif update_direction == "z->x":
            decoder_probs = np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_z.osdw_decoding[i] == 1:
                    if (self.channel_probs_z[i] + self.channel_probs_y[i]) == 0:
                        decoder_probs[i] = 0
                    else:
                        decoder_probs[i] = self.channel_probs_y[i] / (
                            self.channel_probs_z[i] + self.channel_probs_y[i]
                        )
                elif self.bpd_z.osdw_decoding[i] == 0:
                    decoder_probs[i] = self.channel_probs_x[i] / (
                        1 - self.channel_probs_z[i] - self.channel_probs_y[i]
                    )

            self.bpd_x.update_channel_probs(decoder_probs)

    def _apply_probabilistic_deformation(self, error, p, q):
        """
        Apply probabilistic deformation on each qubit of the error vector:
        - with probability p: apply Hadamard (x <-> z)
        - with probability q: apply XY deformation (x -> x+z)
        - else: do nothing
        """
        n = self.N

        for i in range(n):
            r = np.random.rand()

            x_err = error[i].copy()
            z_err = error[i + n].copy()

            if r < p:
                error[i] = z_err
                error[i + n] = x_err

            elif r < p + q:
                error[i] = (x_err + z_err) % 2
                error[i + n] = z_err

        return error

    def _encoded_error_rates_joint(self):
        residual_x = (self.error_x + self.bpd_dec_x) % 2
        residual_z = (self.error_z + self.bpd_dec_z) % 2

        if ((self.lz @ residual_x) % 2).any():
            lw = int(np.sum(residual_x))
            if lw < self.min_logical_weight:
                self.min_logical_weight = lw
        elif ((self.lx @ residual_z) % 2).any():
            lw = int(np.sum(residual_z))
            if lw < self.min_logical_weight:
                self.min_logical_weight = lw
        else:
            self.osdw_success_count += 1

        self.osdw_logical_error_rate = 1 - self.osdw_success_count / self.run_count
        self.osdw_logical_error_rate_eb = np.sqrt(
            (1 - self.osdw_logical_error_rate)
            * self.osdw_logical_error_rate
            / self.run_count
        )
        self.osdw_word_error_rate = 1.0 - (1 - self.osdw_logical_error_rate) ** (
            1 / self.K
        )
        self.osdw_word_error_rate_eb = (
            self.osdw_logical_error_rate_eb
            * ((1 - self.osdw_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

    def _construct_code(self):
        """
        Install (possibly deformed) hx, hz, lx, lz with no validation.
        """
        self.hx = sparse.csr_matrix(self.hx).astype(np.uint8)
        self.hz = sparse.csr_matrix(self.hz).astype(np.uint8)

        self.N = int(self.hz.shape[1])

        if self.lx is not None:
            self.lx = np.asarray(self.lx, dtype=np.uint8)
        if self.lz is not None:
            self.lz = np.asarray(self.lz, dtype=np.uint8)

        if self.lx is not None:
            self.K = int(self.lx.shape[0])
        elif getattr(self, "K", None) is None:
            self.K = 1

        return None

    def _error_channel_setup(self):
        """
        Set up the error channels from the error rate and error bias input parameters.
        """
        xyz_error_bias = np.array(self.xyz_error_bias, dtype=float)
        tot = np.sum(xyz_error_bias)
        if tot <= 0:
            raise ValueError("xyz_error_bias must have positive sum.")

        self.px, self.py, self.pz = self.error_rate * xyz_error_bias / tot

        if self.hadamard_rotate == 0:
            if self.apply_deformed_error:
                px_new = (1 - self.p) * self.px + self.p * self.pz
                py_new = (1 - self.q) * self.py + self.q * self.pz
                pz_new = (
                    (1 - self.p - self.q) * self.pz
                    + self.p * self.px
                    + self.q * self.py
                )
            else:
                px_new = self.px
                py_new = self.py
                pz_new = self.pz

            self.channel_probs_x = np.full(self.N, px_new, dtype=float)
            self.channel_probs_y = np.full(self.N, py_new, dtype=float)
            self.channel_probs_z = np.full(self.N, pz_new, dtype=float)

        elif self.hadamard_rotate == 1:
            n1 = self.N // 2
            self.channel_probs_x = np.hstack(
                [
                    np.full(n1, self.px, dtype=float),
                    np.full(self.N - n1, self.pz, dtype=float),
                ]
            )
            self.channel_probs_z = np.hstack(
                [
                    np.full(n1, self.pz, dtype=float),
                    np.full(self.N - n1, self.px, dtype=float),
                ]
            )
            self.channel_probs_y = np.full(self.N, self.py, dtype=float)

        elif self.hadamard_rotate == 2:
            self.channel_probs_x = np.full(self.N, self.px, dtype=float)
            self.channel_probs_z = np.full(self.N, self.py, dtype=float)
            self.channel_probs_y = np.full(self.N, self.pz, dtype=float)

        elif self.hadamard_rotate == 3:
            n1 = self.N // 2
            l = int(self.l)

            px, py, pz = float(self.px), float(self.py), float(self.pz)

            mask_first = np.zeros(n1, dtype=bool)

            for i in range(l):
                for j in range(l // 3):
                    idx = l * i + 3 * j
                    if 0 <= idx < n1:
                        mask_first[idx] = True

            for i in range(l // 2):
                for j in range(l // 3):
                    idx = l * (2 * i + 1) + (3 * j + 1)
                    if 0 <= idx < n1:
                        mask_first[idx] = True

            first_x = np.full(n1, px, dtype=float)
            first_y = np.full(n1, py, dtype=float)
            first_z = np.full(n1, pz, dtype=float)

            if mask_first.any():
                x_copy = first_x.copy()
                z_copy = first_z.copy()
                first_x[mask_first] = z_copy[mask_first]
                first_z[mask_first] = x_copy[mask_first]

            second_x = np.full(self.N - n1, px, dtype=float)
            second_y = np.full(self.N - n1, pz, dtype=float)
            second_z = np.full(self.N - n1, py, dtype=float)

            self.channel_probs_x = np.hstack([first_x, second_x])
            self.channel_probs_y = np.hstack([first_y, second_y])
            self.channel_probs_z = np.hstack([first_z, second_z])

        else:
            raise ValueError(
                f"The hadamard rotate attribute should be 0, 1, 2, or 3. Not '{self.hadamard_rotate}'"
            )

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)

    def _decoder_setup(self):
        """
        Joint Pauli (non-CSS) decoding on the 2n-variable binary system:
          H_bin = [Hz | Hx], e_bin = [eX | eZ]
        syndrome: s = Hx*eZ + Hz*eX = H_bin @ e_bin  (mod 2)
        """
        self.ms_scaling_factor = float(self.ms_scaling_factor)

        Hz_csr = sparse.csr_matrix(self.hz, dtype=np.uint8)
        Hx_csr = sparse.csr_matrix(self.hx, dtype=np.uint8)
        self.H_bin = sparse.hstack([Hz_csr, Hx_csr], format="csr", dtype=np.uint8)

        p_ex = self.channel_probs_x + self.channel_probs_y
        p_ez = self.channel_probs_z + self.channel_probs_y
        self.channel_probs_bin = np.concatenate([p_ex, p_ez]).astype(float)

    def _generate_error(self):
        """
        Generate a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        """
        self.error_x = np.zeros(self.N).astype(np.uint8)
        self.error_z = np.zeros(self.N).astype(np.uint8)

        for i in range(self.N):
            rand = self.rng.random()
            if rand < self.channel_probs_z[i]:
                self.error_z[i] = 1
                self.error_x[i] = 0
            elif self.channel_probs_z[i] <= rand < (
                self.channel_probs_z[i] + self.channel_probs_x[i]
            ):
                self.error_z[i] = 0
                self.error_x[i] = 1
            elif (self.channel_probs_z[i] + self.channel_probs_x[i]) <= rand < (
                self.channel_probs_x[i]
                + self.channel_probs_y[i]
                + self.channel_probs_z[i]
            ):
                self.error_z[i] = 1
                self.error_x[i] = 1
            else:
                self.error_z[i] = 0
                self.error_x[i] = 0

        return self.error_x, self.error_z

    def run_decode_sim(self):
        """
        Main simulation loop and output control.
        """
        self.start_date = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%A, %B %d, %Y %H:%M:%S"
        )

        pbar = tqdm(
            range(self.run_count + 1, self.target_runs + 1),
            disable=self.tqdm_disable,
            ncols=0,
        )

        start_time = time.time()
        save_time = start_time

        for self.run_count in pbar:
            self._single_run()

            pbar.set_description(
                f"d_max: {self.min_logical_weight}; "
                f"OSDW_WER: {self.osdw_word_error_rate*100:.3g}±{self.osdw_word_error_rate_eb*100:.2g}%; "
                f"OSDW: {self.osdw_logical_error_rate*100:.3g}±{self.osdw_logical_error_rate_eb*100:.2g}%; "
                f"OSD0: {self.osd0_logical_error_rate*100:.3g}±{self.osd0_logical_error_rate_eb*100:.2g}%;"
            )

            current_time = time.time()
            save_loop = current_time - save_time

            if int(save_loop) > self.save_interval or self.run_count == self.target_runs:
                save_time = time.time()
                self.runtime = save_loop + self.runtime

                self.runtime_readable = time.strftime(
                    "%H:%M:%S", time.gmtime(self.runtime)
                )

                if self.output_file is not None:
                    f = open(self.output_file, "w+")
                    print(self.output_dict(), file=f)
                    f.close()

                if (
                    self.osdw_logical_error_rate_eb > 0
                    and self.osdw_logical_error_rate_eb
                    / self.osdw_logical_error_rate
                    < self.error_bar_precision_cutoff
                ):
                    print("\nTarget error bar precision reached. Stopping simulation...")
                    break

        return json.dumps(self.output_dict(), sort_keys=True, indent=4)

    def output_dict(self):
        """
        Format the output.
        """
        output_dict = {}
        for key, value in self.__dict__.items():
            if key in self.output_keys:
                output_dict[key] = value
        return json.dumps(output_dict, sort_keys=True, indent=4)


def create_tile_code(l, m, B=3):
    """
    Create a tile code with given parameters l, m, and B.
    Returns Hx, Hz, lx, lz, and n for the CSS code.
    """
    def get_edge_indices(l, m):
        h_edges = [((x, y), "h") for y in range(m) for x in range(l)]
        v_edges = [((x, y), "v") for y in range(m) for x in range(l)]
        return h_edges + v_edges

    edges = get_edge_indices(l, m)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    num_edges = len(edges)

    red_h_offsets = [(0, 0), (2, 1), (2, 2)]
    red_v_offsets = [(0, 2), (1, 2), (2, 0)]

    blue_h_offsets = [(0, 2), (1, 0), (2, 0)]
    blue_v_offsets = [(0, 0), (0, 1), (2, 2)]

    def get_stabilizer_support(anchor, h_offsets, v_offsets, l, m):
        x0, y0 = anchor
        support = []

        for dx, dy in h_offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < l and 0 <= y < m:
                idx = edge_to_idx.get(((x, y), "h"))
                if idx is not None:
                    support.append(idx)

        for dx, dy in v_offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < l and 0 <= y < m:
                idx = edge_to_idx.get(((x, y), "v"))
                if idx is not None:
                    support.append(idx)

        return sorted(support)

    bulk_anchors = [(x, y) for x in range(l - B + 1) for y in range(m - B + 1)]
    x_boundary_anchors = [
        (x, y) for x in range(l - B + 1) for y in [-2, -1, m - B + 1, m - B + 2]
    ]
    z_boundary_anchors = [
        (x, y) for x in [-2, -1, l - B + 1, l - B + 2] for y in range(m - B + 1)
    ]

    red_stabilizers = []
    blue_stabilizers = []

    for anchor in bulk_anchors:
        red_stabilizers.append(
            get_stabilizer_support(anchor, red_h_offsets, red_v_offsets, l, m)
        )
        blue_stabilizers.append(
            get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets, l, m)
        )

    for anchor in x_boundary_anchors:
        red_stabilizers.append(
            get_stabilizer_support(anchor, red_h_offsets, red_v_offsets, l, m)
        )

    for anchor in z_boundary_anchors:
        blue_stabilizers.append(
            get_stabilizer_support(anchor, blue_h_offsets, blue_v_offsets, l, m)
        )

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

    qcode = css_code(Hx, Hz)
    qcode.test()

    lx = qcode.lx.toarray()
    lz = qcode.lz.toarray()
    H_in = qcode.h.toarray()
    n = qcode.N
    q = n // 2

    H_new = H_in.copy()
    L = np.hstack([lx, lz]).copy()

    for offset in range(q):
        c2 = q + offset
        c4 = 3 * q + offset
        c1 = offset
        c3 = 2 * q + offset

        tmpH2 = H_new[:, c2].copy()
        tmpH4 = H_new[:, c4].copy()

        H_new[:, c2] = (tmpH2 + tmpH4) % 2
        H_new[:, c4] = tmpH4

        tmpL2 = L[:, c2].copy()
        tmpL4 = L[:, c4].copy()
        L[:, c2] = (tmpL2 + tmpL4) % 2
        L[:, c4] = tmpL4

    for i in range(l):
        for j in range(l // 3):
            tempH = H_new[:, l * i + 3 * j].copy()
            H_new[:, l * i + 3 * j] = H_new[:, l * i + 3 * j + n]
            H_new[:, l * i + 3 * j + n] = tempH

            tempL = L[:, l * i + 3 * j].copy()
            L[:, l * i + 3 * j] = L[:, l * i + 3 * j + n]
            L[:, l * i + 3 * j + n] = tempL

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
    Hx_new = H_new[:, :n]
    Hz_new = H_new[:, n:]

    return Hx_new, Hz_new, lx_new, lz_new, n


# =========================
# Manual simulation parameters
# =========================
TARGET_RUNS = 1000000
BP_METHOD = "product_sum"
MS_SCALING = 0.625
OSD_METHOD = "osd_e"
OSD_ORDER = 8
MAX_ITER = 500
CHANNEL_UPDATE = None
HADAMARD_ROTATE = 0
HAD_SECTOR1_LEN = 0
APPLY_DEFORMED_ERROR = False
APPLY_STATIC_DEFORMATION = False
P_DEFORM = 0.0
Q_DEFORM = 0.0
ERROR_BAR_PRECISION_CUTOFF = 1e-15
OUTPUT_ROOT = "results_scaling"
SEED_BASE = 42
# =========================


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return float("nan")
    denom = 1.0 + (z * z) / n
    return z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4 * n * n)) / denom


def _run_chunk(Hx, Hz, Lx, Lz, sim_input_base: dict, chunk_runs: int, worker_seed: int):
    """Run a subset of shots and return (osdw_success, run_count)."""
    if chunk_runs <= 0:
        return (0, 0)

    sim_in = dict(sim_input_base)
    sim_in["target_runs"] = int(chunk_runs)
    sim_in["seed"] = int(worker_seed)
    sim_in["tqdm_disable"] = True

    sim = css_decode_sim(hx=Hx, hz=Hz, lx=Lx, lz=Lz, **sim_in)
    return int(sim.osdw_success_count), int(sim.run_count)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Tile code scaling (one (l, m, bias) combination)"
    )
    ap.add_argument("l", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--bias", type=float, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    l, m = int(args.l), int(args.m)
    eta = float(args.bias)
    target_runs = int(TARGET_RUNS)

    error_rates = np.linspace(0.01, 0.5, 20)
    results = []

    Hx_def, Hz_def, lx_def, lz_def, n = create_tile_code(l, m)

    for error_rate in error_rates:
        print(f"\n[RUN] l={l}, m={m}, eta={eta}, error_rate={error_rate:.4f}")

        sim_input_base = {
            "error_rate": error_rate,
            "target_runs": target_runs,
            "bp_method": BP_METHOD,
            "ms_scaling_factor": MS_SCALING,
            "osd_method": OSD_METHOD,
            "osd_order": OSD_ORDER,
            "xyz_error_bias": [1.0, 1.0, eta],
            "hadamard_rotate": HADAMARD_ROTATE,
            "hadamard_rotate_sector1_length": HAD_SECTOR1_LEN,
            "channel_update": CHANNEL_UPDATE,
            "max_iter": MAX_ITER,
            "tqdm_disable": True,
            "run_sim": True,
            "error_bar_precision_cutoff": ERROR_BAR_PRECISION_CUTOFF,
            "p": P_DEFORM,
            "q": Q_DEFORM,
            "apply_deformed_error": APPLY_DEFORMED_ERROR,
            "apply_static_error_deformation": APPLY_STATIC_DEFORMATION,
            "static_deformation_map_data": None,
            "l": l,
        }

        max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))
        max_workers = max(1, max_workers)

        parts = np.array_split(np.arange(target_runs), max_workers)
        runs_per_chunk = [len(p) for p in parts if len(p) > 0]

        base_seed = int(SEED_BASE + n + 13 * l + 31 * m + int(eta) + int(error_rate * 1000))
        ss = np.random.SeedSequence(base_seed)
        children = ss.spawn(len(runs_per_chunk))
        child_seeds = [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]

        total_success = 0
        total_runs_done = 0

        with ProcessPoolExecutor(max_workers=len(runs_per_chunk)) as ex:
            futs = [
                ex.submit(
                    _run_chunk,
                    Hx_def,
                    Hz_def,
                    lx_def,
                    lz_def,
                    sim_input_base,
                    shots,
                    wseed,
                )
                for shots, wseed in zip(runs_per_chunk, child_seeds)
            ]
            for fut in as_completed(futs):
                succ, cnt = fut.result()
                total_success += succ
                total_runs_done += cnt

        ler = 1.0 - (total_success / total_runs_done)
        ler_eb = wilson_halfwidth(ler, total_runs_done)

        results.append(
            {
                "Physical Z Error Rate": error_rate,
                "Logical Error Rate": ler,
                "Wilson Error Bar": ler_eb,
            }
        )

        print(f"  -> LER={ler:.6e} ± {ler_eb:.6e}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    csv_path = os.path.join(
        OUTPUT_ROOT,
        f"tile_open_middle_deformed_parity_final_non_css_independent_l{l}_m{m}_eta{int(eta)}.csv",
    )
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"[WROTE] {csv_path}")


if __name__ == "__main__":
    main()