from __future__ import annotations

import argparse
import csv
import math
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

import matplotlib.pyplot as plt


def _parse_progress_file(
    progress_file: Path,
) -> Tuple[List[int], List[Optional[int]], List[float], List[float], List[float], List[float]]:
    """Parse iteration, optional generated samples, X, Y, Z, and bias columns."""
    iterations: List[int] = []
    generated_samples: List[Optional[int]] = []
    x_probs: List[float] = []
    y_probs: List[float] = []
    z_probs: List[float] = []
    bias_values: List[float] = []

    with progress_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            row = next(csv.reader([line]))
            if len(row) < 6:
                continue

            try:
                iteration = int(row[0])
                x_value = float(row[2])
                y_value = float(row[3])
                z_value = float(row[4])
                bias_value = float(row[5])
                sample_value: Optional[int] = None
                if len(row) >= 13 and row[12].strip() != "":
                    sample_value = int(row[12])
            except ValueError:
                continue

            iterations.append(iteration)
            generated_samples.append(sample_value)
            x_probs.append(x_value)
            y_probs.append(y_value)
            z_probs.append(z_value)
            bias_values.append(bias_value)

    if not iterations:
        raise ValueError(f"No data rows were parsed from: {progress_file}")

    return iterations, generated_samples, x_probs, y_probs, z_probs, bias_values


def _build_output_path_with_flags(
    progress_file: Path,
    iter_start: int | None,
    iter_end: int | None,
    smooth_window: int,
    bias_log_scale: bool,
    x_axis: str,
    manual_sigfig: int | None = None,
    zk_tail_start: int | None = None,
    zk_rescale_tail: bool = False,
) -> Path:
    stem = progress_file.stem
    parts = ["plots"]
    if iter_start is not None or iter_end is not None:
        start_label = "min" if iter_start is None else str(iter_start)
        end_label = "max" if iter_end is None else str(iter_end)
        parts.append(f"iter_{start_label}-{end_label}")
    if smooth_window > 1:
        parts.append(f"smooth_{smooth_window}")
    if bias_log_scale:
        parts.append("biaslog")
    if x_axis == "samples":
        parts.append("x_samples")
    if manual_sigfig is not None:
        parts.append(f"sigfig{manual_sigfig}")
    if zk_tail_start is not None:
        parts.append(f"zktail_{zk_tail_start}")
    if zk_rescale_tail:
        parts.append("zkrescaled")
    suffix = "_".join(parts)
    return progress_file.with_name(f"{stem}_{suffix}.svg")


def _moving_average(values: List[float], window: int) -> List[float]:
    """Centered moving average that ignores non-finite values in each window."""
    if window <= 1:
        return list(values)

    n = len(values)
    half = window // 2
    smoothed: List[float] = []
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_values = [v for v in values[start:end] if math.isfinite(v)]
        if window_values:
            smoothed.append(sum(window_values) / len(window_values))
        else:
            smoothed.append(math.nan)
    return smoothed


def _sanitize_for_log(values: List[float]) -> List[float]:
    """Replace non-positive/non-finite values with NaN for log-scale plotting."""
    return [v if (math.isfinite(v) and v > 0.0) else math.nan for v in values]


def _parse_counts_totals(counts_file: Path) -> List[int]:
    """Parse cumulative total observation counts from counts JSONL file."""
    totals: List[int] = []
    with counts_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "counts" in data and isinstance(data["counts"], dict):
                counts_obj = data["counts"]
            else:
                counts_obj = data
            if not all(str(pauli_type) in counts_obj or pauli_type in counts_obj for pauli_type in (0, 1, 2, 3)):
                continue
            try:
                total = 0
                for pauli_type in (0, 1, 2, 3):
                    total += int(counts_obj.get(str(pauli_type), counts_obj.get(pauli_type, 0)))
            except (TypeError, ValueError):
                continue
            totals.append(total)
    if not totals:
        raise ValueError(f"No valid count rows parsed from: {counts_file}")
    return totals


def _align_counts_to_progress_length(rows: List[Any], progress_len: int, label: str) -> List[Any]:
    """Align counts-derived rows to progress rows by tail-trimming extras."""
    if len(rows) < progress_len:
        raise ValueError(
            f"{label} rows do not align with parsed progress rows. "
            f"{label}_rows={len(rows)}, progress_rows={progress_len}"
        )
    if len(rows) > progress_len:
        return rows[-progress_len:]
    return rows


def _round_to_n_sigfigs(value: float, n_sigfigs: int) -> float:
    """Round a value to n significant figures."""
    if not math.isfinite(value) or value == 0.0:
        return value
    if n_sigfigs < 1:
        return value
    decimals = n_sigfigs - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, decimals)


def _quantize_series_from_sigfigs(values: List[float], n_sigfigs: int) -> List[float]:
    """Quantize all values to n significant figures."""
    return [_round_to_n_sigfigs(v, n_sigfigs) for v in values]


def _compute_probability_cis(
    x_probs: List[float],
    y_probs: List[float],
    z_probs: List[float],
    bias_values: List[float],
    totals: List[int],
    z_score: float,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Compute normal-approximation CI bounds for X/Y/Z probabilities and bias."""
    x_low: List[float] = []
    x_high: List[float] = []
    y_low: List[float] = []
    y_high: List[float] = []
    z_low: List[float] = []
    z_high: List[float] = []
    b_low: List[float] = []
    b_high: List[float] = []

    for x_prob, y_prob, z_prob, bias_value, total in zip(x_probs, y_probs, z_probs, bias_values, totals):
        n = float(total)
        if n <= 0.0:
            x_low.append(math.nan)
            x_high.append(math.nan)
            y_low.append(math.nan)
            y_high.append(math.nan)
            z_low.append(math.nan)
            z_high.append(math.nan)
            b_low.append(math.nan)
            b_high.append(math.nan)
            continue

        se_x = math.sqrt(max(0.0, x_prob * (1.0 - x_prob) / n))
        se_y = math.sqrt(max(0.0, y_prob * (1.0 - y_prob) / n))
        se_z = math.sqrt(max(0.0, z_prob * (1.0 - z_prob) / n))

        x_low.append(max(0.0, x_prob - z_score * se_x))
        x_high.append(min(1.0, x_prob + z_score * se_x))
        y_low.append(max(0.0, y_prob - z_score * se_y))
        y_high.append(min(1.0, y_prob + z_score * se_y))
        z_low.append(max(0.0, z_prob - z_score * se_z))
        z_high.append(min(1.0, z_prob + z_score * se_z))

        s_xy = x_prob + y_prob
        if s_xy <= 0.0 or not math.isfinite(bias_value):
            b_low.append(math.nan)
            b_high.append(math.nan)
            continue

        var_bias = z_prob * (z_prob + s_xy) / (n * (s_xy ** 3))
        se_bias = math.sqrt(max(0.0, var_bias))
        b_low.append(max(0.0, bias_value - z_score * se_bias))
        b_high.append(bias_value + z_score * se_bias)

    return x_low, x_high, y_low, y_high, z_low, z_high, b_low, b_high


def _compute_bias_se_and_rk(
    x_probs: List[float],
    y_probs: List[float],
    z_probs: List[float],
    bias_values: List[float],
    totals: List[int],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute SE of bias, R_k metric, and relative precision for each iteration."""
    se_bias_list: List[float] = []
    rk_list: List[float] = []
    rel_precision_list: List[float] = []

    for idx, (x_prob, y_prob, z_prob, bias_value, total) in enumerate(
        zip(x_probs, y_probs, z_probs, bias_values, totals)
    ):
        n = float(total)
        if n <= 0.0:
            se_bias_list.append(math.nan)
            rk_list.append(math.nan)
            rel_precision_list.append(math.nan)
            continue

        s_xy = x_prob + y_prob
        if s_xy <= 0.0 or not math.isfinite(bias_value):
            se_bias_list.append(math.nan)
            rk_list.append(math.nan)
            rel_precision_list.append(math.nan)
            continue

        var_bias = z_prob * (z_prob + s_xy) / (n * (s_xy ** 3))
        se_bias = math.sqrt(max(0.0, var_bias))
        se_bias_list.append(se_bias)

        if math.isfinite(bias_value) and bias_value != 0.0:
            rel_precision_list.append(se_bias / abs(bias_value))
        else:
            rel_precision_list.append(math.nan)

        if idx == 0:
            rk_list.append(math.nan)
        else:
            prev_bias = bias_values[idx - 1]
            if math.isfinite(prev_bias) and se_bias > 0.0:
                rk_list.append(abs(bias_value - prev_bias) / se_bias)
            else:
                rk_list.append(math.nan)

    return se_bias_list, rk_list, rel_precision_list


def _compute_bias_difference_zk(
    x_probs: List[float],
    y_probs: List[float],
    z_probs: List[float],
    bias_values: List[float],
    totals: List[int],
) -> Tuple[List[float], List[float], List[float]]:
    """Compute Z_k metric: normalized bias difference with directional information."""
    zk_list: List[float] = []
    abs_zk_list: List[float] = []
    se_d_list: List[float] = []
    zk_rescaled_list: List[float] = []
    abs_zk_rescaled_list: List[float] = []
    zk_rescale_factor: float = math.nan

    for idx, (x_prob, y_prob, z_prob, bias_value, total) in enumerate(
        zip(x_probs, y_probs, z_probs, bias_values, totals)
    ):
        if idx == 0:
            zk_list.append(math.nan)
            abs_zk_list.append(math.nan)
            se_d_list.append(math.nan)
            continue

        prev_bias = bias_values[idx - 1]
        n_k_minus_1 = float(totals[idx - 1])
        n_k = float(total)
        s_k = x_prob + y_prob

        if s_k <= 0.0 or not math.isfinite(bias_value) or not math.isfinite(prev_bias):
            zk_list.append(math.nan)
            abs_zk_list.append(math.nan)
            se_d_list.append(math.nan)
            continue

        if n_k_minus_1 <= 0.0 or n_k <= 0.0 or n_k <= n_k_minus_1:
            zk_list.append(math.nan)
            abs_zk_list.append(math.nan)
            se_d_list.append(math.nan)
            continue

        d_k = bias_value - prev_bias
        var_factor = z_prob * (z_prob + s_k) / (s_k ** 3)
        var_d = max(0.0, var_factor * (1.0 / n_k_minus_1 - 1.0 / n_k))
        se_d = math.sqrt(var_d)
        se_d_list.append(se_d)

        if se_d > 0.0:
            z_k = d_k / se_d
            zk_list.append(z_k)
            abs_zk_list.append(abs(z_k))
        else:
            zk_list.append(math.nan)
            abs_zk_list.append(math.nan)

    return zk_list, abs_zk_list, se_d_list


def _filter_iteration_range(
    iterations: List[int],
    generated_samples: List[Optional[int]],
    x_probs: List[float],
    y_probs: List[float],
    z_probs: List[float],
    bias_values: List[float],
    iter_start: int | None,
    iter_end: int | None,
) -> Tuple[List[int], List[Optional[int]], List[float], List[float], List[float], List[float]]:
    if iter_start is None and iter_end is None:
        return iterations, generated_samples, x_probs, y_probs, z_probs, bias_values

    filtered_iterations: List[int] = []
    filtered_samples: List[Optional[int]] = []
    filtered_x: List[float] = []
    filtered_y: List[float] = []
    filtered_z: List[float] = []
    filtered_bias: List[float] = []

    for it, sample_count, x, y, z, b in zip(iterations, generated_samples, x_probs, y_probs, z_probs, bias_values):
        if iter_start is not None and it < iter_start:
            continue
        if iter_end is not None and it > iter_end:
            continue
        filtered_iterations.append(it)
        filtered_samples.append(sample_count)
        filtered_x.append(x)
        filtered_y.append(y)
        filtered_z.append(z)
        filtered_bias.append(b)

    if not filtered_iterations:
        raise ValueError(
            "No rows remain after applying iteration range filter: "
            f"start={iter_start}, end={iter_end}."
        )

    return filtered_iterations, filtered_samples, filtered_x, filtered_y, filtered_z, filtered_bias


def _summarize_zk_tail(
    iterations: List[int],
    zk_list: List[float],
    tail_start: int | None = None,
) -> Dict[str, float]:
    """Summarize the tail behavior of the Z_k sequence."""
    pairs = [
        (it, z)
        for it, z in zip(iterations, zk_list)
        if math.isfinite(z) and (tail_start is None or it >= tail_start)
    ]
    if len(pairs) < 2:
        return {}

    z = [val for _, val in pairs]
    m = len(z)

    mean_z = sum(z) / m
    var_z = sum((v - mean_z) ** 2 for v in z) / (m - 1)
    std_z = math.sqrt(var_z)

    mean_se = std_z / math.sqrt(m) if std_z > 0.0 else math.nan
    mean_t = mean_z / mean_se if math.isfinite(mean_se) and mean_se > 0.0 else math.nan

    frac_abs_gt_2 = sum(abs(v) > 2.0 for v in z) / m
    frac_abs_gt_3 = sum(abs(v) > 3.0 for v in z) / m

    n_pos = sum(v > 0 for v in z)
    n_neg = sum(v < 0 for v in z)
    sign_total = n_pos + n_neg
    sign_imbalance = ((n_pos - n_neg) / sign_total) if sign_total > 0 else math.nan

    longest_run = 0
    current_run = 0
    prev_sign = 0
    for v in z:
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign == 0:
            current_run = 0
            prev_sign = 0
            continue
        if sign == prev_sign:
            current_run += 1
        else:
            current_run = 1
            prev_sign = sign
        longest_run = max(longest_run, current_run)

    z0 = z[:-1]
    z1 = z[1:]
    mean0 = sum(z0) / len(z0)
    mean1 = sum(z1) / len(z1)
    num = sum((a - mean0) * (b - mean1) for a, b in zip(z0, z1))
    den0 = sum((a - mean0) ** 2 for a in z0)
    den1 = sum((b - mean1) ** 2 for b in z1)
    lag1_autocorr = num / math.sqrt(den0 * den1) if den0 > 0.0 and den1 > 0.0 else math.nan

    return {
        "count": float(m),
        "mean_z": mean_z,
        "std_z": std_z,
        "mean_se": mean_se,
        "mean_t": mean_t,
        "frac_abs_gt_2": frac_abs_gt_2,
        "frac_abs_gt_3": frac_abs_gt_3,
        "sign_imbalance": sign_imbalance,
        "longest_sign_run": float(longest_run),
        "lag1_autocorr": lag1_autocorr,
    }


def _print_zk_summary(summary: Dict[str, float], tail_start: int | None) -> None:
    if not summary:
        print("Z_k tail summary: insufficient finite points to summarize.")
        return
    start_label = "all finite iterations" if tail_start is None else f"iterations >= {tail_start}"
    print(f"Z_k tail summary ({start_label}):")
    order = [
        "count",
        "mean_z",
        "std_z",
        "mean_se",
        "mean_t",
        "frac_abs_gt_2",
        "frac_abs_gt_3",
        "sign_imbalance",
        "longest_sign_run",
        "lag1_autocorr",
    ]
    for key in order:
        value = summary.get(key, math.nan)
        if key in {"count", "longest_sign_run"}:
            print(f"  {key}: {int(value)}")
        else:
            print(f"  {key}: {value:.6g}")




def _summarize_bias_plateau_tail(
    iterations: List[int],
    bias_values: List[float],
    tail_start: int | None = None,
    n_blocks: int = 5,
) -> Dict[str, Any]:
    """Summarize tail plateau behavior of running bias using block medians.

    The tail is split into consecutive blocks of nearly equal size. We compute
    a robust representative (median) for each block, then measure the relative
    spread across block medians and the relative difference between the final
    two blocks.
    """
    pairs = [
        (it, b)
        for it, b in zip(iterations, bias_values)
        if math.isfinite(b) and (tail_start is None or it >= tail_start)
    ]
    if len(pairs) < max(2, n_blocks):
        return {}

    n = len(pairs)
    n_blocks = max(2, min(n_blocks, n))

    def _median(vals: List[float]) -> float:
        vals = sorted(vals)
        m = len(vals)
        if m % 2 == 1:
            return vals[m // 2]
        return 0.5 * (vals[m // 2 - 1] + vals[m // 2])

    block_ranges: List[Tuple[int, int]] = []
    block_medians: List[float] = []
    block_first_last: List[Tuple[float, float]] = []

    for j in range(n_blocks):
        start = (j * n) // n_blocks
        end = ((j + 1) * n) // n_blocks
        block = pairs[start:end]
        if not block:
            continue
        its = [it for it, _ in block]
        vals = [b for _, b in block]
        block_ranges.append((its[0], its[-1]))
        block_medians.append(_median(vals))
        block_first_last.append((vals[0], vals[-1]))

    if len(block_medians) < 2:
        return {}

    final_median = block_medians[-1]
    if not math.isfinite(final_median) or final_median == 0.0:
        return {}

    plateau_spread_rel = (max(block_medians) - min(block_medians)) / abs(final_median)
    last_prev_rel_diff = abs(block_medians[-1] - block_medians[-2]) / abs(final_median)
    tail_first = pairs[0][1]
    tail_last = pairs[-1][1]
    tail_endpoint_rel_drift = abs(tail_last - tail_first) / abs(final_median)

    return {
        "count": float(len(pairs)),
        "n_blocks": float(len(block_medians)),
        "final_median": final_median,
        "plateau_spread_rel": plateau_spread_rel,
        "last_prev_rel_diff": last_prev_rel_diff,
        "tail_endpoint_rel_drift": tail_endpoint_rel_drift,
        "block_ranges": block_ranges,
        "block_medians": block_medians,
        "block_first_last": block_first_last,
    }


def _print_bias_plateau_summary(summary: Dict[str, Any], tail_start: int | None) -> None:
    if not summary:
        print("Bias plateau summary: insufficient finite points to summarize.")
        return
    start_label = "all finite iterations" if tail_start is None else f"iterations >= {tail_start}"
    print(f"Bias plateau summary ({start_label}):")
    print(f"  count: {int(summary['count'])}")
    print(f"  n_blocks: {int(summary['n_blocks'])}")
    print(f"  final_median: {summary['final_median']:.6g}")
    print(f"  plateau_spread_rel: {summary['plateau_spread_rel']:.6g}")
    print(f"  last_prev_rel_diff: {summary['last_prev_rel_diff']:.6g}")
    print(f"  tail_endpoint_rel_drift: {summary['tail_endpoint_rel_drift']:.6g}")
    print("  block_medians:")
    for (it0, it1), med in zip(summary['block_ranges'], summary['block_medians']):
        print(f"    [{it0}, {it1}]: {med:.6g}")

def _rescale_zk_from_tail(
    iterations: List[int],
    zk_list: List[float],
    tail_start: int | None = None,
) -> Tuple[List[float], float]:
    """Rescale Z_k by the empirical tail standard deviation.

    This is the Option B calibration:
        Z_k^(rescaled) = Z_k / std_tail(raw Z_k)
    where std_tail is computed over the finite tail region used for diagnostics.
    """
    summary = _summarize_zk_tail(iterations, zk_list, tail_start=tail_start)
    std_tail = summary.get("std_z", math.nan)
    if not math.isfinite(std_tail) or std_tail <= 0.0:
        return [math.nan if not math.isfinite(z) else z for z in zk_list], math.nan

    return [
        (z / std_tail) if math.isfinite(z) else math.nan
        for z in zk_list
    ], std_tail


def plot_progress_file(
    progress_file: Path,
    output_path: Path | None = None,
    show: bool = False,
    smooth_window: int = 1,
    bias_log_scale: bool = False,
    iter_start: int | None = None,
    iter_end: int | None = None,
    x_axis: str = "iteration",
    counts_file: Path | None = None,
    error_bars: bool = False,
    error_z: float = 1.96,
    sigfig_overlay: bool = True,
    manual_sigfig: int | None = None,
    rel_prec_log_scale: bool = True,
    zk_tail_start: int | None = None,
    print_zk_summary: bool = True,
    show_zk_hist: bool = True,
    zk_rescale_tail: bool = True,
    plateau_tail_start: int | None = None,
    plateau_blocks: int = 5,
    print_plateau_summary: bool = True,
    show_plateau_overlay: bool = True,
) -> Path:
    """Create and save a 2x4 plot grid for probabilities, bias, and diagnostics."""
    all_iterations, all_generated_samples, all_x_probs, all_y_probs, all_z_probs, all_bias_values = _parse_progress_file(progress_file)

    selected_indices = [
        idx
        for idx, iteration in enumerate(all_iterations)
        if (iter_start is None or iteration >= iter_start) and (iter_end is None or iteration <= iter_end)
    ]
    if not selected_indices:
        raise ValueError(
            "No rows remain after applying iteration range filter: "
            f"start={iter_start}, end={iter_end}."
        )

    iterations = [all_iterations[idx] for idx in selected_indices]
    generated_samples = [all_generated_samples[idx] for idx in selected_indices]
    x_probs = [all_x_probs[idx] for idx in selected_indices]
    y_probs = [all_y_probs[idx] for idx in selected_indices]
    z_probs = [all_z_probs[idx] for idx in selected_indices]
    bias_values = [all_bias_values[idx] for idx in selected_indices]

    iterations, generated_samples, x_probs, y_probs, z_probs, bias_values = _filter_iteration_range(
        iterations,
        generated_samples,
        x_probs,
        y_probs,
        z_probs,
        bias_values,
        iter_start,
        iter_end,
    )

    if output_path is None:
        output_path = _build_output_path_with_flags(
            progress_file,
            iter_start=iter_start,
            iter_end=iter_end,
            smooth_window=smooth_window,
            bias_log_scale=bias_log_scale,
            x_axis=x_axis,
            manual_sigfig=manual_sigfig,
            zk_tail_start=zk_tail_start,
            zk_rescale_tail=zk_rescale_tail,
        )
    output_path = output_path.expanduser().resolve(strict=False)

    normalized_x_axis = x_axis.strip().lower()
    if normalized_x_axis not in {"iteration", "samples"}:
        raise ValueError("x_axis must be one of: 'iteration', 'samples'.")

    if normalized_x_axis == "samples":
        if any(sample_count is None for sample_count in generated_samples):
            raise ValueError(
                "Selected x-axis 'samples', but this progress file does not have "
                "Generated_Samples for all rows. Use x_axis='iteration' or regenerate data."
            )
        x_values = [int(sample_count) for sample_count in generated_samples]
        x_label = "Generated samples"
    else:
        x_values = iterations
        x_label = "Iteration"

    x_low = x_high = y_low = y_high = z_low = z_high = b_low = b_high = None
    x_sig = y_sig = z_sig = b_sig = None
    se_bias_list: List[float] = []
    rk_list: List[float] = []
    rel_precision_list: List[float] = []
    zk_list: List[float] = []
    abs_zk_list: List[float] = []
    se_d_list: List[float] = []
    zk_rescaled_list: List[float] = []
    abs_zk_rescaled_list: List[float] = []
    zk_rescale_factor: float = math.nan

    if error_bars:
        if counts_file is None:
            raise ValueError("--error-bars requires --counts-file.")
        if not counts_file.exists() or counts_file.is_dir():
            raise FileNotFoundError(f"Counts file not found or is not a file: {counts_file}")
        totals_all = _align_counts_to_progress_length(
            _parse_counts_totals(counts_file),
            len(all_iterations),
            "counts",
        )
        totals_filtered = [totals_all[idx] for idx in selected_indices]
        x_low, x_high, y_low, y_high, z_low, z_high, b_low, b_high = _compute_probability_cis(
            x_probs,
            y_probs,
            z_probs,
            bias_values,
            totals_filtered,
            error_z,
        )
        se_bias_list, rk_list, rel_precision_list = _compute_bias_se_and_rk(
            x_probs,
            y_probs,
            z_probs,
            bias_values,
            totals_filtered,
        )
        zk_list, abs_zk_list, se_d_list = _compute_bias_difference_zk(
            x_probs,
            y_probs,
            z_probs,
            bias_values,
            totals_filtered,
        )

        if zk_rescale_tail:
            zk_rescaled_list, zk_rescale_factor = _rescale_zk_from_tail(
                iterations,
                zk_list,
                tail_start=zk_tail_start,
            )
            abs_zk_rescaled_list = [abs(z) if math.isfinite(z) else math.nan for z in zk_rescaled_list]

    plateau_tail_start_effective = plateau_tail_start if plateau_tail_start is not None else zk_tail_start
    plateau_summary = _summarize_bias_plateau_tail(
        iterations,
        bias_values,
        tail_start=plateau_tail_start_effective,
        n_blocks=plateau_blocks,
    )

    if manual_sigfig is not None:
        if manual_sigfig < 1:
            raise ValueError("--manual-sigfig must be >= 1.")
        x_sig = _quantize_series_from_sigfigs(x_probs, manual_sigfig)
        y_sig = _quantize_series_from_sigfigs(y_probs, manual_sigfig)
        z_sig = _quantize_series_from_sigfigs(z_probs, manual_sigfig)
        b_sig = _quantize_series_from_sigfigs(bias_values, manual_sigfig)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=False)
    ax_x, ax_y, ax_z, ax_b, ax_rel_prec, ax_zk_signed, ax_zk_abs, ax_hist = axes.ravel()

    if smooth_window > 1:
        x_smooth = _moving_average(x_probs, smooth_window)
        y_smooth = _moving_average(y_probs, smooth_window)
        z_smooth = _moving_average(z_probs, smooth_window)
        bias_base = _sanitize_for_log(bias_values) if bias_log_scale else bias_values
        bias_smooth = _moving_average(bias_base, smooth_window)

        ax_x.plot(x_values, x_probs, color="tab:blue", linewidth=1.0, alpha=0.35, label="Raw")
        ax_x.plot(x_values, x_smooth, color="tab:blue", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
        ax_y.plot(x_values, y_probs, color="tab:orange", linewidth=1.0, alpha=0.35, label="Raw")
        ax_y.plot(x_values, y_smooth, color="tab:orange", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
        ax_z.plot(x_values, z_probs, color="tab:green", linewidth=1.0, alpha=0.35, label="Raw")
        ax_z.plot(x_values, z_smooth, color="tab:green", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
        ax_b.plot(x_values, bias_base, color="tab:red", linewidth=1.0, alpha=0.35, label="Raw")
        ax_b.plot(x_values, bias_smooth, color="tab:red", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
    else:
        bias_base = _sanitize_for_log(bias_values) if bias_log_scale else bias_values
        raw_alpha = 0.4 if (manual_sigfig is not None) else 1.0
        ax_x.plot(x_values, x_probs, color="tab:blue", linewidth=1.8, alpha=raw_alpha)
        ax_y.plot(x_values, y_probs, color="tab:orange", linewidth=1.8, alpha=raw_alpha)
        ax_z.plot(x_values, z_probs, color="tab:green", linewidth=1.8, alpha=raw_alpha)
        ax_b.plot(x_values, bias_base, color="tab:red", linewidth=1.8, alpha=raw_alpha)

    if manual_sigfig is not None and x_sig is not None and y_sig is not None and z_sig is not None and b_sig is not None and sigfig_overlay:
        label_text = f"{manual_sigfig} sig-fig"
        ax_x.plot(x_values, x_sig, color="tab:blue", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)
        ax_y.plot(x_values, y_sig, color="tab:orange", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)
        ax_z.plot(x_values, z_sig, color="tab:green", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)
        ax_b.plot(x_values, b_sig, color="tab:red", linewidth=1.6, linestyle="--", alpha=0.95, label=label_text)

    for axis in (ax_x, ax_y, ax_z, ax_b, ax_rel_prec, ax_zk_signed, ax_zk_abs):
        if zk_tail_start is not None:
            axis.axvline(zk_tail_start, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        if plateau_tail_start_effective is not None and plateau_tail_start_effective != zk_tail_start:
            axis.axvline(plateau_tail_start_effective, color="black", linestyle=":", linewidth=1.0, alpha=0.45)

    ax_x.set_title("X probability")
    ax_x.set_ylabel("X probability")
    ax_x.grid(True, alpha=0.3)
    if error_bars and x_low is not None and x_high is not None:
        ax_x.fill_between(x_values, x_low, x_high, color="tab:blue", alpha=0.18, linewidth=0)

    ax_y.set_title("Y probability")
    ax_y.set_ylabel("Y probability")
    ax_y.grid(True, alpha=0.3)
    if error_bars and y_low is not None and y_high is not None:
        ax_y.fill_between(x_values, y_low, y_high, color="tab:orange", alpha=0.18, linewidth=0)

    ax_z.set_title("Z probability")
    ax_z.set_ylabel("Z probability")
    ax_z.set_xlabel(x_label)
    ax_z.grid(True, alpha=0.3)
    if error_bars and z_low is not None and z_high is not None:
        ax_z.fill_between(x_values, z_low, z_high, color="tab:green", alpha=0.18, linewidth=0)

    ax_b.set_title("Bias")
    ax_b.set_ylabel("Bias")
    ax_b.set_xlabel(x_label)
    ax_b.grid(True, alpha=0.3)
    if bias_log_scale:
        ax_b.set_yscale("log")
    if error_bars and b_low is not None and b_high is not None:
        ax_b.fill_between(x_values, b_low, b_high, color="tab:red", alpha=0.18, linewidth=0)
    if show_plateau_overlay and plateau_summary:
        first_segment = True
        for (it0, it1), med in zip(plateau_summary["block_ranges"], plateau_summary["block_medians"]):
            ax_b.hlines(
                med,
                xmin=it0,
                xmax=it1,
                colors="black",
                linestyles="--",
                linewidth=1.4,
                alpha=0.85,
                label="Plateau block medians" if first_segment else None,
            )
            ax_b.axvspan(it0, it1, color="gray", alpha=0.05)
            first_segment = False

    if error_bars and rel_precision_list:
        ax_rel_prec.plot(x_values, rel_precision_list, color="tab:pink", linewidth=1.8, marker="^", markersize=3, alpha=0.7)
        ax_rel_prec.set_title("Relative Precision of Bias")
        ax_rel_prec.set_ylabel("SE_bias / |bias|")
        ax_rel_prec.set_xlabel(x_label)
        ax_rel_prec.grid(True, alpha=0.3)
        if rel_prec_log_scale:
            ax_rel_prec.set_yscale("log")
    else:
        ax_rel_prec.set_visible(False)

    if error_bars and zk_list:
        ax_zk_signed.plot(x_values, zk_list, color="tab:brown", linewidth=1.3, marker="o", markersize=2.5, alpha=0.65)
        ax_zk_signed.axhline(0.0, color="black", linestyle="-", linewidth=0.7, alpha=0.35)
        ax_zk_signed.axhline(1.96, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, label="±1.96")
        ax_zk_signed.axhline(-1.96, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
        ax_zk_signed.set_title("Z_k: Signed Bias Change")
        ax_zk_signed.set_ylabel("Z_k")
        ax_zk_signed.set_xlabel(x_label)
        ax_zk_signed.grid(True, alpha=0.3)
        ax_zk_signed.legend(loc="best", fontsize=8)

        ax_zk_abs.plot(
            x_values,
            abs_zk_list,
            color="tab:olive",
            linewidth=1.2,
            marker="s",
            markersize=2.2,
            alpha=0.55,
            label="Raw |Z_k|",
        )
        if zk_rescaled_list:
            ax_zk_abs.plot(
                x_values,
                abs_zk_rescaled_list,
                color="tab:cyan",
                linewidth=1.2,
                marker="^",
                markersize=2.0,
                alpha=0.65,
                label="Rescaled |Z_k|",
            )
        ax_zk_abs.axhline(1.96, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, label="1.96")
        ax_zk_abs.set_title("|Z_k| and rescaled |Z_k|")
        ax_zk_abs.set_ylabel("magnitude")
        ax_zk_abs.set_xlabel(x_label)
        ax_zk_abs.grid(True, alpha=0.3)
        ax_zk_abs.legend(loc="best", fontsize=8)
    else:
        ax_zk_signed.set_visible(False)
        ax_zk_abs.set_visible(False)

    if error_bars and zk_list and show_zk_hist:
        hist_pairs = [
            z for it, z in zip(iterations, zk_list)
            if math.isfinite(z) and (zk_tail_start is None or it >= zk_tail_start)
        ]
        hist_pairs_rescaled = [
            z for it, z in zip(iterations, zk_rescaled_list)
            if math.isfinite(z) and (zk_tail_start is None or it >= zk_tail_start)
        ] if zk_rescaled_list else []
        if hist_pairs or hist_pairs_rescaled:
            if hist_pairs:
                ax_hist.hist(
                    hist_pairs,
                    bins=30,
                    color="tab:gray",
                    alpha=0.45,
                    edgecolor="white",
                    label="Raw tail Z_k",
                )
            if hist_pairs_rescaled:
                ax_hist.hist(
                    hist_pairs_rescaled,
                    bins=30,
                    color="tab:cyan",
                    alpha=0.45,
                    edgecolor="white",
                    label="Rescaled tail Z_k",
                )
            ax_hist.axvline(0.0, color="black", linestyle="-", linewidth=0.7, alpha=0.35)
            ax_hist.axvline(2.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.5)
            ax_hist.axvline(-2.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.5)
            title_tail = "all finite" if zk_tail_start is None else f"tail ≥ {zk_tail_start}"
            ax_hist.set_title(f"Histogram of raw/rescaled Z_k ({title_tail})")
            ax_hist.set_xlabel("Z_k")
            ax_hist.set_ylabel("Count")
            ax_hist.grid(True, alpha=0.3)
            ax_hist.legend(loc="best", fontsize=8)
        else:
            ax_hist.set_visible(False)
    else:
        ax_hist.set_visible(False)

    if smooth_window > 1 or (manual_sigfig is not None and sigfig_overlay):
        for axis in (ax_x, ax_y, ax_z, ax_b):
            axis.legend(loc="best", fontsize=8)

    if error_bars and zk_list and print_zk_summary:
        summary = _summarize_zk_tail(iterations, zk_list, tail_start=zk_tail_start)
        _print_zk_summary(summary, tail_start=zk_tail_start)
        if zk_rescaled_list:
            print(f"Z_k rescale factor (empirical tail std): {zk_rescale_factor:.6g}")
            summary_rescaled = _summarize_zk_tail(iterations, zk_rescaled_list, tail_start=zk_tail_start)
            print("Rescaled Z_k tail summary:")
            _print_zk_summary(summary_rescaled, tail_start=zk_tail_start)

    if print_plateau_summary:
        _print_bias_plateau_summary(plateau_summary, tail_start=plateau_tail_start_effective)

    iter_label_start = "min" if iter_start is None else str(iter_start)
    iter_label_end = "max" if iter_end is None else str(iter_end)
    flags_label = (
        f"iter_range={iter_label_start}-{iter_label_end}, smooth_window={smooth_window}, "
        f"bias_log_scale={bias_log_scale}, x_axis={normalized_x_axis}, error_bars={error_bars}, "
        f"error_z={error_z}, zk_tail_start={zk_tail_start}, zk_rescale_tail={zk_rescale_tail}, "
        f"plateau_tail_start={plateau_tail_start_effective}, plateau_blocks={plateau_blocks}"
    )
    fig.suptitle(f"Convergence metrics from {progress_file.name}")
    fig.text(0.5, 0.01, f"Flags: {flags_label}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, format="svg")
    except FileNotFoundError:
        fallback_output = (Path.cwd() / output_path.name).resolve()
        fallback_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fallback_output, format="svg")
        output_path = fallback_output

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot X/Y/Z probability and bias vs iteration from progress_file."
    )
    parser.add_argument("--progress-file", type=Path, required=True, help="Path to effective_probs_<platform>_<timestamp>.txt")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: auto-named .svg)")
    parser.add_argument("--show", action="store_true", help="Display the plot window in addition to saving the file.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Centered moving-average window size (<=1 disables smoothing).")
    parser.add_argument("--bias-log-scale", action="store_true", help="Use logarithmic y-axis for the bias subplot.")
    parser.add_argument("--iter-start", type=int, default=None, help="Minimum iteration to include (inclusive).")
    parser.add_argument("--iter-end", type=int, default=None, help="Maximum iteration to include (inclusive).")
    parser.add_argument("--x-axis", choices=["iteration", "samples"], default="iteration", help="X-axis to use. 'samples' requires Generated_Samples column.")
    parser.add_argument("--preset", choices=["basic", "uncertainty", "quality", "sigfig"], default="basic", help="Convenience preset.")
    parser.add_argument("--counts-file", type=Path, default=None, help="Path to running_counts_<platform>_<timestamp>.jsonl used for uncertainty bands.")
    parser.add_argument("--error-bars", action="store_true", help="Plot approximate confidence bands (requires --counts-file).")
    parser.add_argument("--error-z", type=float, default=1.96, help="Z-score multiplier for uncertainty bands (default: 1.96).")
    parser.add_argument("--no-sigfig-overlay", action="store_true", help="Disable drawing the significant-figure overlay line.")
    parser.add_argument("--manual-sigfig", type=int, default=None, help="Overlay a line rounded to N significant figures.")
    parser.add_argument("--rel-prec-linear", action="store_true", help="Use linear y-axis scale for relative precision plot.")
    parser.add_argument("--zk-tail-start", type=int, default=None, help="Iteration at which to start Z_k tail diagnostics and histogram.")
    parser.add_argument("--no-zk-summary", action="store_true", help="Disable printing Z_k tail summary statistics.")
    parser.add_argument("--no-zk-hist", action="store_true", help="Disable histogram subplot for Z_k tail values.")
    parser.add_argument("--no-zk-rescale-tail", action="store_true", help="Disable Option B rescaling by the empirical tail std of Z_k.")
    parser.add_argument("--plateau-tail-start", type=int, default=None, help="Iteration at which to start bias plateau diagnostics (default: same as --zk-tail-start).")
    parser.add_argument("--plateau-blocks", type=int, default=5, help="Number of blocks for plateau analysis (default: 5).")
    parser.add_argument("--no-plateau-summary", action="store_true", help="Disable printing bias plateau summary statistics.")
    parser.add_argument("--no-plateau-overlay", action="store_true", help="Disable overlay of plateau block medians on bias plot.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    progress_file = args.progress_file
    if not progress_file.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_file}")
    if progress_file.is_dir():
        raise ValueError(
            "--progress-file points to a directory, not a file. "
            "Pass the full path to effective_probs_<platform>_<timestamp>.txt"
        )
    if args.iter_start is not None and args.iter_end is not None and args.iter_start > args.iter_end:
        raise ValueError("--iter-start must be <= --iter-end.")

    if args.preset == "uncertainty":
        args.error_bars = True

    output_path = plot_progress_file(
        progress_file=progress_file,
        output_path=args.output,
        show=args.show,
        smooth_window=args.smooth_window,
        bias_log_scale=args.bias_log_scale,
        iter_start=args.iter_start,
        iter_end=args.iter_end,
        x_axis=args.x_axis,
        counts_file=args.counts_file,
        error_bars=args.error_bars,
        error_z=args.error_z,
        sigfig_overlay=not args.no_sigfig_overlay,
        manual_sigfig=args.manual_sigfig,
        rel_prec_log_scale=not args.rel_prec_linear,
        zk_tail_start=args.zk_tail_start,
        print_zk_summary=not args.no_zk_summary,
        show_zk_hist=not args.no_zk_hist,
        zk_rescale_tail=not args.no_zk_rescale_tail,
        plateau_tail_start=args.plateau_tail_start,
        plateau_blocks=args.plateau_blocks,
        print_plateau_summary=not args.no_plateau_summary,
        show_plateau_overlay=not args.no_plateau_overlay,
    )
    print(f"Saved plots to: {output_path}")


if __name__ == "__main__":
    main()
