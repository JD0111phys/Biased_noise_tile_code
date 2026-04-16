"""Plot convergence metrics from an effective_probs progress file.

This script reads the text file produced by `error_propagation_simulation`
(`effective_probs_<platform>_<timestamp>.txt`) and plots four curves versus
iteration:
- X convergence
- Y convergence
- Z convergence

Rows that contain non-numeric convergence fields (e.g. iteration 0 with
"Initial") are skipped.

Example:
    python -m pauli_distribution.plot_convergence_file \
        --progress-file effective_probs_superconducting_20260329.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def _parse_progress_file(
    progress_file: Path,
) -> Tuple[List[int], List[Optional[int]], List[float], List[float], List[float], List[float], Optional[str]]:
    """Parse progress data and optional convergence mode metadata."""
    iterations: List[int] = []
    generated_samples: List[Optional[int]] = []
    x_conv: List[float] = []
    y_conv: List[float] = []
    z_conv: List[float] = []
    bias_values: List[float] = []
    convergence_mode: Optional[str] = None

    with progress_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                mode_match = re.search(r"mode\s*=\s*([A-Za-z_]+)", line)
                if mode_match is not None:
                    convergence_mode = mode_match.group(1).lower()
                continue

            row = next(csv.reader([line]))
            if len(row) < 11:
                continue

            try:
                iteration = int(row[0])
                bias_value = float(row[5])
                x_value = float(row[7])
                y_value = float(row[8])
                z_value = float(row[9])
                sample_value: Optional[int] = None
                if len(row) >= 13 and row[12].strip() != "":
                    sample_value = int(row[12])
            except ValueError:
                # Skip rows like iteration 0 that contain "Initial".
                continue

            # Append atomically to keep x/y dimensions aligned for plotting.
            iterations.append(iteration)
            x_conv.append(x_value)
            y_conv.append(y_value)
            z_conv.append(z_value)
            bias_values.append(bias_value)
            generated_samples.append(sample_value)

    if not iterations:
        raise ValueError(f"No numeric convergence rows were parsed from: {progress_file}")
    if not (len(iterations) == len(x_conv) == len(y_conv) == len(z_conv) == len(bias_values)):
        raise ValueError("Parsed convergence arrays have inconsistent lengths.")

    bias_conv: List[float] = [math.nan]
    for i in range(1, len(bias_values)):
        current = bias_values[i]
        previous = bias_values[i - 1]
        if math.isfinite(current) and math.isfinite(previous):
            bias_conv.append(abs(current - previous))
        else:
            bias_conv.append(math.nan)

    return iterations, generated_samples, x_conv, y_conv, z_conv, bias_conv, convergence_mode


def _build_output_path_with_flags(
    progress_file: Path,
    iter_start: int | None,
    iter_end: int | None,
    smooth_window: int,
    log_scale: bool,
    with_bias_convergence: bool,
    x_axis: str,
    manual_sigfig: int | None = None,
) -> Path:
    stem = progress_file.stem
    parts = ["convergence_plots"]
    if iter_start is not None or iter_end is not None:
        start_label = "min" if iter_start is None else str(iter_start)
        end_label = "max" if iter_end is None else str(iter_end)
        parts.append(f"iter_{start_label}-{end_label}")
    if smooth_window > 1:
        parts.append(f"smooth_{smooth_window}")
    if log_scale:
        parts.append("log")
    if with_bias_convergence:
        parts.append("biasconv")
    if x_axis == "samples":
        parts.append("x_samples")
    if manual_sigfig is not None:
        parts.append(f"sigfig{manual_sigfig}")
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


def _filter_iteration_range(
    iterations: List[int],
    generated_samples: List[Optional[int]],
    x_conv: List[float],
    y_conv: List[float],
    z_conv: List[float],
    bias_conv: List[float],
    iter_start: int | None,
    iter_end: int | None,
) -> Tuple[List[int], List[Optional[int]], List[float], List[float], List[float], List[float]]:
    if iter_start is None and iter_end is None:
        return iterations, generated_samples, x_conv, y_conv, z_conv, bias_conv

    filtered_iterations: List[int] = []
    filtered_samples: List[Optional[int]] = []
    filtered_x: List[float] = []
    filtered_y: List[float] = []
    filtered_z: List[float] = []
    filtered_bias: List[float] = []

    for it, sample_count, xc, yc, zc, bc in zip(iterations, generated_samples, x_conv, y_conv, z_conv, bias_conv):
        if iter_start is not None and it < iter_start:
            continue
        if iter_end is not None and it > iter_end:
            continue
        filtered_iterations.append(it)
        filtered_samples.append(sample_count)
        filtered_x.append(xc)
        filtered_y.append(yc)
        filtered_z.append(zc)
        filtered_bias.append(bc)

    if not filtered_iterations:
        raise ValueError(
            "No rows remain after applying iteration range filter: "
            f"start={iter_start}, end={iter_end}."
        )

    return filtered_iterations, filtered_samples, filtered_x, filtered_y, filtered_z, filtered_bias


def _has_finite(values: List[float]) -> bool:
    return any(math.isfinite(v) for v in values)


def _compute_convergence_from_stats(
    count_stats: List[Tuple[float, float, float, float, float, float, float, float]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute convergence (absolute change) from count statistics.
    
    Takes probabilities from consecutive count snapshots and computes
    the absolute change in each Pauli probability.
    
    Args:
        count_stats: List of (p_x, p_y, p_z, bias, se_x, se_y, se_z, se_bias) tuples.
    
    Returns:
        Tuple of (x_conv, y_conv, z_conv, bias_conv) lists.
    """
    x_conv: List[float] = [math.nan]  # First point has no previous value
    y_conv: List[float] = [math.nan]
    z_conv: List[float] = [math.nan]
    bias_conv: List[float] = [math.nan]
    
    for i in range(1, len(count_stats)):
        current = count_stats[i]
        previous = count_stats[i - 1]
        
        # Current values: (p_x, p_y, p_z, bias, se_x, se_y, se_z, se_bias)
        c_x, c_y, c_z, c_bias = current[0], current[1], current[2], current[3]
        p_x, p_y, p_z, p_bias = previous[0], previous[1], previous[2], previous[3]
        
        # Convergence is absolute change
        if math.isfinite(c_x) and math.isfinite(p_x):
            x_conv.append(abs(c_x - p_x))
        else:
            x_conv.append(math.nan)
        
        if math.isfinite(c_y) and math.isfinite(p_y):
            y_conv.append(abs(c_y - p_y))
        else:
            y_conv.append(math.nan)
        
        if math.isfinite(c_z) and math.isfinite(p_z):
            z_conv.append(abs(c_z - p_z))
        else:
            z_conv.append(math.nan)
        
        if math.isfinite(c_bias) and math.isfinite(p_bias):
            bias_conv.append(abs(c_bias - p_bias))
        else:
            bias_conv.append(math.nan)
    
    return x_conv, y_conv, z_conv, bias_conv


def _parse_counts_stats(counts_file: Path) -> List[Tuple[float, float, float, float, float, float, float, float]]:
    """Parse cumulative count rows into probabilities and standard errors.

    Returns tuples of:
        (p_x, p_y, p_z, bias, se_x, se_y, se_z, se_bias)
    """
    stats: List[Tuple[float, float, float, float, float, float, float, float]] = []
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
                # Skip metadata rows that do not carry Pauli counts.
                continue

            try:
                c_i = int(counts_obj.get("0", counts_obj.get(0, 0)))
                c_x = int(counts_obj.get("1", counts_obj.get(1, 0)))
                c_y = int(counts_obj.get("2", counts_obj.get(2, 0)))
                c_z = int(counts_obj.get("3", counts_obj.get(3, 0)))
            except (TypeError, ValueError):
                continue
            total = c_i + c_x + c_y + c_z
            if total <= 0:
                stats.append((math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan))
                continue

            n = float(total)
            p_x = c_x / n
            p_y = c_y / n
            p_z = c_z / n
            s_xy = p_x + p_y
            bias = p_z / s_xy if s_xy > 0.0 else math.inf

            se_x = math.sqrt(max(0.0, p_x * (1.0 - p_x) / n))
            se_y = math.sqrt(max(0.0, p_y * (1.0 - p_y) / n))
            se_z = math.sqrt(max(0.0, p_z * (1.0 - p_z) / n))

            if s_xy > 0.0 and math.isfinite(bias):
                var_bias = p_z * (p_z + s_xy) / (n * (s_xy ** 3))
                se_bias = math.sqrt(max(0.0, var_bias))
            else:
                se_bias = math.nan

            stats.append((p_x, p_y, p_z, bias, se_x, se_y, se_z, se_bias))

    if not stats:
        raise ValueError(f"No valid count rows parsed from: {counts_file}")
    return stats


def _compute_convergence_cis(
    x_conv: List[float],
    y_conv: List[float],
    z_conv: List[float],
    bias_conv: List[float],
    count_stats: List[Tuple[float, float, float, float, float, float, float, float]],
    z_score: float,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Compute CI bounds for convergence deltas using propagated row-to-row SE."""
    n_rows = len(x_conv)
    if len(count_stats) == n_rows + 1:
        offset = 1
    elif len(count_stats) == n_rows:
        offset = 0
    else:
        raise ValueError(
            "counts file rows do not align with parsed convergence rows. "
            f"counts_rows={len(count_stats)}, convergence_rows={n_rows}"
        )

    x_low: List[float] = []
    x_high: List[float] = []
    y_low: List[float] = []
    y_high: List[float] = []
    z_low: List[float] = []
    z_high: List[float] = []
    b_low: List[float] = []
    b_high: List[float] = []

    for idx in range(n_rows):
        current = count_stats[idx + offset]
        previous = count_stats[idx + offset - 1] if idx + offset - 1 >= 0 else None

        if previous is None:
            se_dx = se_dy = se_dz = se_db = math.nan
        else:
            se_dx = math.sqrt(current[4] ** 2 + previous[4] ** 2) if math.isfinite(current[4]) and math.isfinite(previous[4]) else math.nan
            se_dy = math.sqrt(current[5] ** 2 + previous[5] ** 2) if math.isfinite(current[5]) and math.isfinite(previous[5]) else math.nan
            se_dz = math.sqrt(current[6] ** 2 + previous[6] ** 2) if math.isfinite(current[6]) and math.isfinite(previous[6]) else math.nan
            se_db = math.sqrt(current[7] ** 2 + previous[7] ** 2) if math.isfinite(current[7]) and math.isfinite(previous[7]) else math.nan

        x_value = x_conv[idx]
        y_value = y_conv[idx]
        z_value = z_conv[idx]
        b_value = bias_conv[idx]

        if math.isfinite(x_value) and math.isfinite(se_dx):
            x_low.append(max(0.0, x_value - z_score * se_dx))
            x_high.append(x_value + z_score * se_dx)
        else:
            x_low.append(math.nan)
            x_high.append(math.nan)

        if math.isfinite(y_value) and math.isfinite(se_dy):
            y_low.append(max(0.0, y_value - z_score * se_dy))
            y_high.append(y_value + z_score * se_dy)
        else:
            y_low.append(math.nan)
            y_high.append(math.nan)

        if math.isfinite(z_value) and math.isfinite(se_dz):
            z_low.append(max(0.0, z_value - z_score * se_dz))
            z_high.append(z_value + z_score * se_dz)
        else:
            z_low.append(math.nan)
            z_high.append(math.nan)

        if math.isfinite(b_value) and math.isfinite(se_db):
            b_low.append(max(0.0, b_value - z_score * se_db))
            b_high.append(b_value + z_score * se_db)
        else:
            b_low.append(math.nan)
            b_high.append(math.nan)

    return x_low, x_high, y_low, y_high, z_low, z_high, b_low, b_high


def plot_convergence_file(
    progress_file: Path,
    output_path: Path | None = None,
    show: bool = False,
    smooth_window: int = 1,
    log_scale: bool = False,
    iter_start: int | None = None,
    iter_end: int | None = None,
    with_bias_convergence: bool = True,
    x_axis: str = "iteration",
    counts_file: Path | None = None,
    error_bars: bool = False,
    error_z: float = 1.96,
) -> Path:
    """Create and save a grid for X/Y/Z convergence and optional bias convergence vs iteration."""
    all_iterations, all_generated_samples, all_x_conv, all_y_conv, all_z_conv, all_bias_conv, convergence_mode = _parse_progress_file(progress_file)

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
    x_conv = [all_x_conv[idx] for idx in selected_indices]
    y_conv = [all_y_conv[idx] for idx in selected_indices]
    z_conv = [all_z_conv[idx] for idx in selected_indices]
    bias_conv = [all_bias_conv[idx] for idx in selected_indices]

    iterations, generated_samples, x_conv, y_conv, z_conv, bias_conv = _filter_iteration_range(
        iterations,
        generated_samples,
        x_conv,
        y_conv,
        z_conv,
        bias_conv,
        iter_start,
        iter_end,
    )

    if output_path is None:
        output_path = _build_output_path_with_flags(
            progress_file,
            iter_start=iter_start,
            iter_end=iter_end,
            smooth_window=smooth_window,
            log_scale=log_scale,
            with_bias_convergence=with_bias_convergence,
            x_axis=x_axis,
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
    if error_bars:
        if counts_file is None:
            raise ValueError("--error-bars requires --counts-file.")
        if not counts_file.exists() or counts_file.is_dir():
            raise FileNotFoundError(f"Counts file not found or is not a file: {counts_file}")
        count_stats_all = _parse_counts_stats(counts_file)
        if len(count_stats_all) < len(all_iterations):
            raise ValueError(
                "counts file rows do not align with parsed convergence rows. "
                f"counts_rows={len(count_stats_all)}, convergence_rows={len(all_iterations)}"
            )
        if len(count_stats_all) > len(all_iterations):
            # Older runs may contain one or more extra trailing count rows.
            aligned_count_stats_all = count_stats_all[-len(all_iterations):]
        else:
            aligned_count_stats_all = count_stats_all
        count_stats = [aligned_count_stats_all[idx] for idx in selected_indices]
        x_low, x_high, y_low, y_high, z_low, z_high, b_low, b_high = _compute_convergence_cis(
            x_conv,
            y_conv,
            z_conv,
            bias_conv,
            count_stats,
            error_z,
        )

    # If convergence not computed in progress file, try to compute from counts
    if (not _has_finite(x_conv)) and counts_file is not None:
        if counts_file.exists() and not counts_file.is_dir():
            count_stats_all = _parse_counts_stats(counts_file)
            if len(count_stats_all) >= len(all_iterations):
                if len(count_stats_all) > len(all_iterations):
                    # Older runs may contain one or more extra trailing count rows.
                    aligned_count_stats_all = count_stats_all[-len(all_iterations):]
                else:
                    aligned_count_stats_all = count_stats_all
                count_stats_selected = [aligned_count_stats_all[idx] for idx in selected_indices]
                # Compute convergence from counts statistics
                x_conv, y_conv, z_conv, bias_conv = _compute_convergence_from_stats(count_stats_selected)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_x, ax_y, ax_z, ax_fourth = axes.ravel()
    ax_b = ax_fourth if with_bias_convergence else None
    if not with_bias_convergence:
        ax_fourth.axis("off")

    if log_scale:
        x_base = _sanitize_for_log(x_conv)
        y_base = _sanitize_for_log(y_conv)
        z_base = _sanitize_for_log(z_conv)
        b_base = _sanitize_for_log(bias_conv)
    else:
        x_base, y_base, z_base = x_conv, y_conv, z_conv
        b_base = bias_conv

    if smooth_window > 1:
        x_smooth = _moving_average(x_base, smooth_window)
        y_smooth = _moving_average(y_base, smooth_window)
        z_smooth = _moving_average(z_base, smooth_window)
        b_smooth = _moving_average(b_base, smooth_window)

        if _has_finite(x_base):
            ax_x.plot(x_values, x_base, color="tab:blue", linewidth=1.0, alpha=0.35, label="Raw")
            ax_x.plot(x_values, x_smooth, color="tab:blue", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
        else:
            ax_x.text(0.5, 0.5, "Not computed for this mode", transform=ax_x.transAxes, ha="center", va="center")

        if _has_finite(y_base):
            ax_y.plot(x_values, y_base, color="tab:orange", linewidth=1.0, alpha=0.35, label="Raw")
            ax_y.plot(x_values, y_smooth, color="tab:orange", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
        else:
            ax_y.text(0.5, 0.5, "Not computed for this mode", transform=ax_y.transAxes, ha="center", va="center")

        if _has_finite(z_base):
            ax_z.plot(x_values, z_base, color="tab:green", linewidth=1.0, alpha=0.35, label="Raw")
            ax_z.plot(x_values, z_smooth, color="tab:green", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
        else:
            ax_z.text(0.5, 0.5, "Not computed for this mode", transform=ax_z.transAxes, ha="center", va="center")
        if with_bias_convergence and ax_b is not None:
            if _has_finite(b_base):
                ax_b.plot(x_values, b_base, color="tab:purple", linewidth=1.0, alpha=0.35, label="Raw")
                ax_b.plot(x_values, b_smooth, color="tab:purple", linewidth=2.0, label=f"Smoothed (w={smooth_window})")
            else:
                ax_b.text(0.5, 0.5, "Not computed for this mode", transform=ax_b.transAxes, ha="center", va="center")
    else:
        if _has_finite(x_base):
            ax_x.plot(x_values, x_base, color="tab:blue", linewidth=1.8)
        else:
            ax_x.text(0.5, 0.5, "Not computed for this mode", transform=ax_x.transAxes, ha="center", va="center")
        if _has_finite(y_base):
            ax_y.plot(x_values, y_base, color="tab:orange", linewidth=1.8)
        else:
            ax_y.text(0.5, 0.5, "Not computed for this mode", transform=ax_y.transAxes, ha="center", va="center")
        if _has_finite(z_base):
            ax_z.plot(x_values, z_base, color="tab:green", linewidth=1.8)
        else:
            ax_z.text(0.5, 0.5, "Not computed for this mode", transform=ax_z.transAxes, ha="center", va="center")
        if with_bias_convergence and ax_b is not None:
            if _has_finite(b_base):
                ax_b.plot(x_values, b_base, color="tab:purple", linewidth=1.8)
            else:
                ax_b.text(0.5, 0.5, "Not computed for this mode", transform=ax_b.transAxes, ha="center", va="center")

    ax_x.set_title("X convergence")
    ax_x.set_ylabel("X convergence")
    ax_x.grid(True, alpha=0.3)
    if error_bars and x_low is not None and x_high is not None:
        ax_x.fill_between(x_values, x_low, x_high, color="tab:blue", alpha=0.18, linewidth=0)

    ax_y.set_title("Y convergence")
    ax_y.set_ylabel("Y convergence")
    ax_y.grid(True, alpha=0.3)
    if error_bars and y_low is not None and y_high is not None:
        ax_y.fill_between(x_values, y_low, y_high, color="tab:orange", alpha=0.18, linewidth=0)

    ax_z.set_title("Z convergence")
    ax_z.set_ylabel("Z convergence")
    ax_z.set_xlabel(x_label)
    ax_z.grid(True, alpha=0.3)
    if error_bars and z_low is not None and z_high is not None:
        ax_z.fill_between(x_values, z_low, z_high, color="tab:green", alpha=0.18, linewidth=0)

    if with_bias_convergence and ax_b is not None:
        ax_b.set_title("Bias convergence")
        ax_b.set_ylabel("Bias convergence")
        ax_b.set_xlabel(x_label)
        ax_b.grid(True, alpha=0.3)
        if error_bars and b_low is not None and b_high is not None:
            ax_b.fill_between(x_values, b_low, b_high, color="tab:purple", alpha=0.18, linewidth=0)

    if log_scale:
        axes_to_scale = [ax_x, ax_y, ax_z]
        if with_bias_convergence and ax_b is not None:
            axes_to_scale.append(ax_b)
        for axis in axes_to_scale:
            axis.set_yscale("log")

    if smooth_window > 1:
        axes_with_legend = [ax_x, ax_y, ax_z]
        if with_bias_convergence and ax_b is not None:
            axes_with_legend.append(ax_b)
        for axis in axes_with_legend:
            axis.legend(loc="best", fontsize=8)

    iter_label_start = "min" if iter_start is None else str(iter_start)
    iter_label_end = "max" if iter_end is None else str(iter_end)
    flags_label = (
        f"iter_range={iter_label_start}-{iter_label_end}, "
        f"smooth_window={smooth_window}, log_scale={log_scale}, "
        f"with_bias_convergence={with_bias_convergence}, x_axis={normalized_x_axis}, "
        f"detected_mode={convergence_mode if convergence_mode is not None else 'unknown'}, "
        f"error_bars={error_bars}, error_z={error_z}"
    )
    fig.suptitle(f"Convergence deltas from {progress_file.name}")
    fig.text(0.5, 0.01, f"Flags: {flags_label}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, format='svg')
    except FileNotFoundError:
        fallback_output = (Path.cwd() / output_path.name).resolve()
        fallback_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fallback_output, format='svg')
        output_path = fallback_output

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot X/Y/Z convergence (and optional bias convergence) vs iteration from progress_file."
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        required=True,
        help="Path to effective_probs_<platform>_<timestamp>.txt",
    )
    parser.add_argument(
        "--preset",
        choices=["basic", "uncertainty"],
        default="basic",
        help="Convenience preset that enables a recommended set of options.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: auto name with active flags)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the file.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window size (<=1 disables smoothing).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic y-axis for all convergence subplots.",
    )
    parser.add_argument(
        "--iter-start",
        type=int,
        default=None,
        help="Minimum iteration to include (inclusive).",
    )
    parser.add_argument(
        "--iter-end",
        type=int,
        default=None,
        help="Maximum iteration to include (inclusive).",
    )
    parser.add_argument(
        "--with-bias-convergence",
        action="store_true",
        help="Kept for backward compatibility (bias convergence is enabled by default).",
    )
    parser.add_argument(
        "--no-bias-convergence",
        action="store_true",
        help="Disable plotting computed bias convergence.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["iteration", "samples"],
        default="iteration",
        help="X-axis to use. 'samples' requires Generated_Samples column in the progress file.",
    )
    parser.add_argument(
        "--counts-file",
        type=Path,
        default=None,
        help="Path to running_counts_<platform>_<timestamp>.jsonl used for uncertainty bands.",
    )
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help="Plot approximate confidence bands for convergence deltas (requires --counts-file).",
    )
    parser.add_argument(
        "--error-z",
        type=float,
        default=1.96,
        help="Z-score multiplier for uncertainty bands (default: 1.96).",
    )
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

    output_path = plot_convergence_file(
        progress_file=progress_file,
        output_path=args.output,
        show=args.show,
        smooth_window=args.smooth_window,
        log_scale=args.log_scale,
        iter_start=args.iter_start,
        iter_end=args.iter_end,
        with_bias_convergence=not args.no_bias_convergence,
        x_axis=args.x_axis,
        counts_file=args.counts_file,
        error_bars=args.error_bars,
        error_z=args.error_z,
    )
    print(f"Saved plots to: {output_path}")


if __name__ == "__main__":
    main()
