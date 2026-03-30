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
    suffix = "_".join(parts)
    return progress_file.with_name(f"{stem}_{suffix}.png")


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
) -> Path:
    """Create and save a grid for X/Y/Z convergence and optional bias convergence vs iteration."""
    iterations, generated_samples, x_conv, y_conv, z_conv, bias_conv, convergence_mode = _parse_progress_file(progress_file)
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

    ax_y.set_title("Y convergence")
    ax_y.set_ylabel("Y convergence")
    ax_y.grid(True, alpha=0.3)

    ax_z.set_title("Z convergence")
    ax_z.set_ylabel("Z convergence")
    ax_z.set_xlabel(x_label)
    ax_z.grid(True, alpha=0.3)

    if with_bias_convergence and ax_b is not None:
        ax_b.set_title("Bias convergence")
        ax_b.set_ylabel("Bias convergence")
        ax_b.set_xlabel(x_label)
        ax_b.grid(True, alpha=0.3)

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
        f"detected_mode={convergence_mode if convergence_mode is not None else 'unknown'}"
    )
    fig.suptitle(f"Convergence deltas from {progress_file.name}")
    fig.text(0.5, 0.01, f"Flags: {flags_label}", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=200)
    except FileNotFoundError:
        fallback_output = (Path.cwd() / output_path.name).resolve()
        fallback_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fallback_output, dpi=200)
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
    )
    print(f"Saved plots to: {output_path}")


if __name__ == "__main__":
    main()
