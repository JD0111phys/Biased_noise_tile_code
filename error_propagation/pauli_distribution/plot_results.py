"""Convenience wrapper for plotting simulation results.

This module provides high-level functions to easily generate plots from
simulation output directories created by error_propagation_simulation().

Example:
    from pathlib import Path
    from pauli_distribution import plot_simulation_results
    
    # Plot all results in an output directory
    output_dir = Path("trial_css_ideal_cnot_10000.0_p_0.003_chosen_seed_100001_20260401_024757")
    plot_simulation_results(output_dir)
"""

from pathlib import Path
from typing import Optional
from .plot_progress_file import plot_progress_file
from .plot_convergence_file import plot_convergence_file


def plot_simulation_results(
    output_dir: Path,
    show: bool = False,
    error_bars: bool = True,
    convergence_error_bars: bool = False,
) -> dict[str, Path]:
    """Generate progress and convergence plots from a simulation output directory.
    
    Args:
        output_dir: Path to the simulation output directory containing
                   effective_probs_*.txt and running_counts_*.jsonl files.
        show: Whether to display plots interactively (requires matplotlib backend).
        error_bars: Include error bars in progress plot.
        convergence_error_bars: Include error bars in convergence plot.
    
    Returns:
        Dictionary with keys 'progress' and 'convergence' containing Path objects
        to the generated SVG files.
    
    Raises:
        FileNotFoundError: If progress or counts files are not found in directory.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError(f"Output directory not found: {output_dir}")
    
    # Find progress and counts files
    progress_files = list(output_dir.glob("effective_probs_*.txt"))
    counts_files = list(output_dir.glob("running_counts_*.jsonl"))
    
    if not progress_files:
        raise FileNotFoundError(
            f"No progress file (effective_probs_*.txt) found in {output_dir}"
        )
    if not counts_files:
        raise FileNotFoundError(
            f"No counts file (running_counts_*.jsonl) found in {output_dir}"
        )
    
    # Use the first (and typically only) files found
    progress_file = progress_files[0]
    counts_file = counts_files[0]
    
    results = {}
    
    # Generate progress plot
    progress_path = plot_progress_file(
        progress_file=progress_file,
        show=show,
        error_bars=error_bars,
        counts_file=counts_file,
    )
    results['progress'] = progress_path
    print(f"Progress plot saved to: {progress_path}")
    
    # Generate convergence plot
    convergence_path = plot_convergence_file(
        progress_file=progress_file,
        show=show,
        error_bars=convergence_error_bars,
        counts_file=counts_file,
    )
    results['convergence'] = convergence_path
    print(f"Convergence plot saved to: {convergence_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m pauli_distribution.plot_results <output_dir>")
        print("\nExample:")
        print("  python -m pauli_distribution.plot_results ./trial_output_dir")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    plot_simulation_results(output_dir)
