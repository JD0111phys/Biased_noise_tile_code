# pauli_distribution/__init__.py
from .pauli_strings import (
    get_pauli_string,
    save_running_counts,
    load_running_counts,
    apply_error,
    gate_operation,
    convert_gate_sequence,
    error_propagation_simulation,
    apply_gate_error_channel
)

__all__ = [
    "get_pauli_string",
    "save_running_counts",
    "load_running_counts",
    "apply_error",
    "gate_operation",
    "convert_gate_sequence",
    "error_propagation_simulation",
    "apply_gate_error_channel",
    "plot_progress_file",
    "plot_convergence_file",
    "plot_simulation_results",
]


def __getattr__(name: str):
    """Lazy import of plot functions to avoid module conflicts when running as scripts."""
    if name == "plot_progress_file":
        from .plot_progress_file import plot_progress_file
        return plot_progress_file
    elif name == "plot_convergence_file":
        from .plot_convergence_file import plot_convergence_file
        return plot_convergence_file
    elif name == "plot_simulation_results":
        from .plot_results import plot_simulation_results
        return plot_simulation_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
