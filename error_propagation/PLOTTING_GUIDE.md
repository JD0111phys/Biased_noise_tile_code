# Visualization Guide - plot_progress_file

The `plot_progress_file.py` module generates publication-quality convergence plots from simulation result files.

## Quick Start

### Command Line Interface (Simplest)

After running a simulation and having an output directory with `effective_probs_*.txt` and optionally `running_counts_*.jsonl`:

```bash
cd error_propagation

# Basic plot with error bars
python -m pauli_distribution.plot_progress_file \
  --progress-file /path/to/effective_probs_ideal_20260401_024757.txt \
  --counts-file /path/to/running_counts_ideal_20260401_024757.jsonl \
  --error-bars

# With advanced diagnostics (Z_k tail analysis + bias plateau detection)
python -m pauli_distribution.plot_progress_file \
  --progress-file /path/to/effective_probs_ideal_20260401_024757.txt \
  --counts-file /path/to/running_counts_ideal_20260401_024757.jsonl \
  --error-bars \
  --zk-tail-start 250 \
  --plateau-tail-start 250 \
  --plateau-blocks 5

# With manual significant figures overlay
python -m pauli_distribution.plot_progress_file \
  --progress-file /path/to/effective_probs_ideal_20260401_024757.txt \
  --counts-file /path/to/running_counts_ideal_20260401_024757.jsonl \
  --error-bars \
  --manual-sigfig 3
```

### Python API

For more control, call the function directly:

```python
from pathlib import Path
from pauli_distribution import plot_progress_file

progress_file = Path("trial_output_dir/effective_probs_ideal_20260401_024757.txt")
counts_file = Path("trial_output_dir/running_counts_ideal_20260401_024757.jsonl")

plot_progress_file(
    progress_file=progress_file,
    counts_file=counts_file,
    error_bars=True,
    zk_tail_start=250,
    show_zk_hist=True,
    plateau_tail_start=250,
    plateau_blocks=5,
)
```

## Output Files

The utility saves an SVG file to the same directory as the input progress file:

- `effective_probs_*_plots.svg` - Main convergence plot with all subplots

The SVG filename includes flags that were used (e.g., `_errorBars_zkTail250.svg`).

## Plot Layout

The 2×4 subplot grid shows:

**Row 1 (Probabilities):**
- X probability vs iteration/samples
- Y probability vs iteration/samples  
- Z probability vs iteration/samples

**Row 2 (Bias & Diagnostics):**
- Bias vs iteration/samples (with optional plateau overlay)
- Relative precision of bias (with optional linear scale)
- Z_k (signed) metric vs iteration/samples (with optional histogram)
- Z_k (absolute) metric vs iteration/samples

## Command-Line Options

### Required Arguments
- `--progress-file` - Path to `effective_probs_*.txt` file

### Optional Input/Output
- `--counts-file` - Path to `running_counts_*.jsonl` for uncertainty calculations
- `--output` - Output image path (default: auto-named .svg)
- `--show` - Display plot window in addition to saving

### Data Filtering
- `--iter-start` - Minimum iteration to include (inclusive)
- `--iter-end` - Maximum iteration to include (inclusive)
- `--x-axis` - Use "iteration" (default) or "samples" on x-axis

### Scaling & Display
- `--smooth-window` - Centered moving-average window size (default: 1, no smoothing)
- `--bias-log-scale` - Use logarithmic y-axis for bias subplot
- `--rel-prec-linear` - Use linear y-axis for relative precision plot (default: log scale)
- `--error-z` - Z-score multiplier for uncertainty bands (default: 1.96 for 95% CI)

### Error Bars & Uncertainty
- `--error-bars` - Plot confidence bands (requires `--counts-file`)

### Significant Figures Overlay
- `--manual-sigfig N` - Overlay line rounded to N significant figures
- `--no-sigfig-overlay` - Disable significant-figure overlay

### Z_k Tail Diagnostics
These options enable statistical analysis of normalized bias differences in the convergence tail:

- `--zk-tail-start` - Iteration to start tail analysis (enables histogram and statistics)
- `--no-zk-summary` - Disable console output of Z_k statistics
- `--no-zk-hist` - Disable Z_k histogram subplot
- `--no-zk-rescale-tail` - Disable Option B empirical tail std rescaling

Z_k tail statistics include:
- Mean and standard deviation of Z_k in tail region
- Lag-1 autocorrelation
- Fraction of values exceeding ±2σ and ±3σ thresholds
- Sign balance metrics

### Bias Plateau Analysis
Detects convergence plateaus via block-median analysis:

- `--plateau-tail-start` - Iteration to start plateau analysis (default: same as `--zk-tail-start`)
- `--plateau-blocks` - Number of equal-sized blocks for analysis (default: 5)
- `--no-plateau-summary` - Disable console output of plateau statistics
- `--no-plateau-overlay` - Disable overlay of plateau block medians on bias plot

Plateau statistics include:
- Block-wise median bias values
- Spread (max - min) across blocks
- Endpoint drift and linear trend analysis

### Presets
- `--preset uncertainty` - Automatically enables `--error-bars` if counts file exists
- `--preset quality` - Enables all diagnostics (equivalent to combining multiple analysis flags)

## Common Usage Patterns

### Batch Processing Multiple Trials

Use the provided `batch_plot_data5.ps1` script (PowerShell):

```powershell
# From project root
.\batch_plot_data5.ps1
```

Or create a custom script for your data directory:

```bash
#!/bin/bash
for dir in data_5/trial_*/; do
    PROG_FILE=$(ls "$dir"/effective_probs_*.txt | head -1)
    COUNTS_FILE=$(ls "$dir"/running_counts_*.jsonl | head -1)
    python -m pauli_distribution.plot_progress_file \
        --progress-file "$PROG_FILE" \
        --counts-file "$COUNTS_FILE" \
        --error-bars \
        --manual-sigfig 3
done
```

### Diagnostics for Convergence Troubleshooting

```bash
# Analyze tail region starting at iteration 250
python -m pauli_distribution.plot_progress_file \
  --progress-file /path/to/effective_probs_*.txt \
  --counts-file /path/to/running_counts_*.jsonl \
  --error-bars \
  --zk-tail-start 250 \
  --plateau-tail-start 250 \
  --plateau-blocks 10
```

This will:
1. Print Z_k tail statistics (mean, std, autocorrelation, sign patterns)
2. Print bias plateau statistics (median spread, trend analysis)
3. Add histogram and plateau overlay to the plot

### Publication-Quality Plot with Significant Figures

```bash
python -m pauli_distribution.plot_progress_file \
  --progress-file /path/to/effective_probs_*.txt \
  --counts-file /path/to/running_counts_*.jsonl \
  --error-bars \
  --manual-sigfig 3 \
  --smooth-window 5 \
  --rel-prec-linear
```

### Focused Iteration Range

```bash
python -m pauli_distribution.plot_progress_file \
  --progress-file /path/to/effective_probs_*.txt \
  --iter-start 100 \
  --iter-end 500 \
  --zk-tail-start 250
```

## Input File Formats

### Progress File (`effective_probs_*.txt`)

Tab-separated columns (first line is header):
```
Iteration    X_prob    X_std    Y_prob    Y_std    Z_prob    Z_std    Bias    Bias_std    Generated_Samples
0            0.495     ...      0.491     ...      0.500     ...      ...     ...         100000
1            0.498     ...      0.492     ...      0.499     ...      ...     ...         100000
...
```

### Counts File (`running_counts_*.jsonl`)

JSON lines format, one object per iteration:
```json
{"Iteration": 0, "X_count": 49500, "Y_count": 49100, "Z_count": 50000, "Total_samples": 100000}
{"Iteration": 1, "X_count": 49800, "Y_count": 49200, "Z_count": 49900, "Total_samples": 100000}
```

## Understanding the Diagnostics

### Relative Precision (RP)

$$\text{RP} = \frac{\text{SE}_\text{bias}}{|\text{bias}|}$$

Shows how relative the uncertainly is to bias magnitude. Lower is better (bias is more precise relative to its magnitude).

### Z_k (Normalized Bias Difference)

$$Z_k = \frac{\text{bias}_k - \text{bias}_{k-1}}{\text{SE}_{\Delta_k}}$$

Measures normalized step changes in bias. Values close to ±1 indicate consistent uncertainty scaling. Large deviations suggest anomalies or correlations.

### Bias Plateau

Block-median analysis in convergence tail detects plateaus:
- **Spread**: Max - min bias across blocks (convergence quality)
- **Trend**: Linear regression slope (ongoing drift)
- **Endpoint Drift**: Difference between first and last block medians
