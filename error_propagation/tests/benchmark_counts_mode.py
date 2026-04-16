"""Simple benchmark for flattened vs counts-only sampling paths.

Run from this directory:
    python benchmark_counts_mode.py --samples 2000 --repeats 5
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import statistics
import sys
import time

# Ensure local package import works when script is run from tests dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pauli_distribution import get_pauli_string  # noqa: E402


def _flattened_to_counts(values: list[int]) -> dict[int, int]:
    c = Counter(values)
    return {0: int(c.get(0, 0)), 1: int(c.get(1, 0)), 2: int(c.get(2, 0)), 3: int(c.get(3, 0))}


def run_benchmark(samples: int, repeats: int, platform: str) -> None:
    kwargs = {
        "keep_qubits": [0, 1, 2, 3, 4, 5],
        "samples": samples,
        "p": 0.003,
        "system_bias": 1000.0,
        "gate_sequence": [("H", [0, 2, 4]), ("CX", [0, 1, 2, 3, 4, 5]), ("S", [1, 3, 5])],
        "ancilla": [0],
        "qubit_platform": platform,
        "use_compressed_space": True,
    }

    flattened_times: list[float] = []
    counts_times: list[float] = []

    for i in range(repeats):
        seed = 1234 + i

        t0 = time.perf_counter()
        flattened = get_pauli_string(random_seed=seed, return_counts=False, **kwargs)
        t1 = time.perf_counter()
        flattened_times.append(t1 - t0)

        t2 = time.perf_counter()
        counted = get_pauli_string(random_seed=seed, return_counts=True, **kwargs)
        t3 = time.perf_counter()
        counts_times.append(t3 - t2)

        if i == 0:
            # One correctness check per run configuration.
            flat_counts = _flattened_to_counts(flattened)
            if flat_counts != counted:
                raise RuntimeError(
                    f"Mismatch between flattened and counts-only modes. flat={flat_counts}, counts={counted}"
                )

    flat_mean = statistics.mean(flattened_times)
    counts_mean = statistics.mean(counts_times)
    speedup = flat_mean / counts_mean if counts_mean > 0 else float("inf")

    print("Benchmark completed")
    print(f"samples={samples}, repeats={repeats}, platform={platform}")
    print(f"flattened mean:   {flat_mean:.6f} s")
    print(f"counts-only mean: {counts_mean:.6f} s")
    print(f"speedup (flat/counts): {speedup:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark get_pauli_string flattened vs counts-only path")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--platform", type=str, default="superconducting")
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    run_benchmark(samples=args.samples, repeats=args.repeats, platform=args.platform)


if __name__ == "__main__":
    main()
