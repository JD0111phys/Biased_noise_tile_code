# This is the simulation for the code, platform, and compilation (cnot, ideal, cz) specified in the file name.
# example.py
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List
from pauli_distribution import (
    error_propagation_simulation,
    save_running_counts)


gate_tile_middle = [
("H", [249, 853, 609, 501, 861, 381, 613, 369, 261, 865, 729, 621, 373, 625, 741, 489, 849, 385]),
("CZ", [246, 249, 366, 369, 486, 489, 606, 609, 726, 729, 846, 849, 370, 373, 610, 613, 850, 853, 258, 261, 378, 381, 498, 501, 618, 621, 738, 741, 858, 861]),
("CX", [250, 253, 490, 493, 730, 733, 254, 257, 374, 377, 494, 497, 614, 617, 734, 737, 854, 857]),
("CX", [248, 489, 368, 609, 488, 729, 608, 849, 372, 613, 612, 853, 260, 501, 380, 621, 500, 741, 620, 861, 384, 625, 624, 865]),
("CZ", [252, 493, 492, 733, 256, 497, 376, 617, 496, 737, 616, 857, 264, 505, 504, 745, 268, 509, 388, 629, 508, 749, 628, 869]),
("CX", [6, 307, 126, 427, 246, 547, 366, 667, 486, 787, 606, 907, 10, 311, 130, 431, 250, 551, 370, 671, 490, 791, 610, 911, 14, 315, 134, 435, 254, 555, 374, 675, 494, 795, 614, 915, 18, 319, 138, 439, 258, 559, 378, 679, 498, 799, 618, 919]),
("CY", [248, 307, 368, 427, 488, 547, 608, 667, 252, 311, 372, 431, 492, 551, 612, 671, 256, 315, 376, 435, 496, 555, 616, 675, 260, 319, 380, 439, 500, 559, 620, 679, 264, 323, 384, 443, 504, 563, 624, 683, 268, 327, 388, 447, 508, 567, 628, 687]),
("CZ", [130, 261, 250, 381, 370, 501, 490, 621, 610, 741, 730, 861, 254, 385, 494, 625, 734, 865]),
("CX", [126, 257, 246, 377, 366, 497, 486, 617, 606, 737, 726, 857, 134, 265, 374, 505, 614, 745, 138, 269, 258, 389, 378, 509, 498, 629, 618, 749, 738, 869]),
("CX", [244, 249, 364, 369, 484, 489, 604, 609, 368, 373, 608, 613, 256, 261, 376, 381, 496, 501, 616, 621, 380, 385, 620, 625]),
("CZ", [248, 253, 488, 493, 252, 257, 372, 377, 492, 497, 612, 617, 260, 265, 500, 505, 264, 269, 384, 389, 504, 509, 624, 629]),
("CX", [6, 311, 126, 431, 246, 551, 366, 671, 486, 791, 606, 911, 10, 315, 130, 435, 250, 555, 370, 675, 490, 795, 610, 915, 14, 319, 134, 439, 254, 559, 374, 679, 494, 799, 614, 919, 18, 323, 138, 443, 258, 563, 378, 683, 498, 803, 618, 923]),
("CY", [248, 427, 368, 547, 488, 667, 608, 787, 252, 431, 372, 551, 492, 671, 612, 791, 256, 435, 376, 555, 496, 675, 616, 795, 260, 439, 380, 559, 500, 679, 620, 799, 264, 443, 384, 563, 504, 683, 624, 803, 268, 447, 388, 567, 508, 687, 628, 807]),
("CZ", [10, 261, 130, 381, 250, 501, 370, 621, 490, 741, 610, 861, 134, 385, 374, 625, 614, 865]),
("CX", [6, 257, 126, 377, 246, 497, 366, 617, 486, 737, 606, 857, 14, 265, 254, 505, 494, 745, 18, 269, 138, 389, 258, 509, 378, 629, 498, 749, 618, 869]),
("CX", [240, 249, 360, 369, 480, 489, 600, 609, 364, 373, 604, 613, 252, 261, 372, 381, 492, 501, 612, 621, 376, 385, 616, 625]),
("CZ", [244, 253, 484, 493, 248, 257, 368, 377, 488, 497, 608, 617, 256, 265, 496, 505, 260, 269, 380, 389, 500, 509, 620, 629]),
("CX", [246, 315, 366, 435, 486, 555, 606, 675, 726, 795, 846, 915, 250, 319, 370, 439, 490, 559, 610, 679, 730, 799, 850, 919, 254, 323, 374, 443, 494, 563, 614, 683, 734, 803, 854, 923, 258, 327, 378, 447, 498, 567, 618, 687, 738, 807, 858, 927]),
("CY", [240, 547, 360, 667, 480, 787, 600, 907, 244, 551, 364, 671, 484, 791, 604, 911, 248, 555, 368, 675, 488, 795, 608, 915, 252, 559, 372, 679, 492, 799, 612, 919, 256, 563, 376, 683, 496, 803, 616, 923, 260, 567, 380, 687, 500, 807, 620, 927]),
]

ancilla_middle_tile = [
    6, 10, 14, 18,
    126, 130, 134, 138,
    240, 244, 246, 248, 250, 252, 254, 256, 258, 260, 264, 268,
    360, 364, 366, 368, 370, 372, 374, 376, 378, 380, 384, 388,
    480, 484, 486, 488, 490, 492, 494, 496, 498, 500, 504, 508,
    600, 604, 606, 608, 610, 612, 614, 616, 618, 620, 624, 628,
    726, 730, 734, 738,
    846, 850, 854, 858
]

qubits_middle_tile = 928

used_qubits_middle: List[int] = []
for _, qubits in gate_tile_middle:
    used_qubits_middle.extend(qubits)
keep_qubits_middle = list(sorted(set(used_qubits_middle)))
chosen_seed = 2100000
platform = "ideal"
bias_sys = 10000.0
prob = 0.003
samples_per_iteration = 10000  # Increased for better convergence
total_samples = 2_000_000  # Set a maximum total sample limit for safety


# Create new directory for all output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"trial_middle_{platform}_bias_sys_{bias_sys}_p_{prob}_chosen_seed_{chosen_seed}_{timestamp}")
output_dir.mkdir(exist_ok=True)
os.chdir(output_dir)


# Start timing the simulation
start_time = time.time()
print(f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")


# Run the simulation
progress_file,counts_file = error_propagation_simulation(
    keep_qubits=keep_qubits_middle,
    ancilla=ancilla_middle_tile,
    p_param=prob,
    system_bias=bias_sys,
    qubit_platform=platform,
    gate_sequence=gate_tile_middle,
    samples_per_iteration=samples_per_iteration,
    total_samples=total_samples,  # Set a maximum total sample limit for safety
    chosen_seed=chosen_seed,
    timestamp=timestamp
)


# Calculate and display execution time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Save timing information to files
with open(progress_file, "a") as f:
    f.write(f"# Execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
    f.write(f"# Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Also save timing to the running counts file
save_running_counts({'timing_seconds': elapsed_time, 'completed_at': datetime.now().isoformat()}, counts_file, append=True, seed=None) #type: ignore
