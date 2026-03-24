# Biased Noise Tile Code

This repository contains the code used to study tile codes under biased noise
and Clifford deformations.

## Repository structure

- tile_code_and_clifford_deformation  
  Tile code construction and Clifford deformation routines.

- infinite_bias_threshold  
  Simulations for infinite bias threshold.

- finite_bias_circuit_level_simulation  
  Circuit-level simulations under finite bias noise.

- finite_bias_code_capacity_model  
  Code-capacity simulations under finite bias noise.

- error_propagation  
  Tools for analyzing Pauli error propagation in circuits.

- blo_study  
  Scripts for studying basis logical operators.

## Requirements

The code uses Python and requires the following packages:

-Core computation & data structures

numpy

scipy

pandas

-Quantum Simulation

stim

sinter

bposd

ldpc

galois

-Utilities

tqdm

pytest 

## Running simulations

Example:

python finite_bias_circuit_level_simulation/circuit_level_css.py 9 9 --bias 10000

where `l` and `m` are the lattice dimensions and `bias` is the Z-bias parameter of the Pauli noise model.
