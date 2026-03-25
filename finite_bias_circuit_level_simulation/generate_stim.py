import argparse
import circuit_level_css

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=6, help="x distance")
    parser.add_argument("--m", type=int, default=6, help="z distance")
    parser.add_argument("--p", type=float, default=0.005, help="error rate")
    parser.add_argument("--bias", type=float, default=10000.0, help="bias")
    parser.add_argument("--rounds", type=int, default=8, help="number of rounds")
    parser.add_argument("--out", type=str, default="circuit_8_rounds.stim", help="output file")
    args = parser.parse_args()

    circuit = circuit_level_css.generate_circuit(
        rounds=args.rounds,
        x_distance=args.l,
        z_distance=args.m,
        before_round_data_depolarization=args.p,
        before_measure_flip_probability=args.p,
        after_reset_flip_probability=args.p,
        after_clifford_depolarization=args.p,
        after_single_clifford_probability=args.p,
        bias=args.bias,
    )

    circuit.to_file(args.out)
    print(f"Successfully generated {args.out} for l={args.l}, m={args.m}, rounds={args.rounds}, p={args.p}, bias={args.bias}")

if __name__ == "__main__":
    main()
