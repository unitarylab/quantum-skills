"""Small example using only the implementation bundled beside this file."""
import argparse

from quantum_backtracking_implementation import main as implementation_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    forwarded = ["--execute"] if args.execute else []
    if args.output_dir is not None:
        forwarded.extend(["--output-dir", args.output_dir])
    implementation_main(forwarded)


if __name__ == "__main__":
    main()
