#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

from adaptive_core.calibration import calibrate_thresholds_from_npz


def main():
    parser = argparse.ArgumentParser(description="Calibrate adaptive thresholds from NPZ event files")
    parser.add_argument("--input", required=True, help="Directory containing .npz event files")
    parser.add_argument(
        "--output",
        default=str(THIS_DIR.parent / "adaptive_core" / "config" / "adaptive_thresholds.json"),
        help="Output JSON path",
    )
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--samples-per-file", type=int, default=3)
    parser.add_argument("--sparse-percentile", type=float, default=25.0)
    parser.add_argument("--dense-percentile", type=float, default=75.0)

    args = parser.parse_args()

    result = calibrate_thresholds_from_npz(
        input_dir=args.input,
        output_json=args.output,
        max_files=args.max_files,
        samples_per_file=args.samples_per_file,
        sparse_percentile=args.sparse_percentile,
        dense_percentile=args.dense_percentile,
    )

    print("Calibration complete")
    print(f"sparse_threshold: {result['sparse_threshold']:.6f}")
    print(f"dense_threshold:  {result['dense_threshold']:.6f}")
    print(f"scores:           {result['calibration_stats']['n_scores']}")
    print(f"output:           {args.output}")


if __name__ == "__main__":
    main()
