#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "density_thresholds": {
        "sparse_threshold": 0.234,
        "dense_threshold": 0.630,
    },
    "selection_logic": {
        "sparse_events": "density_score <= sparse_threshold -> DWT",
        "medium_events": "sparse_threshold < density_score < dense_threshold -> DTFT",
        "dense_events": "density_score >= dense_threshold -> DCT",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Generate default threshold config")
    parser.add_argument(
        "--output",
        default="config/adaptive_thresholds.json",
        help="Output config path",
    )
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
