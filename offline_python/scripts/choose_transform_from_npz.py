#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

from adaptive_core.density import compute_density_score, load_thresholds, choose_transform
from adaptive_core.transforms import encode_events
from adaptive_core.selection import apply_selection_method
from adaptive_core.inverse import decode_events


def events_from_npz(path: Path) -> np.ndarray:
    data = np.load(str(path))
    if all(k in data for k in ("x", "y", "t", "p")):
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        t = data["t"].astype(np.float64)
        p = data["p"].astype(np.float64)
        if t.size > 0 and np.max(t) > 10000:
            t = t / 1e6
        if p.size > 0 and np.min(p) >= 0 and np.max(p) <= 1:
            p = 2 * p - 1
        return np.column_stack([t, x, y, p])

    if "events" in data:
        events = data["events"][:, :4].astype(np.float64)
        if events.shape[1] >= 1 and events[:, 0].size > 0 and np.max(events[:, 0]) > 10000:
            events[:, 0] = events[:, 0] / 1e6
        return events

    raise ValueError("Unsupported npz format; expected x,y,t,p or events")


def main():
    parser = argparse.ArgumentParser(description="Pick transform for one NPZ file and optionally encode")
    parser.add_argument("--file", required=True, help="Input npz file")
    parser.add_argument(
        "--thresholds",
        default=str(THIS_DIR.parent / "adaptive_core" / "config" / "adaptive_thresholds.json"),
        help="Threshold JSON path",
    )
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.3)
    parser.add_argument("--decode", action="store_true", help="Run inverse reconstruction from compressed coefficients")
    parser.add_argument("--t-resolution", type=float, default=0.02, help="Temporal resolution for decode")
    parser.add_argument(
        "--threshold-method",
        choices=["adaptive", "fixed", "percentile"],
        default="adaptive",
        help="Thresholding strategy for inverse event extraction",
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=None,
        help="Threshold parameter for fixed/percentile modes",
    )
    parser.add_argument(
        "--polarity-method",
        choices=["sign", "magnitude", "positive"],
        default="sign",
        help="Polarity extraction mode during inverse event extraction",
    )

    args = parser.parse_args()

    events = events_from_npz(Path(args.file))
    info = compute_density_score(events)

    thresholds = load_thresholds(args.thresholds)
    selected = choose_transform(
        info["density_score"],
        thresholds["sparse_threshold"],
        thresholds["dense_threshold"],
    )

    volume, scales_or_freqs = encode_events(events, selected, h=args.height, w=args.width, m=args.M)
    method = "magnitude" if selected in {"DTFT", "DWT"} else "frequency_low"
    compressed, _, meta = apply_selection_method(volume, method, keep_ratio=args.keep_ratio)

    print(f"events:           {len(events)}")
    print(f"density_score:    {info['density_score']:.6f}")
    print(f"thresholds:       sparse={thresholds['sparse_threshold']:.6f}, dense={thresholds['dense_threshold']:.6f}")
    print(f"selected:         {selected}")
    print(f"selection_method: {method}")
    print(f"nonzero_before:   {int(np.count_nonzero(volume))}")
    print(f"nonzero_after:    {int(np.count_nonzero(compressed))}")
    print(f"selection_info:   {meta}")

    if args.decode:
        t_duration = float(np.max(events[:, 0]) - np.min(events[:, 0])) if len(events) > 0 else 0.0
        t_start = float(np.min(events[:, 0])) if len(events) > 0 else 0.0
        reconstructed_events, _, _ = decode_events(
            coefficient_volume=compressed,
            method=selected,
            scales_or_freqs=scales_or_freqs,
            t_duration=t_duration,
            t_resolution=args.t_resolution,
            threshold_method=args.threshold_method,
            threshold_value=args.threshold_value,
            polarity_method=args.polarity_method,
            t_start=t_start,
        )

        print(f"decode_enabled:   true")
        print(f"reconstructed_N:  {len(reconstructed_events)}")
        if len(reconstructed_events) > 0:
            print(
                f"recon_t_range:    [{reconstructed_events[:, 0].min():.6f}, {reconstructed_events[:, 0].max():.6f}]"
            )


if __name__ == "__main__":
    main()
