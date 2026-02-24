#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

from adaptive_core.calibration import calibrate_thresholds_from_npz
from adaptive_core.density import compute_density_score, choose_transform, load_thresholds
from adaptive_core.transforms import encode_events
from adaptive_core.selection import apply_selection_method
from adaptive_core.inverse import decode_events


def _load_gpu_functions():
    from adaptive_core.transforms_cuda import encode_events_gpu, decode_events_gpu
    return encode_events_gpu, decode_events_gpu


def _events_from_npz(path: Path) -> np.ndarray:
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
        if events[:, 0].size > 0 and np.max(events[:, 0]) > 10000:
            events[:, 0] = events[:, 0] / 1e6
        if events[:, 3].size > 0 and np.min(events[:, 3]) >= 0 and np.max(events[:, 3]) <= 1:
            events[:, 3] = 2 * events[:, 3] - 1
        return events

    raise ValueError(f"Unsupported npz format in {path}")


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass
class EvalResult:
    m: int
    keep_ratio: float
    t_resolution: float
    threshold_method: str
    threshold_value: float | None
    mean_recon_ratio: float
    mean_nnz_ratio: float
    mean_runtime_s: float
    success_rate: float
    score: float


def _score_candidate(
    recon_ratio: float,
    nnz_ratio: float,
    runtime_s: float,
    target_recon_ratio: float,
) -> float:
    # 1. Reconstruction target proximity (higher better)
    recon_term = max(0.0, 1.0 - abs(recon_ratio - target_recon_ratio) / max(target_recon_ratio, 1e-8))
    # 2. Compression preference (lower nnz ratio is better)
    comp_term = 1.0 - np.clip(nnz_ratio, 0.0, 1.0)
    # 3. Runtime preference (smaller is better)
    speed_term = 1.0 / (1.0 + runtime_s)
    return 0.55 * recon_term + 0.30 * comp_term + 0.15 * speed_term


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune thresholds + offline parameters for a new event dataset and write a txt report."
    )
    parser.add_argument("--input", required=True, help="Dataset root with .npz files (e.g., ../../gt)")
    parser.add_argument(
        "--thresholds-output",
        default=str(THIS_DIR.parent / "adaptive_core" / "config" / "adaptive_thresholds.json"),
        help="Where calibrated thresholds JSON is stored",
    )
    parser.add_argument(
        "--report-output",
        default=str(THIS_DIR / "tuning_report.txt"),
        help="Text report output path",
    )
    parser.add_argument(
        "--params-output",
        default=str(THIS_DIR / "recommended_params.json"),
        help="Machine-readable recommended params output path",
    )

    # Calibration knobs
    parser.add_argument("--cal-max-files", type=int, default=30)
    parser.add_argument("--cal-samples-per-file", type=int, default=3)
    parser.add_argument("--cal-sparse-percentile", type=float, default=25.0)
    parser.add_argument("--cal-dense-percentile", type=float, default=75.0)

    # Evaluation sweep knobs
    parser.add_argument("--eval-max-files", type=int, default=8)
    parser.add_argument("--max-events-per-file", type=int, default=50000)
    parser.add_argument("--M-list", default="8,16")
    parser.add_argument("--keep-ratio-list", default="0.2,0.3,0.5")
    parser.add_argument("--t-resolution-list", default="0.01,0.02,0.05")
    parser.add_argument(
        "--threshold-mode-list",
        default="adaptive,percentile",
        help="Comma-separated: adaptive,fixed,percentile",
    )
    parser.add_argument(
        "--percentile-value-list",
        default="95",
        help="Used when threshold mode includes percentile",
    )
    parser.add_argument(
        "--fixed-threshold-list",
        default="0.1",
        help="Used when threshold mode includes fixed",
    )
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--target-recon-ratio", type=float, default=0.05)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA-accelerated encode/decode (requires cupy-cuda12x)",
    )

    args = parser.parse_args()

    if args.gpu:
        encode_fn, decode_fn = _load_gpu_functions()
        import cupy as cp
        cp.cuda.Device(0).use()
        cp.zeros(1)
        cp.cuda.Stream.null.synchronize()
        print("backend: GPU (CuPy)")
    else:
        encode_fn, decode_fn = encode_events, decode_events
        print("backend: CPU (NumPy)")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # 1) Calibrate thresholds from dataset
    cal_result = calibrate_thresholds_from_npz(
        input_dir=input_path,
        output_json=args.thresholds_output,
        max_files=args.cal_max_files,
        samples_per_file=args.cal_samples_per_file,
        sparse_percentile=args.cal_sparse_percentile,
        dense_percentile=args.cal_dense_percentile,
    )
    thresholds = load_thresholds(args.thresholds_output)

    # 2) Gather evaluation files
    eval_files = sorted(input_path.rglob("*.npz"))[: args.eval_max_files]
    if not eval_files:
        raise RuntimeError("No evaluation .npz files found")

    m_list = _parse_int_list(args.M_list)
    keep_ratios = _parse_float_list(args.keep_ratio_list)
    t_resolutions = _parse_float_list(args.t_resolution_list)
    threshold_modes = [x.strip() for x in args.threshold_mode_list.split(",") if x.strip()]
    percentile_vals = _parse_float_list(args.percentile_value_list)
    fixed_vals = _parse_float_list(args.fixed_threshold_list)

    # 3) Evaluate grid
    results: list[EvalResult] = []
    failures: list[str] = []

    for m in m_list:
        for keep_ratio in keep_ratios:
            for t_res in t_resolutions:
                for threshold_mode in threshold_modes:
                    if threshold_mode == "adaptive":
                        threshold_values: list[float | None] = [None]
                    elif threshold_mode == "percentile":
                        threshold_values = percentile_vals
                    elif threshold_mode == "fixed":
                        threshold_values = fixed_vals
                    else:
                        failures.append(f"Unknown threshold mode skipped: {threshold_mode}")
                        continue

                    for threshold_value in threshold_values:
                        recon_ratios = []
                        nnz_ratios = []
                        runtimes = []
                        ok = 0

                        for f in eval_files:
                            try:
                                events = _events_from_npz(f)
                                if len(events) == 0:
                                    continue

                                if len(events) > args.max_events_per_file:
                                    rng = np.random.default_rng(42)
                                    idx = rng.choice(len(events), args.max_events_per_file, replace=False)
                                    events = events[idx]

                                density_info = compute_density_score(events)
                                selected = choose_transform(
                                    density_info["density_score"],
                                    thresholds["sparse_threshold"],
                                    thresholds["dense_threshold"],
                                )

                                t0 = time.perf_counter()
                                volume, scales_or_freqs = encode_fn(
                                    events=events,
                                    method=selected,
                                    h=args.height,
                                    w=args.width,
                                    m=m,
                                )
                                sel_method = "magnitude" if selected in {"DTFT", "DWT"} else "frequency_low"
                                compressed, _, _ = apply_selection_method(
                                    volume, sel_method, keep_ratio=keep_ratio
                                )

                                t_duration = float(np.max(events[:, 0]) - np.min(events[:, 0]))
                                t_start = float(np.min(events[:, 0]))
                                reconstructed_events, _, _ = decode_fn(
                                    coefficient_volume=compressed,
                                    method=selected,
                                    scales_or_freqs=scales_or_freqs,
                                    t_duration=t_duration,
                                    t_resolution=t_res,
                                    threshold_method=threshold_mode,
                                    threshold_value=threshold_value,
                                    polarity_method="sign",
                                    t_start=t_start,
                                )
                                if args.gpu:
                                    cp.cuda.Stream.null.synchronize()
                                dt = time.perf_counter() - t0

                                recon_ratio = len(reconstructed_events) / max(len(events), 1)
                                nnz_ratio = np.count_nonzero(compressed) / max(np.count_nonzero(volume), 1)

                                recon_ratios.append(float(recon_ratio))
                                nnz_ratios.append(float(nnz_ratio))
                                runtimes.append(float(dt))
                                ok += 1
                            except Exception as exc:
                                failures.append(
                                    f"{f.name} failed for M={m}, keep={keep_ratio}, t_res={t_res}, "
                                    f"th_mode={threshold_mode}, th_val={threshold_value}: {exc}"
                                )

                        if ok == 0:
                            continue

                        mean_recon = float(np.mean(recon_ratios))
                        mean_nnz = float(np.mean(nnz_ratios))
                        mean_rt = float(np.mean(runtimes))
                        success_rate = ok / len(eval_files)
                        score = _score_candidate(
                            recon_ratio=mean_recon,
                            nnz_ratio=mean_nnz,
                            runtime_s=mean_rt,
                            target_recon_ratio=args.target_recon_ratio,
                        )

                        results.append(
                            EvalResult(
                                m=m,
                                keep_ratio=keep_ratio,
                                t_resolution=t_res,
                                threshold_method=threshold_mode,
                                threshold_value=threshold_value,
                                mean_recon_ratio=mean_recon,
                                mean_nnz_ratio=mean_nnz,
                                mean_runtime_s=mean_rt,
                                success_rate=success_rate,
                                score=score,
                            )
                        )

    if not results:
        raise RuntimeError("No successful tuning result. Check dataset and sweep ranges.")

    results.sort(key=lambda x: x.score, reverse=True)
    best = results[0]

    # 4) Write outputs
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    params_path = Path(args.params_output)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    with params_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "thresholds_json": str(Path(args.thresholds_output)),
                "thresholds": thresholds,
                "recommended_params": {
                    "M": best.m,
                    "keep_ratio": best.keep_ratio,
                    "t_resolution": best.t_resolution,
                    "threshold_method": best.threshold_method,
                    "threshold_value": best.threshold_value,
                },
                "target_recon_ratio": args.target_recon_ratio,
                "evaluation_files": [str(p) for p in eval_files],
            },
            f,
            indent=2,
        )

    with report_path.open("w", encoding="utf-8") as f:
        f.write("ADAPTIVE CORE DATASET TUNING REPORT\n")
        f.write("===================================\n\n")
        f.write(f"Dataset: {input_path}\n")
        f.write(f"Calibration output JSON: {args.thresholds_output}\n")
        f.write(f"Sparse threshold: {thresholds['sparse_threshold']:.6f}\n")
        f.write(f"Dense threshold:  {thresholds['dense_threshold']:.6f}\n")
        f.write(f"Calibration scores: {cal_result['calibration_stats']['n_scores']}\n\n")

        f.write("Evaluation setup\n")
        f.write("----------------\n")
        f.write(f"Eval files used: {len(eval_files)}\n")
        f.write(f"M grid: {m_list}\n")
        f.write(f"keep_ratio grid: {keep_ratios}\n")
        f.write(f"t_resolution grid: {t_resolutions}\n")
        f.write(f"threshold mode grid: {threshold_modes}\n")
        f.write(f"target recon ratio: {args.target_recon_ratio}\n\n")

        f.write("Top 10 candidates\n")
        f.write("-----------------\n")
        for i, r in enumerate(results[:10], start=1):
            f.write(
                f"{i:02d}. score={r.score:.4f} | M={r.m}, keep={r.keep_ratio}, "
                f"t_res={r.t_resolution}, th={r.threshold_method}"
            )
            if r.threshold_value is not None:
                f.write(f"({r.threshold_value})")
            f.write(
                f" | recon={r.mean_recon_ratio:.4f}, nnz={r.mean_nnz_ratio:.4f}, "
                f"runtime={r.mean_runtime_s:.4f}s, success={r.success_rate:.2f}\n"
            )

        f.write("\nRecommended parameters\n")
        f.write("----------------------\n")
        f.write(f"M={best.m}\n")
        f.write(f"keep_ratio={best.keep_ratio}\n")
        f.write(f"t_resolution={best.t_resolution}\n")
        f.write(f"threshold_method={best.threshold_method}\n")
        f.write(f"threshold_value={best.threshold_value}\n")
        f.write(f"score={best.score:.6f}\n")

        if failures:
            f.write("\nFailures / warnings\n")
            f.write("-------------------\n")
            for line in failures[:100]:
                f.write(f"- {line}\n")
            if len(failures) > 100:
                f.write(f"... ({len(failures) - 100} more)\n")

    print("Tuning complete")
    print(f"Thresholds JSON: {args.thresholds_output}")
    print(f"Report TXT:      {report_path}")
    print(f"Params JSON:     {params_path}")
    print(
        "Recommended: "
        f"M={best.m}, keep_ratio={best.keep_ratio}, t_resolution={best.t_resolution}, "
        f"threshold_method={best.threshold_method}, threshold_value={best.threshold_value}"
    )


if __name__ == "__main__":
    main()
