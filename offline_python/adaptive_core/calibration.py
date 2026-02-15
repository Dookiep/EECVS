from pathlib import Path
import numpy as np

from .density import compute_density_score, save_thresholds


def _events_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path))

    if all(k in data for k in ("x", "y", "t", "p")):
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        t = data["t"].astype(np.float64)
        p = data["p"].astype(np.float64)
        if t.size > 0 and np.max(t) > 10000:
            t = t / 1e6

        # Normalize polarity to {-1, +1} if file stores {0, 1}
        if p.size > 0 and np.min(p) >= 0 and np.max(p) <= 1:
            p = 2 * p - 1

        return np.column_stack([t, x, y, p])

    if "events" in data:
        ev = data["events"].astype(np.float64)
        if ev.ndim != 2 or ev.shape[1] < 4:
            raise ValueError(f"Unsupported events array format in {npz_path}")
        ev = ev[:, :4]
        if ev[:, 0].size > 0 and np.max(ev[:, 0]) > 10000:
            ev[:, 0] = ev[:, 0] / 1e6
        return ev

    raise ValueError(f"Unsupported npz format in {npz_path}")


def calibrate_thresholds_from_npz(
    input_dir: str | Path,
    output_json: str | Path,
    max_files: int = 30,
    samples_per_file: int = 3,
    sample_sizes: tuple[int, ...] = (1000, 2000, 3000),
    sparse_percentile: float = 25.0,
    dense_percentile: float = 75.0,
) -> dict:
    input_path = Path(input_dir)
    npz_files = sorted(input_path.rglob("*.npz"))[:max_files]

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {input_dir}")

    rng = np.random.default_rng(42)
    scores: list[float] = []

    for npz_file in npz_files:
        try:
            events = _events_from_npz(npz_file)
        except Exception:
            continue

        if events.size == 0:
            continue

        for _ in range(samples_per_file):
            target_n = int(rng.choice(sample_sizes))
            if len(events) <= target_n:
                sample = events
            else:
                idx = rng.choice(len(events), target_n, replace=False)
                sample = events[idx]

            info = compute_density_score(sample)
            scores.append(info["density_score"])

    if not scores:
        raise RuntimeError("Calibration produced no valid density scores")

    sparse_threshold = float(np.percentile(scores, sparse_percentile))
    dense_threshold = float(np.percentile(scores, dense_percentile))

    stats = {
        "calibration_stats": {
            "n_scores": len(scores),
            "min_density": float(np.min(scores)),
            "max_density": float(np.max(scores)),
            "mean_density": float(np.mean(scores)),
            "std_density": float(np.std(scores)),
            "sparse_percentile": sparse_percentile,
            "dense_percentile": dense_percentile,
        },
        "metadata": {
            "source_dir": str(input_path),
            "max_files": max_files,
            "samples_per_file": samples_per_file,
            "sample_sizes": list(sample_sizes),
        },
    }

    save_thresholds(output_json, sparse_threshold, dense_threshold, extra=stats)

    return {
        "sparse_threshold": sparse_threshold,
        "dense_threshold": dense_threshold,
        **stats,
    }
