import json
from pathlib import Path
import numpy as np


DEFAULT_THRESHOLDS = {
    "sparse_threshold": 0.234,
    "dense_threshold": 0.630,
}


def compute_spatial_density(x_events: np.ndarray, y_events: np.ndarray, spatial_window_size: int = 32) -> float:
    if x_events.size == 0:
        return 0.0

    x_min, x_max = np.min(x_events), np.max(x_events)
    y_min, y_max = np.min(y_events), np.max(y_events)

    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    n_bins_x = max(1, int(x_range / spatial_window_size))
    n_bins_y = max(1, int(y_range / spatial_window_size))

    hist, _, _ = np.histogram2d(
        x_events,
        y_events,
        bins=[n_bins_x, n_bins_y],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    non_zero_bins = np.sum(hist > 0)
    total_bins = n_bins_x * n_bins_y
    occupancy_ratio = non_zero_bins / total_bins if total_bins > 0 else 0.0
    mean_density = np.mean(hist[hist > 0]) if non_zero_bins > 0 else 0.0

    max_possible_density = len(x_events) / non_zero_bins if non_zero_bins > 0 else 1.0
    mean_density_norm = mean_density / max_possible_density

    return float((occupancy_ratio + mean_density_norm) / 2.0)


def compute_spatial_clustering(
    x_events: np.ndarray,
    y_events: np.ndarray,
    sample_points: int = 200,
    sample_pairs: int = 100,
) -> float:
    if x_events.size < 2:
        return 0.0

    rng = np.random.default_rng(42)
    n_sample = min(sample_points, x_events.size)
    idx = rng.choice(x_events.size, n_sample, replace=False)

    x_sample = x_events[idx]
    y_sample = y_events[idx]

    x_norm = (x_sample - np.min(x_sample)) / (np.max(x_sample) - np.min(x_sample) + 1e-8)
    y_norm = (y_sample - np.min(y_sample)) / (np.max(y_sample) - np.min(y_sample) + 1e-8)

    coords = np.column_stack([x_norm, y_norm])
    n_points = len(coords)
    n_pairs = min(sample_pairs, n_points * (n_points - 1) // 2)

    distances = np.empty(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        a, b = rng.choice(n_points, 2, replace=False)
        distances[i] = np.sqrt(((coords[a] - coords[b]) ** 2).sum())

    mean_dist = np.mean(distances) if distances.size > 0 else 1.0
    std_dist = np.std(distances) if distances.size > 0 else 0.0
    return float(1.0 / (1.0 + std_dist / (mean_dist + 1e-8)))


def compute_density_score(
    events: np.ndarray,
    spatial_window_size: int = 32,
) -> dict:
    if events.size == 0:
        return {
            "density_score": 0.0,
            "spatial_density": 0.0,
            "event_rate": 0.0,
            "temporal_variance": 0.0,
            "spatial_clustering": 0.0,
            "n_events": 0,
        }

    if events.ndim != 2 or events.shape[1] < 3:
        raise ValueError("events must be shape (N, 4) or (N, >=3) with [t, x, y, ...]")

    t_events = events[:, 0].astype(np.float64)
    x_events = events[:, 1].astype(np.float64)
    y_events = events[:, 2].astype(np.float64)

    spatial_density = compute_spatial_density(x_events, y_events, spatial_window_size)

    t_duration = np.max(t_events) - np.min(t_events) if t_events.size > 1 else 1e-6
    t_duration = max(t_duration, 1e-6)
    event_rate = len(events) / t_duration
    temporal_variance = np.var(t_events) if t_events.size > 1 else 0.0

    spatial_clustering = compute_spatial_clustering(x_events, y_events)

    spatial_norm = np.clip(spatial_density / 100.0, 0, 1)
    rate_norm = np.clip(event_rate / 10000.0, 0, 1)
    cluster_norm = np.clip(spatial_clustering, 0, 1)
    temporal_norm = np.clip(temporal_variance / 1000.0, 0, 1)

    density_score = (
        0.4 * spatial_norm
        + 0.3 * rate_norm
        + 0.2 * cluster_norm
        + 0.1 * temporal_norm
    )

    return {
        "density_score": float(density_score),
        "spatial_density": float(spatial_density),
        "event_rate": float(event_rate),
        "temporal_variance": float(temporal_variance),
        "spatial_clustering": float(spatial_clustering),
        "n_events": int(len(events)),
    }


def choose_transform(density_score: float, sparse_threshold: float, dense_threshold: float) -> str:
    if density_score <= sparse_threshold:
        return "DWT"
    if density_score >= dense_threshold:
        return "DCT"
    return "DTFT"


def load_thresholds(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return DEFAULT_THRESHOLDS.copy()

    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "density_thresholds" in cfg:
        return {
            "sparse_threshold": float(cfg["density_thresholds"]["sparse_threshold"]),
            "dense_threshold": float(cfg["density_thresholds"]["dense_threshold"]),
        }

    return {
        "sparse_threshold": float(cfg.get("sparse_threshold", DEFAULT_THRESHOLDS["sparse_threshold"])),
        "dense_threshold": float(cfg.get("dense_threshold", DEFAULT_THRESHOLDS["dense_threshold"])),
    }


def save_thresholds(path: str | Path, sparse_threshold: float, dense_threshold: float, extra: dict | None = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "density_thresholds": {
            "sparse_threshold": float(sparse_threshold),
            "dense_threshold": float(dense_threshold),
        }
    }
    if extra:
        payload.update(extra)

    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
