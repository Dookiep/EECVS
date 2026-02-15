import numpy as np

from .transforms import (
    event_driven_dct_decode,
    event_driven_dtft_decode,
    event_driven_dwt_decode,
)


def reconstruct_temporal_volume(
    coefficient_volume: np.ndarray,
    method: str,
    scales_or_freqs: np.ndarray,
    t_duration: float,
    t_resolution: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    if t_resolution <= 0:
        raise ValueError("t_resolution must be > 0")

    if t_duration <= 0:
        time_axis = np.array([0.0], dtype=np.float64)
    else:
        time_axis = np.arange(0.0, t_duration, t_resolution, dtype=np.float64)
        if time_axis.size == 0:
            time_axis = np.array([0.0], dtype=np.float64)

    method_u = method.upper()
    if method_u == "DCT":
        temporal_volume = event_driven_dct_decode(coefficient_volume, time_axis, scales_or_freqs)
    elif method_u in {"DTFT", "CES"}:
        temporal_volume = event_driven_dtft_decode(coefficient_volume, time_axis, scales_or_freqs)
    elif method_u == "DWT":
        temporal_volume = event_driven_dwt_decode(coefficient_volume, time_axis, scales_or_freqs)
    else:
        raise ValueError(f"Unsupported method for inverse reconstruction: {method}")

    return temporal_volume, time_axis


def temporal_volume_to_events(
    temporal_volume: np.ndarray,
    time_axis: np.ndarray,
    threshold_method: str = "adaptive",
    threshold_value: float | None = None,
    polarity_method: str = "sign",
) -> np.ndarray:
    if temporal_volume.ndim != 3:
        raise ValueError("temporal_volume must be shape (H, W, T)")

    if time_axis.size != temporal_volume.shape[2]:
        raise ValueError("time_axis length must match temporal_volume.shape[2]")

    if temporal_volume.size == 0 or time_axis.size == 0:
        return np.empty((0, 4), dtype=np.float64)

    if threshold_method == "adaptive":
        mean_val = float(np.mean(temporal_volume))
        std_val = float(np.std(temporal_volume))
        threshold_pos = mean_val + 2.0 * std_val
        threshold_neg = mean_val - 2.0 * std_val
    elif threshold_method == "percentile":
        p = 95.0 if threshold_value is None else float(threshold_value)
        threshold_pos = float(np.percentile(temporal_volume, p))
        threshold_neg = float(np.percentile(temporal_volume, 100.0 - p))
    elif threshold_method == "fixed":
        val = 0.1 if threshold_value is None else float(threshold_value)
        threshold_pos = val
        threshold_neg = -val
    else:
        raise ValueError("threshold_method must be one of: adaptive, percentile, fixed")

    events_list: list[list[float]] = []

    for t_idx, t in enumerate(time_axis):
        frame = temporal_volume[:, :, t_idx]

        if polarity_method == "sign":
            pos_y, pos_x = np.where(frame > threshold_pos)
            neg_y, neg_x = np.where(frame < threshold_neg)

            for y, x in zip(pos_y, pos_x):
                events_list.append([float(t), float(x), float(y), 1.0])
            for y, x in zip(neg_y, neg_x):
                events_list.append([float(t), float(x), float(y), -1.0])

        elif polarity_method == "magnitude":
            mag_mask = np.abs(frame) > threshold_pos
            ys, xs = np.where(mag_mask)
            for y, x in zip(ys, xs):
                p = 1.0 if frame[y, x] >= 0 else -1.0
                events_list.append([float(t), float(x), float(y), p])

        elif polarity_method == "positive":
            ys, xs = np.where(frame > threshold_pos)
            for y, x in zip(ys, xs):
                events_list.append([float(t), float(x), float(y), 1.0])

        else:
            raise ValueError("polarity_method must be one of: sign, magnitude, positive")

    if not events_list:
        return np.empty((0, 4), dtype=np.float64)

    return np.array(events_list, dtype=np.float64)


def decode_events(
    coefficient_volume: np.ndarray,
    method: str,
    scales_or_freqs: np.ndarray,
    t_duration: float,
    t_resolution: float = 0.001,
    threshold_method: str = "adaptive",
    threshold_value: float | None = None,
    polarity_method: str = "sign",
    t_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temporal_volume, time_axis = reconstruct_temporal_volume(
        coefficient_volume=coefficient_volume,
        method=method,
        scales_or_freqs=scales_or_freqs,
        t_duration=t_duration,
        t_resolution=t_resolution,
    )

    events = temporal_volume_to_events(
        temporal_volume=temporal_volume,
        time_axis=time_axis,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        polarity_method=polarity_method,
    )

    if events.size > 0:
        events[:, 0] += float(t_start)

    return events, temporal_volume, time_axis
