import numpy as np


def _sanitize_events(events: np.ndarray, h: int, w: int):
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must be shape (N, 4) with [t, x, y, p]")

    t = events[:, 0].astype(np.float64)
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    p = events[:, 3].astype(np.float64)

    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return t[in_bounds], x[in_bounds], y[in_bounds], p[in_bounds]


def sample_frequencies_dct(m: int, max_freq: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, max_freq, size=m).astype(np.float64)


def sample_frequencies_dtft(m: int, sigma: float = 0.5, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=sigma, size=m).astype(np.float64)


def sample_frequencies_dwt(m: int, sigma: float = 0.3, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=sigma, size=m).astype(np.float64)


def event_driven_dct_encode(events: np.ndarray, h: int, w: int, t_window: float, freqs: np.ndarray) -> np.ndarray:
    volume = np.zeros((h, w, 2 * len(freqs)), dtype=np.float64)
    if events.size == 0:
        return volume

    t, x, y, p = _sanitize_events(events, h, w)
    if t.size == 0:
        return volume

    t_norm = t / t_window if t_window > 0 else np.zeros_like(t)
    for i, f in enumerate(freqs):
        phase = np.pi * f * t_norm
        np.add.at(volume[..., 2 * i], (y, x), p * np.cos(phase))
        np.add.at(volume[..., 2 * i + 1], (y, x), p * np.sin(phase))

    return volume


def event_driven_dtft_encode(events: np.ndarray, h: int, w: int, _t_window: float, freqs: np.ndarray) -> np.ndarray:
    volume = np.zeros((h, w, 2 * len(freqs)), dtype=np.float64)
    if events.size == 0:
        return volume

    t, x, y, p = _sanitize_events(events, h, w)
    if t.size == 0:
        return volume

    two_pi_t = 2.0 * np.pi * t
    for i, f in enumerate(freqs):
        phase = two_pi_t * f
        np.add.at(volume[..., 2 * i], (y, x), p * np.cos(phase))
        np.add.at(volume[..., 2 * i + 1], (y, x), p * np.sin(-phase))

    return volume


def _morlet_wavelet(t: np.ndarray, scale: float):
    t_scaled = t / scale
    envelope = np.exp(-0.5 * t_scaled**2) / (scale * (np.sqrt(np.pi) ** 0.25))
    omega = 5.0
    return envelope * np.cos(omega * t_scaled), envelope * np.sin(omega * t_scaled)


def event_driven_dwt_encode(events: np.ndarray, h: int, w: int, t_window: float, scales: np.ndarray) -> np.ndarray:
    volume = np.zeros((h, w, 2 * len(scales)), dtype=np.float64)
    if events.size == 0:
        return volume

    t, x, y, p = _sanitize_events(events, h, w)
    if t.size == 0:
        return volume

    t_norm = (t - t.min()) / t_window if t_window > 0 else np.zeros_like(t)
    for i, scale in enumerate(scales):
        real_w, imag_w = _morlet_wavelet(t_norm, scale)
        np.add.at(volume[..., 2 * i], (y, x), p * real_w)
        np.add.at(volume[..., 2 * i + 1], (y, x), p * imag_w)

    return volume


def event_driven_dct_decode(dct_volume: np.ndarray, t_grid: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    h, w, _ = dct_volume.shape
    n_t = len(t_grid)
    reconstructed = np.zeros((h, w, n_t), dtype=np.float64)

    for i, f in enumerate(freqs):
        cos_coeff = dct_volume[..., 2 * i]
        sin_coeff = dct_volume[..., 2 * i + 1]
        for t_idx, t in enumerate(t_grid):
            phase = np.pi * f * t
            reconstructed[..., t_idx] += cos_coeff * np.cos(phase) + sin_coeff * np.sin(phase)

    return reconstructed


def event_driven_dtft_decode(dtft_volume: np.ndarray, t_grid: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    h, w, _ = dtft_volume.shape
    n_t = len(t_grid)
    reconstructed = np.zeros((h, w, n_t), dtype=np.float64)

    for i, f in enumerate(freqs):
        cos_coeff = dtft_volume[..., 2 * i]
        sin_coeff = dtft_volume[..., 2 * i + 1]
        for t_idx, t in enumerate(t_grid):
            phase = 2.0 * np.pi * f * t
            reconstructed[..., t_idx] += cos_coeff * np.cos(phase) + sin_coeff * np.sin(-phase)

    return reconstructed


def event_driven_dwt_decode(dwt_volume: np.ndarray, t_grid: np.ndarray, scales: np.ndarray) -> np.ndarray:
    h, w, _ = dwt_volume.shape
    n_t = len(t_grid)
    reconstructed = np.zeros((h, w, n_t), dtype=np.float64)

    for i, scale in enumerate(scales):
        real_coeff = dwt_volume[..., 2 * i]
        imag_coeff = dwt_volume[..., 2 * i + 1]
        for t_idx, t in enumerate(t_grid):
            real_w, imag_w = _morlet_wavelet(np.array([t], dtype=np.float64), scale)
            reconstructed[..., t_idx] += real_coeff * real_w[0] + imag_coeff * imag_w[0]

    return reconstructed


def encode_events(
    events: np.ndarray,
    method: str,
    h: int,
    w: int,
    m: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if events.size == 0:
        return np.zeros((h, w, 2 * m), dtype=np.float64), np.zeros(m, dtype=np.float64)

    t_window = float(np.max(events[:, 0]) - np.min(events[:, 0]))
    method_u = method.upper()

    if method_u == "DCT":
        freqs = sample_frequencies_dct(m, seed=seed)
        return event_driven_dct_encode(events, h, w, t_window, freqs), freqs
    if method_u == "DTFT":
        freqs = sample_frequencies_dtft(m, seed=seed)
        return event_driven_dtft_encode(events, h, w, t_window, freqs), freqs
    if method_u == "DWT":
        scales = sample_frequencies_dwt(m, seed=seed)
        return event_driven_dwt_encode(events, h, w, t_window, scales), scales

    raise ValueError(f"Unsupported method: {method}")
