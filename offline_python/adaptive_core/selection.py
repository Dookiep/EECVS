import numpy as np


def magnitude_selection(volume: np.ndarray, keep_ratio: float):
    if keep_ratio >= 1.0:
        mask = np.ones_like(volume, dtype=bool)
        return volume.copy(), mask, {"method": "magnitude", "selected": int(mask.sum())}

    nonzero = volume != 0
    if nonzero.sum() == 0:
        return np.zeros_like(volume), np.zeros_like(volume, dtype=bool), {
            "method": "magnitude",
            "selected": 0,
        }

    abs_vals = np.abs(volume[nonzero])
    n_keep = max(1, int(len(abs_vals) * keep_ratio))
    threshold = np.partition(abs_vals, -n_keep)[-n_keep]
    mask = (np.abs(volume) >= threshold) & nonzero
    return volume * mask.astype(volume.dtype), mask, {
        "method": "magnitude",
        "selected": int(mask.sum()),
        "total_nonzero": int(nonzero.sum()),
        "threshold": float(threshold),
    }


def frequency_low_selection(volume: np.ndarray, keep_ratio: float):
    if keep_ratio >= 1.0:
        mask = np.ones_like(volume, dtype=bool)
        return volume.copy(), mask, {"method": "frequency_low", "selected": int(mask.sum())}

    h, w, c = volume.shape
    nonzero = volume != 0
    total_nonzero = int(nonzero.sum())
    if total_nonzero == 0:
        return np.zeros_like(volume), np.zeros_like(volume, dtype=bool), {
            "method": "frequency_low",
            "selected": 0,
        }

    target = max(1, int(total_nonzero * keep_ratio))
    mask = np.zeros_like(volume, dtype=bool)
    selected = 0

    for ch in range(c):
        ch_mask = nonzero[:, :, ch]
        count = int(ch_mask.sum())
        if count == 0:
            continue
        if selected + count <= target:
            mask[:, :, ch] = ch_mask
            selected += count
        else:
            remaining = target - selected
            if remaining <= 0:
                break
            ys, xs = np.where(ch_mask)
            idx = np.arange(len(ys))
            keep_idx = idx[:remaining]
            mask[ys[keep_idx], xs[keep_idx], ch] = True
            selected += remaining
            break

    return volume * mask.astype(volume.dtype), mask, {
        "method": "frequency_low",
        "selected": int(mask.sum()),
        "total_nonzero": total_nonzero,
    }


def frequency_high_selection(volume: np.ndarray, keep_ratio: float):
    rev = volume[:, :, ::-1]
    selected, mask_rev, info = frequency_low_selection(rev, keep_ratio)
    return selected[:, :, ::-1], mask_rev[:, :, ::-1], {
        "method": "frequency_high",
        "selected": info.get("selected", 0),
        "total_nonzero": info.get("total_nonzero", 0),
    }


SELECTION_METHODS = {
    "magnitude": magnitude_selection,
    "frequency_low": frequency_low_selection,
    "frequency_high": frequency_high_selection,
}


def apply_selection_method(volume: np.ndarray, method_name: str, keep_ratio: float = 0.3):
    if method_name not in SELECTION_METHODS:
        available = ", ".join(SELECTION_METHODS.keys())
        raise ValueError(f"Unknown selection method '{method_name}'. Available: {available}")
    return SELECTION_METHODS[method_name](volume, keep_ratio)
