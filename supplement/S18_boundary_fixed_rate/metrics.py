from __future__ import annotations

import numpy as np


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = 0.5 * ((rho - sigma) + (rho - sigma).conj().T)
    return float(np.sum(np.abs(np.linalg.eigvalsh(diff))))


def monotone_envelope(y: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.asarray(y, dtype=float))


def crossing_time(y: np.ndarray, t: np.ndarray, eps: float) -> float:
    y = monotone_envelope(np.asarray(y, dtype=float))
    t = np.asarray(t, dtype=float)
    idx = np.where(y <= eps)[0]
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1 = t[i - 1], t[i]
    y0, y1 = y[i - 1], y[i]
    if np.isclose(y1, y0):
        return float(t1)
    return float(t0 + (eps - y0) * (t1 - t0) / (y1 - y0))
