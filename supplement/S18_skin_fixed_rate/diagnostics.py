from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from metrics import crossing_time, monotone_envelope


def semilogy_slope(mu: np.ndarray, t: np.ndarray, floor: float = 1.0e-14) -> np.ndarray:
    mu = monotone_envelope(np.asarray(mu, dtype=float))
    t = np.asarray(t, dtype=float)
    logmu = np.log(np.clip(mu, floor, None))
    window = min(11, len(t) if len(t) % 2 == 1 else len(t) - 1)
    if window >= 5:
        poly = 3 if window >= 7 else 2
        logmu = savgol_filter(logmu, window_length=window, polyorder=poly, mode="interp")
    return -np.gradient(logmu, t)


def threshold_stats(curves: np.ndarray, t: np.ndarray, eps: float) -> dict[str, object]:
    curves = np.asarray(curves, dtype=float)
    mu = curves.mean(axis=0)
    tstar = crossing_time(mu, t, eps)
    tmix = np.array([crossing_time(c, t, eps) for c in curves], dtype=float)
    H = float(np.nanquantile(tmix, 0.9) - np.nanquantile(tmix, 0.1))
    g_at_tstar = np.array(
        [np.interp(tstar, t, monotone_envelope(c)) for c in curves],
        dtype=float,
    )
    V = float(np.quantile(g_at_tstar, 0.9) - np.quantile(g_at_tstar, 0.1))
    s = semilogy_slope(mu, t)
    s_cross = float(np.interp(tstar, t, s))
    m_cross = float(eps * s_cross)
    return {
        "mu": mu,
        "tstar": float(tstar),
        "tmix": tmix,
        "H": H,
        "V": V,
        "g_at_tstar": g_at_tstar,
        "slope_curve": s,
        "s_cross": s_cross,
        "m_cross": m_cross,
    }


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if mask.sum() < 2:
        return np.nan
    coeff = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    return float(coeff[0])
