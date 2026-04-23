from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import eig, eigvals
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

from diagnostics import threshold_stats
from metrics import crossing_time


def Q_skin(L: int, g_right: float, g_left: float, scale: float = 1.0) -> csc_matrix:
    Q = np.zeros((L, L), dtype=float)
    for j in range(L):
        if j < L - 1:
            Q[j + 1, j] += scale * g_right
            Q[j, j] -= scale * g_right
        if j > 0:
            Q[j - 1, j] += scale * g_left
            Q[j, j] -= scale * g_left
    return csc_matrix(Q)


def pi_skin(L: int, g_right: float, g_left: float) -> np.ndarray:
    if np.isclose(g_right, g_left):
        return np.full(L, 1.0 / L)
    logw = np.arange(L, dtype=float) * np.log(g_right / g_left)
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def _is_linear_grid(t: np.ndarray) -> bool:
    if len(t) < 3:
        return True
    dt = np.diff(np.asarray(t, dtype=float))
    return bool(np.allclose(dt, dt[0], atol=1.0e-12, rtol=1.0e-9))


def evolve_population_states(Q: csc_matrix, p0_batch: np.ndarray, t: np.ndarray) -> np.ndarray:
    p0_batch = np.atleast_2d(np.asarray(p0_batch, dtype=float))
    B = p0_batch.T
    t = np.asarray(t, dtype=float)
    if _is_linear_grid(t):
        P = expm_multiply(
            Q,
            B,
            start=float(t[0]),
            stop=float(t[-1]),
            num=len(t),
            endpoint=True,
        )
    else:
        P = np.stack([expm_multiply(Q * float(tt), B) for tt in t], axis=0)
    if P.ndim == 2:
        P = P[:, :, None]
    return np.real_if_close(P)


def population_curves(Q: csc_matrix, p0_batch: np.ndarray, pi: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = evolve_population_states(Q, p0_batch, t)
    return np.abs(P - pi[None, :, None]).sum(axis=1).T


def population_curves_batched(
    Q: csc_matrix,
    p0_all: np.ndarray,
    pi: np.ndarray,
    t: np.ndarray,
    batch_size: int = 100,
) -> np.ndarray:
    p0_all = np.asarray(p0_all, dtype=float)
    curves = np.empty((len(p0_all), len(t)), dtype=float)
    for start in range(0, len(p0_all), batch_size):
        stop = min(start + batch_size, len(p0_all))
        curves[start:stop] = population_curves(Q, p0_all[start:stop], pi, t)
    return curves


def spectral_gap_and_left_mode(Q: csc_matrix) -> tuple[float, np.ndarray]:
    Qd = np.asarray(Q.toarray(), dtype=float)
    evals = eigvals(Qd)
    nonzero = np.where(np.real(evals) < -1.0e-10)[0]
    slow_idx = int(nonzero[np.argmax(np.real(evals[nonzero]))])
    slow_eval = evals[slow_idx]
    Delta = float(-np.real(slow_eval))

    evals_T, vecs_T = eig(Qd.T)
    left_idx = int(np.argmin(np.abs(evals_T - slow_eval)))
    l2 = np.real_if_close(vecs_T[:, left_idx]).astype(float)
    l2 /= np.max(np.abs(l2))
    if l2[0] < 0:
        l2 = -l2
    return Delta, l2


def slow_overlap_data(
    left_mode: np.ndarray,
    populations: np.ndarray,
    delta: float,
) -> dict[str, object]:
    a2 = np.asarray(populations, dtype=float) @ np.asarray(left_mode, dtype=float)
    abs_a2 = np.abs(a2)
    return {
        "a2": a2,
        "abs_a2": abs_a2,
        "A_max": float(np.max(np.abs(left_mode))),
        "alpha_typ": float(np.quantile(abs_a2, 1.0 - delta)),
    }


def reference_curve(Q: csc_matrix, pi: np.ndarray, t: np.ndarray) -> np.ndarray:
    ref = np.full((1, len(pi)), 1.0 / len(pi))
    return population_curves(Q, ref, pi, t)[0]


def scout_time_grid(
    Q: csc_matrix,
    pi: np.ndarray,
    samples: np.ndarray,
    eps: float,
    Delta: float,
    scout_points: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    L = len(pi)
    t_max = max(8.0, 6.0 / max(Delta, 1.0e-12))
    basis = np.eye(L, dtype=float)
    for _ in range(10):
        t = np.concatenate(([0.0], np.geomspace(1.0e-3, t_max, scout_points - 1)))
        scout_curves = population_curves(Q, samples, pi, t)
        basis_curves = population_curves(Q, basis, pi, t)
        mu = scout_curves.mean(axis=0)
        tstar = crossing_time(mu, t, eps)
        tmix_basis = np.array([crossing_time(c, t, eps) for c in basis_curves], dtype=float)
        if np.isfinite(tstar) and np.all(np.isfinite(tmix_basis)):
            return t, scout_curves, basis_curves, float(tstar), tmix_basis
        t_max *= 2.0
    raise RuntimeError("Single-particle scout window failed to reach the threshold.")


def run_single_particle_case(
    *,
    L: int,
    g_right: float,
    g_left: float,
    scale: float,
    populations: np.ndarray,
    eps_values: Sequence[float],
    delta: float,
    scout_samples: int = 64,
    scout_points: int = 80,
    linear_points: int = 320,
    batch_size: int = 100,
) -> dict[str, object]:
    eps_values = [float(eps) for eps in eps_values]
    eps_floor = float(min(eps_values))

    Q = Q_skin(L, g_right, g_left, scale=scale)
    pi = pi_skin(L, g_right, g_left)
    Delta, left_mode = spectral_gap_and_left_mode(Q)

    t_scout, scout_curves, basis_scout, tstar_scout, tmix_basis_scout = scout_time_grid(
        Q,
        pi,
        populations[: min(len(populations), scout_samples)],
        eps_floor,
        Delta,
        scout_points=scout_points,
    )
    tmix_scout = np.array([crossing_time(c, t_scout, eps_floor) for c in scout_curves], dtype=float)
    scout_anchor = np.nanmax(
        [
            tstar_scout,
            np.nanquantile(tmix_scout, 0.9),
            np.nanmax(tmix_basis_scout),
            3.0 / max(Delta, 1.0e-12),
        ]
    )
    t = np.linspace(0.0, 1.25 * scout_anchor, linear_points)
    curves = population_curves_batched(Q, populations, pi, t, batch_size=batch_size)
    basis_curves = population_curves(Q, np.eye(L, dtype=float), pi, t)
    h = reference_curve(Q, pi, t)
    overlap_data = slow_overlap_data(left_mode, populations, delta)

    eps_summary: dict[float, dict[str, object]] = {}
    for eps in eps_values:
        stats = threshold_stats(curves, t, eps)
        tmix_basis = np.array([crossing_time(c, t, eps) for c in basis_curves], dtype=float)
        tworst = float(np.nanmax(tmix_basis))
        t_ref = float(crossing_time(h, t, eps))
        eps_summary[eps] = {
            "eps": eps,
            "tstar": stats["tstar"],
            "tmix": stats["tmix"],
            "mu": stats["mu"],
            "H": stats["H"],
            "V": stats["V"],
            "slope_curve": stats["slope_curve"],
            "s_cross": stats["s_cross"],
            "m_cross": stats["m_cross"],
            "t_ref": t_ref,
            "tworst": tworst,
            "tmix_std": float(np.nanstd(stats["tmix"])),
            "V_over_m": float(stats["V"] / max(stats["m_cross"], 1.0e-12)),
            "gap_ratio": float(stats["s_cross"] / max(Delta, 1.0e-12)),
            "delta_t_pred": float(
                np.log(max(overlap_data["A_max"], 1.0e-14) / max(overlap_data["alpha_typ"], 1.0e-14))
                / max(Delta, 1.0e-14)
            ),
            "delta_t_act": float(tworst - np.nanquantile(stats["tmix"], 1.0 - delta)),
        }

    return {
        "L": L,
        "scale": float(scale),
        "Q": Q,
        "pi": pi,
        "t": t,
        "curves": curves,
        "basis_curves": basis_curves,
        "h": h,
        "Delta": Delta,
        "left_mode": left_mode,
        "overlap_data": overlap_data,
        "eps_summary": eps_summary,
        "scout_time_grid": t_scout,
        "scout_curves": scout_curves,
        "basis_scout": basis_scout,
    }
