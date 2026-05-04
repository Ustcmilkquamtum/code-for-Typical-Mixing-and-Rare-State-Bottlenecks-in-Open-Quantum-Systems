#!/usr/bin/env python3
r"""
PRL-style numerical figures for a unital boundary-mode toy Lindbladian.

The model on L qubits is

    L(rho) = Delta/2 (X_1 rho X_1 - rho)
           + Gamma/2 (Z_1 rho Z_1 - rho)
           + Gamma/2 sum_{j=2}^L [(X_j rho X_j - rho) + (Z_j rho Z_j - rho)].

For 0 < Delta < Gamma it is unital, has the unique stationary state I/2^L,
and its unique slowest nontrivial left eigenoperator is

    L_2 = Z_1 \otimes I_{2...L},       Tr(L_2)=0,

with decay rate Delta.

The script produces two logically separate checks:

1. Fixed epsilon: full-density Haar pure-state relaxation curves and
   fixed-threshold mixing-time spread.  This is a concentration check.

2. Scaled epsilon_L = eps0 * 2^{-L/2}: boundary-overlap one-mode gap.
   This is the clean numerical check of the linear rare-state bottleneck law.

All outputs are written to the current working directory.
"""

from __future__ import annotations

import csv
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    Delta: float = 0.25
    Gamma: float = 4.0

    # Fixed-threshold concentration figure.  The exploratory value 0.70 is not
    # needed here; 0.55 is still safely above the small-L typical slow-overlap
    # scale, but low enough to look less like a trivial early-time crossing.
    eps_fixed: float = 0.55

    # Scaled threshold for the one-mode boundary-overlap gap.
    eps0_scaled: float = 0.10
    delta_quantile: float = 0.10  # q_{1-delta}=q0.9

    seed: int = 20260504

    fixed_L_values: Tuple[int, ...] = (3, 4, 5, 6)
    fixed_samples_by_L: Tuple[Tuple[int, int], ...] = ((3, 64), (4, 56), (5, 48), (6, 36))
    fixed_t_max: float = 2.8
    fixed_n_times: int = 64
    fixed_n_plot_samples: int = 36

    scaled_L_min: int = 4
    scaled_L_max: int = 20
    scaled_n_overlap_samples: int = 50_000


# -----------------------------------------------------------------------------
# Plot style: close to the manuscript figures.
# -----------------------------------------------------------------------------

COL = {
    "samples": "0.72",
    "mean": "black",
    "threshold": "#2166ac",  # blue threshold guide, close to Fig. S5/S6
    "crossing": "#b2182b",   # red dotted crossing marker
    "red": "#b2182b",
    "blue": "#2166ac",
    "purple": "#6a3d9a",
    "gray": "0.55",
    "fit": "0.40",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.size": 8.2,
        "axes.labelsize": 8.6,
        "axes.titlesize": 8.2,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "lines.linewidth": 1.25,
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> List[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
        print(f"saved {path}")
    return paths


# -----------------------------------------------------------------------------
# Haar sampling
# -----------------------------------------------------------------------------


def haar_vectors(n_samples: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random pure state vectors in C^d."""
    z = rng.normal(size=(n_samples, d)) + 1j * rng.normal(size=(n_samples, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def haar_populations(n_samples: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Populations |psi_x|^2 of Haar-random pure states: Dirichlet(1,...,1)."""
    x = rng.exponential(scale=1.0, size=(n_samples, d))
    x /= x.sum(axis=1, keepdims=True)
    return x


# -----------------------------------------------------------------------------
# Full-density exact Pauli-channel simulation
# -----------------------------------------------------------------------------


def pauli_channel_probabilities(lambda_x: float, lambda_y: float, lambda_z: float) -> Tuple[float, float, float, float]:
    """Kraus probabilities p_I,p_X,p_Y,p_Z from one-qubit Pauli eigenvalues."""
    p_i = (1.0 + lambda_x + lambda_y + lambda_z) / 4.0
    p_x = (1.0 + lambda_x - lambda_y - lambda_z) / 4.0
    p_y = (1.0 - lambda_x + lambda_y - lambda_z) / 4.0
    p_z = (1.0 - lambda_x - lambda_y + lambda_z) / 4.0
    probs = np.array([p_i, p_x, p_y, p_z], dtype=float)
    if probs.min() < -1e-12:
        raise ValueError(f"non-positive Pauli-channel probability: {probs}")
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return tuple(float(x) for x in probs)


def site_cache(L: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Permutation and phase arrays for Pauli conjugations on all sites."""
    d = 1 << L
    idx = np.arange(d)
    out = []
    for site in range(L):
        perm = idx ^ (1 << site)
        z_phase = (1 - 2 * ((idx >> site) & 1)).astype(float)
        y_phase = 1j * z_phase.astype(complex)  # Y|0>=i|1>, Y|1>=-i|0>
        out.append((perm, z_phase, y_phase))
    return out


def apply_one_site_pauli_channel_batch(
    rho: np.ndarray,
    cache_entry: Tuple[np.ndarray, np.ndarray, np.ndarray],
    probs: Tuple[float, float, float, float],
) -> np.ndarray:
    """Apply pI*rho+pX*XrhoX+pY*YrhoY+pZ*ZrhoZ to a batch of density matrices."""
    p_i, p_x, p_y, p_z = probs
    perm, z_phase, y_phase = cache_entry

    x_rho_x = rho[:, perm, :][:, :, perm]
    z_rho_z = (z_phase[None, :, None] * z_phase[None, None, :]) * rho

    # If U|r> = c_r |perm(r)>, then (U rho U^dagger)_{ab}
    # = c_{perm(a)} c^*_{perm(b)} rho_{perm(a),perm(b)}.
    old_phase = y_phase[perm]
    y_rho_y = (old_phase[None, :, None] * np.conj(old_phase)[None, None, :]) * x_rho_x

    return p_i * rho + p_x * x_rho_x + p_y * y_rho_y + p_z * z_rho_z


def apply_full_pauli_channel_batch(rho0: np.ndarray, L: int, t: float, p: Params, cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Exact full density-matrix channel e^{tL} for the toy Lindbladian."""
    out = np.array(rho0, copy=True)

    # Boundary site: X noise at Delta and Z noise at Gamma in Pauli-transfer rates.
    lam_x = math.exp(-p.Gamma * t)
    lam_z = math.exp(-p.Delta * t)
    lam_y = math.exp(-(p.Gamma + p.Delta) * t)
    out = apply_one_site_pauli_channel_batch(out, cache[0], pauli_channel_probabilities(lam_x, lam_y, lam_z))

    # Bulk sites: X and Z noises both at Gamma.
    lam_xz = math.exp(-p.Gamma * t)
    lam_y_bulk = math.exp(-2.0 * p.Gamma * t)
    bulk_probs = pauli_channel_probabilities(lam_xz, lam_y_bulk, lam_xz)
    for site in range(1, L):
        out = apply_one_site_pauli_channel_batch(out, cache[site], bulk_probs)
    return out


def trace_norm_distances_to_uniform_batch(rhos: np.ndarray) -> np.ndarray:
    """Trace norm ||rho - I/d||_1 for a batch of density matrices. No factor 1/2."""
    d = rhos.shape[1]
    vals = np.linalg.eigvalsh(rhos - np.eye(d, dtype=rhos.dtype)[None, :, :] / d)
    return np.sum(np.abs(vals), axis=1).real


def full_density_curves(psis: np.ndarray, L: int, times: np.ndarray, p: Params) -> np.ndarray:
    """Exact full trace-norm relaxation curves for Haar-random pure states."""
    rho0 = psis[:, :, None] * np.conj(psis[:, None, :])
    cache = site_cache(L)
    curves = np.empty((len(times), len(psis)), dtype=float)
    for k, t in enumerate(times):
        rhos_t = apply_full_pauli_channel_batch(rho0, L, float(t), p, cache)
        curves[k] = trace_norm_distances_to_uniform_batch(rhos_t)
    return curves


# -----------------------------------------------------------------------------
# Population-sector tools for the scaled-epsilon validation
# -----------------------------------------------------------------------------


def bitflip_rates(L: int, p: Params) -> np.ndarray:
    rates = np.full(L, p.Gamma, dtype=float)
    rates[0] = p.Delta
    return rates


def apply_product_bitflip(P: np.ndarray, L: int, t: float, rates: np.ndarray) -> np.ndarray:
    """Exact diagonal population channel generated by the X-type jumps."""
    Q = np.array(P, copy=True, dtype=float)
    n = Q.shape[0]
    for site, rate in enumerate(rates):
        e = math.exp(-rate * t)
        a = 0.5 * (1.0 + e)
        b = 0.5 * (1.0 - e)
        low = 1 << site
        high = 1 << (L - site - 1)
        V = Q.reshape(n, high, 2, low)
        v0 = V[:, :, 0, :].copy()
        v1 = V[:, :, 1, :].copy()
        V[:, :, 0, :] = a * v0 + b * v1
        V[:, :, 1, :] = b * v0 + a * v1
    return Q


def population_trace_distance(P: np.ndarray) -> np.ndarray:
    d = P.shape[1]
    return np.abs(P - 1.0 / d).sum(axis=1)


def population_curves(P0: np.ndarray, L: int, times: np.ndarray, p: Params) -> np.ndarray:
    rates = bitflip_rates(L, p)
    curves = np.empty((len(times), P0.shape[0]), dtype=float)
    for k, t in enumerate(times):
        curves[k] = population_trace_distance(apply_product_bitflip(P0, L, float(t), rates))
    return curves


def abs_z1_overlap_from_populations(P: np.ndarray) -> np.ndarray:
    d = P.shape[1]
    signs = np.ones(d)
    signs[(np.arange(d) & 1) == 1] = -1.0
    return np.abs(P @ signs)


# -----------------------------------------------------------------------------
# Common crossing and quantile helpers
# -----------------------------------------------------------------------------


def crossing_times_from_grid(times: np.ndarray, curves: np.ndarray, eps: float) -> np.ndarray:
    """First crossing below eps, with log-linear interpolation between grid points."""
    if curves.ndim == 1:
        curves = curves[:, None]
    out = np.full(curves.shape[1], np.nan)
    for s in range(curves.shape[1]):
        y = curves[:, s]
        if y[0] <= eps:
            out[s] = times[0]
            continue
        idx = np.flatnonzero(y <= eps)
        if len(idx) == 0:
            continue
        j = int(idx[0])
        if j == 0:
            out[s] = times[0]
            continue
        t0, t1 = float(times[j - 1]), float(times[j])
        y0, y1 = max(float(y[j - 1]), 1e-300), max(float(y[j]), 1e-300)
        frac = (math.log(eps) - math.log(y0)) / (math.log(y1) - math.log(y0))
        frac = min(1.0, max(0.0, frac))
        out[s] = t0 + frac * (t1 - t0)
    return out


def exact_abs_z1_quantile(L: int, probability: float) -> float:
    """Exact Haar quantile of |<Z_1>|.

    If B~Beta(2^{L-1},2^{L-1}), then <Z_1>=2B-1.  By symmetry,
    P(|2B-1| <= q) = probability implies F_B((1+q)/2)=(1+probability)/2.
    """
    a = 1 << (L - 1)
    return float(2.0 * beta_dist.ppf((1.0 + probability) / 2.0, a, a) - 1.0)


def sample_abs_z1_overlap(L: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    half_dim = 1 << (L - 1)
    B = rng.beta(half_dim, half_dim, size=n_samples)
    return np.abs(2.0 * B - 1.0)


# -----------------------------------------------------------------------------
# Figure 1: fixed-epsilon full-density concentration
# -----------------------------------------------------------------------------


def make_fixed_epsilon_figures(p: Params, output_dir: Path) -> Tuple[List[str], List[Path]]:
    rng = np.random.default_rng(p.seed)
    times = np.linspace(0.0, p.fixed_t_max, p.fixed_n_times)
    samples_by_L: Dict[int, int] = dict(p.fixed_samples_by_L)

    rows: List[Dict[str, float]] = []
    curves_by_L: Dict[int, np.ndarray] = {}
    mean_by_L: Dict[int, np.ndarray] = {}
    tmean_by_L: Dict[int, float] = {}

    for L in p.fixed_L_values:
        d = 1 << L
        n_samples = samples_by_L[L]
        psis = haar_vectors(n_samples, d, rng)
        curves = full_density_curves(psis, L, times, p)
        tmix = crossing_times_from_grid(times, curves, p.eps_fixed)
        mean_curve = curves.mean(axis=1)
        tmean = crossing_times_from_grid(times, mean_curve, p.eps_fixed)[0]
        if np.any(~np.isfinite(tmix)) or not np.isfinite(tmean):
            raise RuntimeError(f"time grid too short for fixed-epsilon check at L={L}")

        q10, q50, q90 = np.quantile(tmix, [0.1, 0.5, 0.9])
        rows.append({
            "L": float(L),
            "d": float(d),
            "epsilon": float(p.eps_fixed),
            "n_samples": float(n_samples),
            "mean_tmix": float(np.mean(tmix)),
            "median_tmix": float(q50),
            "std_tmix": float(np.std(tmix, ddof=1)),
            "q10_tmix": float(q10),
            "q90_tmix": float(q90),
            "q90_minus_q10": float(q90 - q10),
            "mean_curve_crossing": float(tmean),
        })
        curves_by_L[L] = curves
        mean_by_L[L] = mean_curve
        tmean_by_L[L] = float(tmean)

    csv_path = output_dir / "toy_fixed_epsilon_full_density_stats_prlstyle.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")

    # Bundle panels.  No suptitle; the caption should carry the model details.
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.45), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, L in zip(axes, p.fixed_L_values):
        d = 1 << L
        curves = curves_by_L[L]
        n_plot = min(p.fixed_n_plot_samples, curves.shape[1])
        plot_idx = np.linspace(0, curves.shape[1] - 1, n_plot, dtype=int)
        for s in plot_idx:
            ax.semilogy(times, np.maximum(curves[:, s], 1e-14), color=COL["samples"], lw=0.55, alpha=0.24)
        ax.semilogy(times, np.maximum(mean_by_L[L], 1e-14), color=COL["mean"], lw=1.55)
        ax.axhline(p.eps_fixed, color=COL["threshold"], ls="--", lw=1.0)
        ax.axvline(tmean_by_L[L], color=COL["crossing"], ls=":", lw=0.95)
        ax.text(0.055, 0.085, rf"$L={L},\ d={d}$", transform=ax.transAxes)
        ax.grid(True, which="both", lw=0.32, alpha=0.22)
        ax.set_xlim(0.0, 1.55)
        ax.set_ylim(1e-2, 2.1)
    axes[0].set_ylabel(r"$g_L(t,\psi)$")
    axes[2].set_ylabel(r"$g_L(t,\psi)$")
    axes[2].set_xlabel("time")
    axes[3].set_xlabel("time")

    axes[0].plot([], [], color=COL["samples"], lw=1.0, label="Haar samples")
    axes[0].plot([], [], color=COL["mean"], lw=1.55, label="empirical mean")
    axes[0].plot([], [], color=COL["threshold"], ls="--", lw=1.0, label=rf"$\epsilon={p.eps_fixed:.2f}$")
    axes[0].legend(frameon=False, loc="upper right", handlelength=2.0)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.105, top=0.985, wspace=0.10, hspace=0.13)
    paths = save_figure(fig, output_dir, "toy_fixed_epsilon_full_density_bundles_prlstyle")
    plt.close(fig)

    # Spread panel, styled like the lower-right panel of Fig. 2.
    ds = np.array([r["d"] for r in rows])
    stds = np.array([r["std_tmix"] for r in rows])
    widths = np.array([r["q90_minus_q10"] for r in rows])
    xguide = 1.0 / np.sqrt(ds)
    amp = float(np.sum(widths * xguide) / np.sum(xguide * xguide))
    guide = amp * xguide

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.plot(ds, stds, marker="o", ms=3.5, color=COL["red"], label="sample std")
    ax.plot(ds, widths, marker="D", ms=3.3, color=COL["purple"], label=r"$q_{0.9}-q_{0.1}$")
    ax.plot(ds, guide, ls=":", color=COL["gray"], lw=1.0, label=r"$d^{-1/2}$ guide")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(int(d)) for d in ds])
    ax.set_xlabel(r"Hilbert-space dimension $d$")
    ax.set_ylabel("spread of mixing times")
    ax.grid(True, which="both", lw=0.32, alpha=0.24)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.18, top=0.985)
    paths += save_figure(fig, output_dir, "toy_fixed_epsilon_full_density_spread_prlstyle")
    plt.close(fig)

    lines = ["fixed-epsilon full-density concentration check:"]
    for r in rows:
        lines.append(
            f"  L={int(r['L'])}, d={int(r['d'])}, n={int(r['n_samples'])}: "
            f"mean={r['mean_tmix']:.6f}, std={r['std_tmix']:.6f}, "
            f"q90-q10={r['q90_minus_q10']:.6f}, mean-crossing={r['mean_curve_crossing']:.6f}"
        )
    return lines, [csv_path] + paths


# -----------------------------------------------------------------------------
# Figure 2: scaled-epsilon one-mode gap
# -----------------------------------------------------------------------------


def make_scaled_epsilon_gap_figure(p: Params, output_dir: Path) -> Tuple[List[str], List[Path]]:
    rng = np.random.default_rng(p.seed + 1)
    qprob = 1.0 - p.delta_quantile

    rows: List[Dict[str, float]] = []
    for L in range(p.scaled_L_min, p.scaled_L_max + 1):
        d = 1 << L
        eps_L = p.eps0_scaled / math.sqrt(d)
        abs_a_mc = sample_abs_z1_overlap(L, p.scaled_n_overlap_samples, rng)
        q_abs_mc = float(np.quantile(abs_a_mc, qprob))
        q_abs_beta = exact_abs_z1_quantile(L, qprob)

        # In the one-mode tail, g_t(psi)=|<Z_1>| exp(-Delta t), with ||R_2||_1=1.
        t_worst = math.log(1.0 / eps_L) / p.Delta
        t_typ_mc = math.log(q_abs_mc / eps_L) / p.Delta
        t_typ_beta = math.log(q_abs_beta / eps_L) / p.Delta
        gap_mc = t_worst - t_typ_mc
        gap_beta = t_worst - t_typ_beta
        guide = L * math.log(2.0) / (2.0 * p.Delta)

        rows.append({
            "L": float(L),
            "d": float(d),
            "epsilon_L": float(eps_L),
            "q_abs_overlap_MC": float(q_abs_mc),
            "q_abs_overlap_Beta": float(q_abs_beta),
            "t_worst_basis": float(t_worst),
            "t_typ_q90_MC": float(t_typ_mc),
            "t_typ_q90_Beta": float(t_typ_beta),
            "gap_MC": float(gap_mc),
            "gap_Beta_prediction": float(gap_beta),
            "linear_slope_guide": float(guide),
        })

    csv_path = output_dir / "toy_scaled_epsilon_gap_data_prlstyle.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")

    Ls = np.array([r["L"] for r in rows])
    gap_mc = np.array([r["gap_MC"] for r in rows])
    gap_beta = np.array([r["gap_Beta_prediction"] for r in rows])
    guide = np.array([r["linear_slope_guide"] for r in rows])
    slope, intercept = np.polyfit(Ls, gap_mc, 1)
    fit = slope * Ls + intercept
    theory_slope = math.log(2.0) / (2.0 * p.Delta)
    r2 = 1.0 - float(np.sum((gap_mc - fit) ** 2) / np.sum((gap_mc - np.mean(gap_mc)) ** 2))

    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    ax.plot(Ls, gap_mc, marker="o", ms=3.3, color="black", label="sampled gap")
    ax.plot(Ls, gap_beta, marker="s", ms=3.0, ls="--", color=COL["red"], label="Beta prediction")
    ax.plot(Ls, guide, ls="--", color=COL["blue"], lw=1.0, label=r"$L\,\log 2/(2\Delta)$")
    ax.plot(Ls, fit, ls=":", color=COL["fit"], lw=1.0, label=rf"fit: {slope:.2f}$L${intercept:+.2f}")
    ax.set_xlabel(r"system size $L$")
    ax.set_ylabel("time difference")
    ax.set_xticks(np.arange(p.scaled_L_min, p.scaled_L_max + 1, 4))
    ax.grid(True, lw=0.32, alpha=0.24)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.17, top=0.985)
    paths = save_figure(fig, output_dir, "toy_scaled_epsilon_gap_prlstyle")
    plt.close(fig)

    lines = ["scaled-epsilon one-mode boundary-overlap check:"]
    lines.append(f"  fit slope={slope:.8f}, intercept={intercept:.8f}, R^2={r2:.8f}")
    lines.append(f"  theory slope log(2)/(2 Delta)={theory_slope:.8f}")
    lines.append(f"  relative slope error={(slope - theory_slope) / theory_slope:+.3%}")
    lines.append(f"  max |sampled gap - Beta prediction|={np.max(np.abs(gap_mc - gap_beta)):.6f}")
    return lines, [csv_path] + paths


# -----------------------------------------------------------------------------
# Internal validation
# -----------------------------------------------------------------------------


def validation_checks(p: Params) -> List[str]:
    if not (0.0 < p.Delta < p.Gamma):
        raise ValueError("Need 0 < Delta < Gamma for a unique slow Z_1 mode.")

    lines = [
        f"parameters: Delta={p.Delta}, Gamma={p.Gamma}, eps_fixed={p.eps_fixed}, eps0_scaled={p.eps0_scaled}",
        "analytic Pauli spectrum:",
        "  site 1: r(Z_1)=Delta, r(X_1)=Gamma, r(Y_1)=Gamma+Delta",
        "  sites j>=2: r(X_j)=Gamma, r(Z_j)=Gamma, r(Y_j)=2 Gamma",
        "  hence Z_1 tensor I is the unique slowest nontrivial Pauli eigenoperator and Tr(L_2)=0.",
    ]

    # Check that the scaled-epsilon crossings are genuinely one-mode in the exact population sector.
    rng = np.random.default_rng(p.seed + 2)
    times = np.linspace(0.0, 32.0, 260)
    lines.append("small exact population-sector validation at scaled epsilon:")
    for L in (4, 5, 6):
        d = 1 << L
        eps_L = p.eps0_scaled / math.sqrt(d)
        P0 = haar_populations(160, d, rng)
        curves = population_curves(P0, L, times, p)
        tmix_exact = crossing_times_from_grid(times, curves, eps_L)
        if np.any(~np.isfinite(tmix_exact)):
            raise RuntimeError(f"validation grid too short at L={L}")
        abs_a = abs_z1_overlap_from_populations(P0)
        tmix_slow = np.maximum(0.0, np.log(np.maximum(abs_a, 1e-300) / eps_L) / p.Delta)
        q_exact = float(np.quantile(tmix_exact, 0.9))
        q_slow = float(np.quantile(tmix_slow, 0.9))
        lines.append(f"  L={L}: exact q0.9={q_exact:.6f}, slow-tail q0.9={q_slow:.6f}, difference={q_exact-q_slow:+.3e}")
    return lines


def zip_outputs(output_dir: Path, files: Iterable[Path]) -> Path:
    zip_path = output_dir / "toy_boundary_prlstyle_final_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, arcname=path.name)
    print(f"saved {zip_path}")
    return zip_path


def main() -> None:
    set_style()
    p = Params()
    output_dir = Path(os.getcwd())

    all_files: List[Path] = []
    lines: List[str] = []
    lines.extend(validation_checks(p))
    lines.append("")

    fixed_lines, fixed_files = make_fixed_epsilon_figures(p, output_dir)
    lines.extend(fixed_lines)
    lines.append("")
    all_files.extend(fixed_files)

    scaled_lines, scaled_files = make_scaled_epsilon_gap_figure(p, output_dir)
    lines.extend(scaled_lines)
    all_files.extend(scaled_files)

    summary_path = output_dir / "toy_boundary_prlstyle_final_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {summary_path}")
    all_files.append(summary_path)
    all_files.append(Path(__file__).resolve())

    zip_outputs(output_dir, all_files)


if __name__ == "__main__":
    main()
