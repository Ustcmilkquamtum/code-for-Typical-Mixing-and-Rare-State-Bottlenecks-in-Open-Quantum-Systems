from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sampling import haar_population_batch
from single_particle_skin import run_single_particle_case


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OLD_EPS = 0.35
NEW_EPS = 0.01
DELTA = 0.10
RNG_SEED = 31042

G_RIGHT = 1.6
G_LEFT = 0.4

REPRESENTATIVE_LS = [16, 24, 32, 64, 128, 192]
SCALING_LS = [16, 24, 32, 48, 64, 96, 128, 192]

NUM_SAMPLES = 500
NUM_BUNDLE = 200
SCOUT_SAMPLES = 64
SCOUT_POINTS = 90
LINEAR_POINTS = 520
BATCH_SIZE = 100

OLD_DARK = "#1f77b4"
OLD_LIGHT = "#9ecae1"
NEW_DARK = "#d62728"
NEW_LIGHT = "#f4a3a8"


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def run_case(L: int) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + 1000 + L)
    populations = haar_population_batch(L, NUM_SAMPLES, rng)
    return run_single_particle_case(
        L=L,
        g_right=G_RIGHT,
        g_left=G_LEFT,
        scale=1.0,
        populations=populations,
        eps_values=[OLD_EPS, NEW_EPS],
        delta=DELTA,
        scout_samples=SCOUT_SAMPLES,
        scout_points=SCOUT_POINTS,
        linear_points=LINEAR_POINTS,
        batch_size=BATCH_SIZE,
    )


def fit_log_gap(L_values: np.ndarray, gaps: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.log(np.asarray(L_values, dtype=float))
    y = np.asarray(gaps, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit_y = slope * x + intercept
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    return x, fit_y, float(slope), float(r2)


def save_bundle_figure(results: dict[int, dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)

    for ax, L in zip(axes.flat, REPRESENTATIVE_LS):
        result = results[L]
        times = np.asarray(result["t"], dtype=float)
        curves = np.asarray(result["curves"][:NUM_BUNDLE], dtype=float)
        mean_curve = np.asarray(result["eps_summary"][OLD_EPS]["mu"], dtype=float)
        reference_curve = np.asarray(result["h"], dtype=float)
        old_summary = result["eps_summary"][OLD_EPS]
        new_summary = result["eps_summary"][NEW_EPS]

        for curve in curves:
            ax.semilogy(times, np.clip(curve, 1.0e-8, None), color="0.78", lw=0.85, alpha=0.45)
        ax.semilogy(times, np.clip(mean_curve, 1.0e-8, None), color="black", lw=2.3)
        ax.semilogy(times, np.clip(reference_curve, 1.0e-8, None), color="0.35", ls="--", lw=1.8)
        ax.axhline(OLD_EPS, color=OLD_DARK, lw=1.6)
        ax.axvline(float(old_summary["tstar"]), color=OLD_LIGHT, lw=1.4, ls=":")
        ax.axhline(NEW_EPS, color=NEW_DARK, lw=1.6)
        ax.axvline(float(new_summary["tstar"]), color=NEW_LIGHT, lw=1.4, ls="--")
        ax.set_title(f"L={L}")
        ax.set_xlabel("time")
        ax.set_ylim(1.0e-5, 2.2)

    axes[0, 0].set_ylabel(r"$g_t(\psi)$")
    axes[1, 0].set_ylabel(r"$g_t(\psi)$")

    handles = [
        Line2D([0], [0], color="0.78", lw=1.0, label="sample curves"),
        Line2D([0], [0], color="black", lw=2.3, label=r"$\mu_L(t)$"),
        Line2D([0], [0], color="0.35", lw=1.8, ls="--", label=r"$h_L(t)$"),
        Line2D([0], [0], color=OLD_DARK, lw=1.6, label=rf"$\epsilon={OLD_EPS:.2f}$"),
        Line2D([0], [0], color=OLD_LIGHT, lw=1.4, ls=":", label=rf"$t_*(\epsilon={OLD_EPS:.2f})$"),
        Line2D([0], [0], color=NEW_DARK, lw=1.6, label=rf"$\epsilon={NEW_EPS:.2f}$"),
        Line2D([0], [0], color=NEW_LIGHT, lw=1.4, ls="--", label=rf"$t_*(\epsilon={NEW_EPS:.2f})$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(
        FIG_DIR / "figure_open_gap_strong_skin_six_bundle_threshold_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def save_scaling_figure(results: dict[int, dict[str, object]]) -> None:
    L_values = np.array(SCALING_LS, dtype=float)
    q90_old = []
    tworst_old = []
    gap_old = []
    q90_new = []
    tworst_new = []
    gap_new = []

    for L in SCALING_LS:
        old_summary = results[L]["eps_summary"][OLD_EPS]
        new_summary = results[L]["eps_summary"][NEW_EPS]
        q90_old.append(float(np.nanquantile(old_summary["tmix"], 0.9)))
        tworst_old.append(float(old_summary["tworst"]))
        gap_old.append(float(old_summary["delta_t_act"]))
        q90_new.append(float(np.nanquantile(new_summary["tmix"], 0.9)))
        tworst_new.append(float(new_summary["tworst"]))
        gap_new.append(float(new_summary["delta_t_act"]))

    q90_old = np.asarray(q90_old, dtype=float)
    tworst_old = np.asarray(tworst_old, dtype=float)
    gap_old = np.asarray(gap_old, dtype=float)
    q90_new = np.asarray(q90_new, dtype=float)
    tworst_new = np.asarray(tworst_new, dtype=float)
    gap_new = np.asarray(gap_new, dtype=float)

    logL_old, fit_old, _, r2_old = fit_log_gap(L_values, gap_old)
    logL_new, fit_new, _, r2_new = fit_log_gap(L_values, gap_new)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0))

    axes[0].plot(
        L_values,
        tworst_old,
        color=OLD_DARK,
        marker="o",
        lw=2.3,
        label=rf"$t_{{\rm worst}}(\epsilon={OLD_EPS:.2f})$",
    )
    axes[0].plot(
        L_values,
        q90_old,
        color=OLD_LIGHT,
        marker="s",
        lw=2.3,
        label=rf"$q_{{0.9}}(t_{{\rm mix}})(\epsilon={OLD_EPS:.2f})$",
    )
    axes[0].plot(
        L_values,
        tworst_new,
        color=NEW_DARK,
        marker="o",
        lw=2.3,
        label=rf"$t_{{\rm worst}}(\epsilon={NEW_EPS:.2f})$",
    )
    axes[0].plot(
        L_values,
        q90_new,
        color=NEW_LIGHT,
        marker="s",
        lw=2.3,
        label=rf"$q_{{0.9}}(t_{{\rm mix}})(\epsilon={NEW_EPS:.2f})$",
    )
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("mixing time")
    axes[0].set_title("Worst and typical mixing scales")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(
        logL_old,
        gap_old,
        color=OLD_DARK,
        marker="o",
        lw=2.3,
        label=rf"$\Delta t_{{\rm act}}(\epsilon={OLD_EPS:.2f})$",
    )
    axes[1].plot(
        logL_old,
        fit_old,
        color=OLD_LIGHT,
        lw=2.0,
        ls="--",
        label=rf"fit at $\epsilon={OLD_EPS:.2f}$",
    )
    axes[1].plot(
        logL_new,
        gap_new,
        color=NEW_DARK,
        marker="o",
        lw=2.3,
        label=rf"$\Delta t_{{\rm act}}(\epsilon={NEW_EPS:.2f})$",
    )
    axes[1].plot(
        logL_new,
        fit_new,
        color=NEW_LIGHT,
        lw=2.0,
        ls="--",
        label=rf"fit at $\epsilon={NEW_EPS:.2f}$",
    )
    axes[1].set_xlabel(r"$\log L$")
    axes[1].set_ylabel(r"$t_{\rm worst}-q_{0.9}(t_{\rm mix})$")
    axes[1].set_title("Fitted separation scales")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].text(
        0.03,
        0.96,
        "\n".join(
            [
                rf"$\epsilon={OLD_EPS:.2f}$: $R^2={r2_old:.3f}$",
                rf"$\epsilon={NEW_EPS:.2f}$: $R^2={r2_new:.3f}$",
            ]
        ),
        transform=axes[1].transAxes,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "figure_open_gap_strong_skin_gap_scaling_threshold_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    results = {L: run_case(L) for L in SCALING_LS}
    save_bundle_figure(results)
    save_scaling_figure(results)


if __name__ == "__main__":
    main()
