from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boundary_davies import run_boundary_bundle_only
from boundary_davies import local_gap_and_left_mode
from metrics import crossing_time
from sampling import sample_boundary_overlaps


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "outputs" / "boundary_davies" / "figures"
DATA_DIR = ROOT / "outputs" / "boundary_davies" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


PANEL_LS = [4, 5, 6, 7, 8, 9]
EPS = 0.25
LAMBDA_S = 0.25
NUM_SAMPLES = 10
NUM_TIMES = 40
RNG = np.random.default_rng(52032)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def thermal_populations(beta: float = 1.2, omega: float = 1.0) -> tuple[float, float]:
    p_ground = 1.0 / (1.0 + np.exp(-beta * omega))
    p_excited = 1.0 - p_ground
    return p_ground, p_excited


def local_distribution(start_bit: int, rate: float, times: np.ndarray) -> np.ndarray:
    p_ground, p_excited = thermal_populations()
    eq_excited = p_excited
    excited_t = eq_excited + (float(start_bit) - eq_excited) * np.exp(-rate * times)
    return np.stack([1.0 - excited_t, excited_t], axis=1)


def basis_index_to_bits(index: int, L: int) -> list[int]:
    return [(index >> shift) & 1 for shift in range(L - 1, -1, -1)]


def diagonal_basis_worst_curve(L: int, lambda_s: float, times: np.ndarray) -> tuple[np.ndarray, float]:
    p_ground, p_excited = thermal_populations()
    sigma_local = np.array([p_ground, p_excited], dtype=float)
    sigma_prob = sigma_local.copy()
    for _ in range(L - 1):
        sigma_prob = np.kron(sigma_prob, sigma_local)

    local_cache: dict[tuple[int, float], np.ndarray] = {
        (0, lambda_s): local_distribution(0, lambda_s, times),
        (1, lambda_s): local_distribution(1, lambda_s, times),
        (0, 1.0): local_distribution(0, 1.0, times),
        (1, 1.0): local_distribution(1, 1.0, times),
    }

    best_curve = None
    best_t = -np.inf
    for basis_index in range(2 ** L):
        bits = basis_index_to_bits(basis_index, L)
        curve = np.empty(len(times), dtype=float)
        for tidx in range(len(times)):
            prob = local_cache[(bits[0], lambda_s)][tidx]
            for bit in bits[1:]:
                prob = np.kron(prob, local_cache[(bit, 1.0)][tidx])
            curve[tidx] = np.abs(prob - sigma_prob).sum()
        t_cross = float(crossing_time(curve, times, EPS))
        if t_cross > best_t:
            best_t = t_cross
            best_curve = curve
    return best_curve, best_t


def build_exact_plot_rows(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in results:
        tmix = np.array([crossing_time(c, result["times"], EPS) for c in result["curves"]], dtype=float)
        worst_curve, t_worst_basis = diagonal_basis_worst_curve(result["L"], result["lambda_s"], result["times"])
        rows.append(
            {
                "L": result["L"],
                "lambda_s": result["lambda_s"],
                "tstar": result["tstar"],
                "t_ref": float(crossing_time(result["h"], result["times"], EPS)),
                "q90_tmix": float(np.nanquantile(tmix, 0.9)),
                "t_worst_basis": float(t_worst_basis),
                "delta_t_act_basis": float(t_worst_basis - np.nanquantile(tmix, 0.9)),
            }
        )
    df = pd.DataFrame(rows).sort_values("L").reset_index(drop=True)
    df.to_csv(DATA_DIR / "open_gap_constant_exact_L4_L9.csv", index=False)
    return df


def build_prediction_rows() -> pd.DataFrame:
    rng = np.random.default_rng(62032)
    rows = []
    for L in PANEL_LS:
        Delta, A_B = local_gap_and_left_mode(LAMBDA_S, beta=1.2, omega=1.0, gphi=0.5)
        overlaps = sample_boundary_overlaps(dim=2 ** L, n_samples=4000, rng=rng, A_B=A_B)
        alpha_typ = float(np.quantile(np.abs(overlaps), 0.9))
        rows.append(
            {
                "L": L,
                "lambda_s": LAMBDA_S,
                "Delta": Delta,
                "alpha_typ": alpha_typ,
                "delta_t_pred": float(np.log(max(np.linalg.norm(A_B, ord=2), 1.0e-14) / max(alpha_typ, 1.0e-14)) / max(Delta, 1.0e-14)),
                "guide_linear": float(L * np.log(2.0) / (2.0 * max(Delta, 1.0e-12))),
            }
        )
    df = pd.DataFrame(rows).sort_values("L").reset_index(drop=True)
    df.to_csv(DATA_DIR / "open_gap_constant_pred_L4_L9.csv", index=False)
    return df


def make_bundle_panels() -> None:
    results = []
    for L in PANEL_LS:
        print(f"computing open-gap bundle panel for L={L}")
        results.append(
            run_boundary_bundle_only(
                L=L,
                lambda_s=LAMBDA_S,
                eps=EPS,
                rng=RNG,
                beta=1.2,
                omega=1.0,
                gphi=0.5,
                num_samples=NUM_SAMPLES,
                num_times=NUM_TIMES,
            )
        )

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), sharex=False, sharey=True)
    axes = axes.flatten()
    for ax, result in zip(axes, results):
        for curve in result["curves"]:
            ax.semilogy(result["times"], np.clip(curve, 1.0e-6, None), color="0.77", lw=0.9, alpha=0.42)
        ax.semilogy(result["times"], np.clip(result["mu"], 1.0e-6, None), color="#b33f62", lw=2.4)
        ax.semilogy(result["times"], np.clip(result["h"], 1.0e-6, None), color="#2e67a5", lw=2.0, ls="--")
        ax.axhline(result["eps"], color="#b58800", lw=1.3)
        ax.axvline(result["tstar"], color="#b33f62", lw=1.2, ls=":")
        ax.set_title(f"L={result['L']}, d={2 ** result['L']}")
        ax.set_xlabel("time")
        ax.set_ylim(1.0e-4, 2.2)

    axes[0].set_ylabel(r"$g_L(t,\psi)$")
    axes[3].set_ylabel(r"$g_L(t,\psi)$")
    fig.suptitle(rf"Constant-$\lambda_s$ relaxation bundles with $\lambda_s(L)=0.25$ and $\epsilon={EPS:.2f}$", y=1.02)
    fig.tight_layout()
    save(fig, "open_gap_constant_bundle_3x2")
    return results


def make_linear_separation(exact: pd.DataFrame, pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.plot(
        exact["L"],
        exact["delta_t_act_basis"],
        marker="o",
        lw=2.2,
        color="#111111",
        label=r"$t_{\rm worst}^{\rm basis} - q_{0.9}(t_{\rm mix})$",
    )
    ax.plot(
        pred["L"],
        pred["delta_t_pred"],
        marker="s",
        lw=2.0,
        color="#b33f62",
        label=r"$\Delta t_{\rm pred}^{(\delta)}$",
    )
    ax.plot(
        pred["L"],
        pred["guide_linear"],
        lw=1.9,
        ls="--",
        color="#2e67a5",
        label=r"$L \log 2 / (2\Delta_L)$",
    )

    if len(exact) >= 2:
        coeff = np.polyfit(exact["L"], exact["delta_t_act_basis"], 1)
        fit_x = np.linspace(exact["L"].min(), exact["L"].max(), 100)
        fit_y = np.polyval(coeff, fit_x)
        ax.plot(
            fit_x,
            fit_y,
            lw=1.6,
            ls=":",
            color="#555555",
            label=rf"fit to exact points: {coeff[0]:.2f} L + {coeff[1]:.2f}",
        )

    ax.set_xlabel("L")
    ax.set_ylabel("time difference")
    ax.set_title(rf"Typical-versus-worst separation in the constant-$\lambda_s$ branch, $\epsilon={EPS:.2f}$")
    ax.legend()
    fig.tight_layout()
    save(fig, "open_gap_constant_linear_separation")


def main() -> None:
    results = make_bundle_panels()
    exact = build_exact_plot_rows(results)
    pred = build_prediction_rows()
    make_linear_separation(exact, pred)
    print("saved:")
    for stem in (
        "open_gap_constant_bundle_3x2",
        "open_gap_constant_linear_separation",
    ):
        print(f" - {stem}.png")
        print(f" - {stem}.pdf")


if __name__ == "__main__":
    main()
