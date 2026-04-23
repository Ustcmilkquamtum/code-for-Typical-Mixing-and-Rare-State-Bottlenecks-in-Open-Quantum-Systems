from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = {
    "samples": "#B5BBC2",
    "band": "#F3D2DA",
    "mean": "#B33F62",
    "reference": "#2E67A5",
    "threshold": "#B58800",
    "guide": "#6B6B6B",
    "box": "#A9C4D8",
}

LEGEND_FONT_SIZE = 12
SMALL_PLOT_LEGEND_FONT_SIZE = 11


@dataclass(frozen=True)
class DaviesModelConfig:
    n_qubits: int
    beta: float = 1.2
    omega: float = 1.0
    relaxation_rate: float = 1.0

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def ground_population(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self.beta * self.omega)))

    @property
    def excited_population(self) -> float:
        return float(1.0 - self.ground_population)

    @property
    def coherence_rate(self) -> float:
        return float(self.relaxation_rate / 2.0)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for operator in operators:
        out = np.kron(out, operator)
    return out


@lru_cache(maxsize=None)
def thermal_qubit_state(config: DaviesModelConfig) -> np.ndarray:
    return np.diag(
        [config.ground_population, config.excited_population]
    ).astype(complex)


@lru_cache(maxsize=None)
def stationary_state(config: DaviesModelConfig) -> np.ndarray:
    return kron_all([thermal_qubit_state(config) for _ in range(config.n_qubits)])


def reference_state(config: DaviesModelConfig) -> np.ndarray:
    return np.eye(config.dim, dtype=complex) / config.dim


def haar_random_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    return psi


def density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def trace_norm_hermitian(matrix: np.ndarray) -> float:
    hermitian = (matrix + matrix.conj().T) / 2.0
    return float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def time_grid(
    relaxation_rate: float,
    epsilon: float,
    num_times: int = 72,
    t_min: float = 1.0e-3,
    t_max: float | None = None,
) -> np.ndarray:
    if t_max is None:
        t_max = max(6.0, 2.8 * np.log(2.0 / epsilon) / max(relaxation_rate, 1.0e-9))
    return np.geomspace(t_min, float(t_max), num_times)


def crossing_time(times: np.ndarray, curve: np.ndarray, epsilon: float) -> float:
    below = np.flatnonzero(curve <= epsilon)
    if below.size == 0:
        return float("inf")
    idx = int(below[0])
    if idx == 0:
        return float(times[0])
    t0, t1 = float(times[idx - 1]), float(times[idx])
    y0, y1 = float(curve[idx - 1]), float(curve[idx])
    fraction = (epsilon - y0) / (y1 - y0)
    return float(np.exp(np.log(t0) + fraction * (np.log(t1) - np.log(t0))))


def local_slope_at_crossing(times: np.ndarray, curve: np.ndarray, epsilon: float) -> float:
    crossing = crossing_time(times, curve, epsilon)
    derivative = np.gradient(curve, times)
    return float(-np.interp(crossing, times, derivative))


def local_kraus_operators(
    config: DaviesModelConfig, delta_t: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    damping = float(1.0 - np.exp(-config.relaxation_rate * delta_t))
    survival = float(np.sqrt(max(0.0, 1.0 - damping)))
    p_ground = config.ground_population
    p_excited = config.excited_population
    return (
        np.sqrt(p_ground) * np.array([[1.0, 0.0], [0.0, survival]], dtype=complex),
        np.sqrt(p_ground) * np.array([[0.0, np.sqrt(damping)], [0.0, 0.0]], dtype=complex),
        np.sqrt(p_excited) * np.array([[survival, 0.0], [0.0, 1.0]], dtype=complex),
        np.sqrt(p_excited) * np.array([[0.0, 0.0], [np.sqrt(damping), 0.0]], dtype=complex),
    )


def step_kraus_sequence(
    times: np.ndarray, config: DaviesModelConfig
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    deltas = np.diff(np.concatenate(([0.0], np.asarray(times, dtype=float))))
    return [local_kraus_operators(config, float(delta_t)) for delta_t in deltas]


@lru_cache(maxsize=None)
def _permutation_data(
    n_qubits: int, qubit: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    permutation = tuple([qubit] + [idx for idx in range(n_qubits) if idx != qubit])
    inverse = tuple(np.argsort(permutation).tolist())
    return permutation, inverse


def _permute_density_matrix(rho: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    n_qubits = len(permutation)
    tensor = rho.reshape([2] * (2 * n_qubits))
    return np.transpose(
        tensor, axes=list(permutation) + [n_qubits + idx for idx in permutation]
    ).reshape(rho.shape)


def apply_local_channel(
    rho: np.ndarray,
    n_qubits: int,
    qubit: int,
    kraus_operators: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    permutation, inverse = _permutation_data(n_qubits, qubit)
    rho_permuted = _permute_density_matrix(rho, permutation)
    dim_rest = 2 ** (n_qubits - 1)
    tensor = rho_permuted.reshape(2, dim_rest, 2, dim_rest)
    out = np.zeros_like(tensor)
    for operator in kraus_operators:
        out += np.einsum("ac,cidj,bd->aibj", operator, tensor, operator.conj(), optimize=True)
    return _permute_density_matrix(out.reshape(rho.shape), inverse)


def relaxation_curve(
    rho0: np.ndarray,
    config: DaviesModelConfig,
    sigma: np.ndarray,
    times: np.ndarray,
    step_kraus: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    rho = np.array(rho0, copy=True)
    curve = np.empty_like(times, dtype=float)
    for idx, local_channel in enumerate(step_kraus):
        for qubit in range(config.n_qubits):
            rho = apply_local_channel(rho, config.n_qubits, qubit, local_channel)
        curve[idx] = trace_norm_hermitian(rho - sigma)
    return curve


def run_bundle_experiment(
    config: DaviesModelConfig,
    epsilon: float,
    num_samples: int = 48,
    num_times: int = 72,
    seed: int = 2026,
    t_max: float | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sigma = stationary_state(config)
    times = time_grid(
        relaxation_rate=config.relaxation_rate,
        epsilon=epsilon,
        num_times=num_times,
        t_max=t_max,
    )
    step_kraus = step_kraus_sequence(times, config)

    sample_curves = np.empty((num_samples, times.size), dtype=float)
    for idx in range(num_samples):
        rho0 = density_matrix(haar_random_state(config.dim, rng))
        sample_curves[idx] = relaxation_curve(rho0, config, sigma, times, step_kraus)

    mean_curve = np.mean(sample_curves, axis=0)
    band_lo, band_hi = np.quantile(sample_curves, [0.1, 0.9], axis=0)
    reference_curve = relaxation_curve(
        reference_state(config), config, sigma, times, step_kraus
    )
    sample_mixing = np.array(
        [crossing_time(times, curve, epsilon) for curve in sample_curves], dtype=float
    )

    summary = {
        "t_ref": crossing_time(times, reference_curve, epsilon),
        "t_typ": crossing_time(times, mean_curve, epsilon),
        "mixing_std": float(np.std(sample_mixing, ddof=1)),
        "mixing_iqr80": float(np.quantile(sample_mixing, 0.9) - np.quantile(sample_mixing, 0.1)),
        "slope_at_typ": local_slope_at_crossing(times, mean_curve, epsilon),
    }

    return {
        "config": config,
        "epsilon": float(epsilon),
        "times": times,
        "sample_curves": sample_curves,
        "mean_curve": mean_curve,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "reference_curve": reference_curve,
        "sample_mixing": sample_mixing,
        "summary": summary,
    }


def run_panel_study(
    sizes: list[int],
    epsilon: float,
    beta: float = 1.2,
    omega: float = 1.0,
    relaxation_rate: float = 1.0,
    num_samples: int = 48,
    num_times: int = 72,
    seed: int = 2026,
    t_max: float | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for offset, n_qubits in enumerate(sizes):
        config = DaviesModelConfig(
            n_qubits=n_qubits,
            beta=beta,
            omega=omega,
            relaxation_rate=relaxation_rate,
        )
        results.append(
            run_bundle_experiment(
                config=config,
                epsilon=epsilon,
                num_samples=num_samples,
                num_times=num_times,
                seed=seed + 97 * offset,
                t_max=t_max,
            )
        )
    return results


def summary_table(results: list[dict[str, object]]) -> str:
    headers = ["n", "d", "t_ref", "t_typ", "slope", "std[t_mix]", "q90-q10"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for result in results:
        config = result["config"]
        summary = result["summary"]
        lines.append(
            " | ".join(
                [
                    f"{config.n_qubits}",
                    f"{config.dim}",
                    f"{summary['t_ref']:.3f}",
                    f"{summary['t_typ']:.3f}",
                    f"{summary['slope_at_typ']:.3f}",
                    f"{summary['mixing_std']:.3f}",
                    f"{summary['mixing_iqr80']:.3f}",
                ]
            )
        )
    return "\n".join(lines)


def save_bundle_panel_figure(results: list[dict[str, object]], output_path: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    epsilon = float(results[0]["epsilon"])

    for axis, result in zip(axes_flat, results):
        _plot_bundle_panel(axis, result, epsilon)

    for axis in axes[:, 0]:
        axis.set_ylabel(r"$g_t(\psi)=\|\Lambda_t(\rho_\psi)-\sigma\|_1$")
    for axis in axes[-1, :]:
        axis.set_xlabel("time")
    for axis in axes_flat:
        axis.set_xlim(float(results[0]["times"][0]), float(results[0]["times"][-1]))
        axis.set_ylim(0.0, 2.05)

    handles = [
        Line2D([0], [0], color=COLORS["samples"], lw=1.2, label="Haar samples"),
        Line2D([0], [0], color=COLORS["mean"], lw=2.3, label="empirical mean"),
        Line2D([0], [0], color=COLORS["reference"], lw=2.0, ls="--", label="reference state"),
        Line2D([0], [0], color=COLORS["threshold"], lw=1.4, ls=":", label=rf"threshold $\epsilon={epsilon:.2f}$"),
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_bundle_panel(
    axis: plt.Axes, result: dict[str, object], epsilon: float
) -> None:
    config = result["config"]
    times = np.asarray(result["times"])
    sample_curves = np.asarray(result["sample_curves"])
    mean_curve = np.asarray(result["mean_curve"])
    band_lo = np.asarray(result["band_lo"])
    band_hi = np.asarray(result["band_hi"])
    reference_curve = np.asarray(result["reference_curve"])
    summary = result["summary"]

    for curve in sample_curves:
        axis.plot(times, curve, color=COLORS["samples"], lw=0.85, alpha=0.17)
    axis.fill_between(times, band_lo, band_hi, color=COLORS["band"], alpha=0.35)
    axis.plot(times, mean_curve, color=COLORS["mean"], lw=2.3)
    axis.plot(times, reference_curve, color=COLORS["reference"], lw=2.0, ls="--")
    axis.axhline(epsilon, color=COLORS["threshold"], lw=1.4, ls=":")
    axis.axvline(float(summary["t_typ"]), color=COLORS["mean"], lw=1.0, ls=":")
    axis.set_xscale("log")
    axis.set_title(rf"$n={config.n_qubits}$, $d={config.dim}$")
    axis.text(
        0.04,
        0.07,
        rf"$\sigma[t_{{\rm mix}}]={summary['mixing_std']:.3f}$"
        "\n"
        rf"$m_*(\epsilon)\approx {summary['slope_at_typ']:.3f}$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )


def save_mixing_concentration_figure(
    results: list[dict[str, object]], output_path: Path
) -> None:
    configure_matplotlib()
    mixing_fig_label_size = 20
    mixing_fig_tick_size = 19
    mixing_fig_legend_size = 19
    epsilon = float(results[0]["epsilon"])
    dims = np.array([result["config"].dim for result in results], dtype=float)
    labels = [rf"$d={int(dim)}$" for dim in dims]
    mixing_samples = [np.asarray(result["sample_mixing"]) for result in results]
    t_ref = np.array([result["summary"]["t_ref"] for result in results], dtype=float)
    t_typ = np.array([result["summary"]["t_typ"] for result in results], dtype=float)
    spread_std = np.array(
        [result["summary"]["mixing_std"] for result in results], dtype=float
    )
    spread_iqr = np.array(
        [result["summary"]["mixing_iqr80"] for result in results], dtype=float
    )
    slopes = np.array(
        [result["summary"]["slope_at_typ"] for result in results], dtype=float
    )
    guide = spread_std[0] * (slopes[0] / slopes) * np.sqrt(dims[0] / dims)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 10.8))

    boxplot = axes[0].boxplot(
        mixing_samples,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": COLORS["mean"], "linewidth": 1.8},
        whiskerprops={"color": COLORS["guide"], "linewidth": 1.2},
        capprops={"color": COLORS["guide"], "linewidth": 1.2},
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(COLORS["box"])
        patch.set_edgecolor("none")
        patch.set_alpha(0.85)

    positions = np.arange(1, len(results) + 1)
    axes[0].plot(positions, t_typ, color=COLORS["mean"], marker="o", lw=2.0, label=r"$t_{\rm typ}$")
    axes[0].plot(
        positions,
        t_ref,
        color=COLORS["reference"],
        marker="s",
        lw=1.9,
        ls="--",
        label=r"$t_{\rm ref}$",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, fontsize=mixing_fig_tick_size)
    axes[0].set_ylabel(r"$t_{\mathrm{mix}}(\psi,\epsilon)$", fontsize=mixing_fig_label_size)
    axes[0].legend(loc="upper right", fontsize=mixing_fig_legend_size)
    axes[0].tick_params(axis="both", labelsize=mixing_fig_tick_size)

    axes[1].plot(dims, spread_std, color=COLORS["mean"], marker="o", lw=2.1, label="sample std")
    axes[1].plot(
        dims,
        spread_iqr,
        color="#6E4AA1",
        marker="D",
        lw=2.0,
        label=r"$q_{0.9}-q_{0.1}$",
    )
    axes[1].plot(
        dims,
        guide,
        color=COLORS["guide"],
        lw=1.5,
        ls=":",
        label=r"$\propto (m_*\sqrt{d})^{-1}$ guide",
    )
    axes[1].set_xticks(dims)
    axes[1].set_xlabel("Hilbert-space dimension d", fontsize=mixing_fig_label_size)
    axes[1].set_ylabel("spread of mixing times", fontsize=mixing_fig_label_size)
    axes[1].legend(loc="upper right", fontsize=mixing_fig_legend_size)
    axes[1].tick_params(axis="both", labelsize=mixing_fig_tick_size)

    fig.tight_layout(h_pad=3.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_bundle_and_mixing_mosaic_figure(
    results: list[dict[str, object]], output_path: Path
) -> None:
    configure_matplotlib()
    epsilon = float(results[0]["epsilon"])
    dims = np.array([result["config"].dim for result in results], dtype=float)
    labels = [rf"$d={int(dim)}$" for dim in dims]
    mixing_samples = [np.asarray(result["sample_mixing"]) for result in results]
    t_ref = np.array([result["summary"]["t_ref"] for result in results], dtype=float)
    t_typ = np.array([result["summary"]["t_typ"] for result in results], dtype=float)
    spread_std = np.array(
        [result["summary"]["mixing_std"] for result in results], dtype=float
    )
    spread_iqr = np.array(
        [result["summary"]["mixing_iqr80"] for result in results], dtype=float
    )
    slopes = np.array(
        [result["summary"]["slope_at_typ"] for result in results], dtype=float
    )
    guide = spread_std[0] * (slopes[0] / slopes) * np.sqrt(dims[0] / dims)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.5, 9.2),
        gridspec_kw={"width_ratios": [1.55, 1.55, 1.0]},
        constrained_layout=True,
    )
    flat_axes = axes.flatten()

    _plot_bundle_panel(flat_axes[0], results[0], epsilon)
    _plot_bundle_panel(flat_axes[1], results[1], epsilon)
    _plot_bundle_panel(flat_axes[3], results[2], epsilon)
    _plot_bundle_panel(flat_axes[4], results[3], epsilon)
    for axis in (flat_axes[0], flat_axes[1], flat_axes[2]):
        axis.tick_params(labelbottom=False)

    positions = np.arange(1, len(results) + 1)
    boxplot = flat_axes[2].boxplot(
        mixing_samples,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": COLORS["mean"], "linewidth": 1.8},
        whiskerprops={"color": COLORS["guide"], "linewidth": 1.2},
        capprops={"color": COLORS["guide"], "linewidth": 1.2},
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(COLORS["box"])
        patch.set_edgecolor("none")
        patch.set_alpha(0.85)

    flat_axes[2].plot(
        positions, t_typ, color=COLORS["mean"], marker="o", lw=2.0, label=r"$t_{\rm typ}$"
    )
    flat_axes[2].plot(
        positions,
        t_ref,
        color=COLORS["reference"],
        marker="s",
        lw=1.9,
        ls="--",
        label=r"$t_{\rm ref}$",
    )
    flat_axes[2].set_xticks(positions)
    flat_axes[2].set_xticklabels(labels)
    flat_axes[2].set_ylabel(r"$t_{\mathrm{mix}}(\psi,\epsilon)$")
    flat_axes[2].set_xlabel("Hilbert-space dimension d")
    flat_axes[2].legend(loc="upper right", fontsize=SMALL_PLOT_LEGEND_FONT_SIZE)

    flat_axes[5].plot(
        dims, spread_std, color=COLORS["mean"], marker="o", lw=2.1, label="sample std"
    )
    flat_axes[5].plot(
        dims,
        spread_iqr,
        color="#6E4AA1",
        marker="D",
        lw=2.0,
        label=r"$q_{0.9}-q_{0.1}$",
    )
    flat_axes[5].plot(
        dims,
        guide,
        color=COLORS["guide"],
        lw=1.5,
        ls=":",
        label=r"$\propto (m_*\sqrt{d})^{-1}$ guide",
    )
    flat_axes[5].set_xticks(dims)
    flat_axes[5].set_xlabel("Hilbert-space dimension d")
    flat_axes[5].set_ylabel("spread of mixing times")
    flat_axes[5].legend(loc="upper right", fontsize=SMALL_PLOT_LEGEND_FONT_SIZE)

    for axis in (flat_axes[0], flat_axes[1], flat_axes[3], flat_axes[4]):
        axis.set_xlim(float(results[0]["times"][0]), float(results[0]["times"][-1]))
    for axis in (flat_axes[0], flat_axes[1], flat_axes[3], flat_axes[4]):
        axis.set_ylim(0.0, 2.05)
    for axis in (flat_axes[3], flat_axes[4]):
        axis.set_xlabel("time")
    flat_axes[5].set_xlim(dims[0], dims[-1])
    flat_axes[2].set_xlim(min(positions), max(positions))

    for axis in (flat_axes[0], flat_axes[3]):
        axis.set_ylabel(r"$g_t(\psi)=\|\Lambda_t(\rho_\psi)-\sigma\|_1$")

    handles = [
        Line2D([0], [0], color=COLORS["samples"], lw=1.2, label="Haar samples"),
        Line2D([0], [0], color=COLORS["mean"], lw=2.3, label="empirical mean"),
        Line2D([0], [0], color=COLORS["reference"], lw=2.0, ls="--", label="reference state"),
        Line2D([0], [0], color=COLORS["threshold"], lw=1.4, ls=":", label=rf"threshold $\epsilon={epsilon:.2f}$"),
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _run_default_simulation() -> None:
    output_dir = ensure_output_dir(Path(__file__).resolve().parent / "outputs")
    results = run_panel_study(
        sizes=[3, 4, 5, 6],
        epsilon=0.70,
        beta=1.2,
        omega=1.0,
        relaxation_rate=1.0,
        num_samples=48,
        num_times=72,
        seed=2026,
        t_max=6.0,
    )
    save_bundle_panel_figure(results, output_dir / "davies_many_body_bundle_panels.pdf")
    save_mixing_concentration_figure(
        results, output_dir / "davies_many_body_mixing_concentration.pdf"
    )


if __name__ == "__main__":
    _run_default_simulation()
