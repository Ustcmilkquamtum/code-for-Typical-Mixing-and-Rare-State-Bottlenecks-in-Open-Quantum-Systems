from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "gray": "#A7A9AC",
    "mean": "#B03A48",
    "band": "#F3B0B8",
    "reference": "#2E5EAA",
    "worst": "#111111",
    "fast": "#2E8B57",
    "threshold": "#BC8F00",
    "guide": "#6E6E6E",
    "logical": "#8C5A2B",
    "syndrome": "#0B7189",
}


@dataclass(frozen=True)
class ModelConfig:
    n_syndrome: int
    beta: float = 1.8
    leakage_scale: float = 0.75
    syndrome_rate: float = 2.0

    @property
    def n_qubits(self) -> int:
        return 1 + self.n_syndrome

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def leakage_rate(self) -> float:
        return float(np.exp(-self.leakage_scale * self.n_syndrome))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 10,
        }
    )


def thermal_qubit_state(beta: float) -> np.ndarray:
    excited = np.exp(-beta) / (1.0 + np.exp(-beta))
    return np.diag([1.0 - excited, excited]).astype(complex)


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for op in operators:
        out = np.kron(out, op)
    return out


@lru_cache(maxsize=None)
def stationary_state(config: ModelConfig) -> np.ndarray:
    return kron_all(
        [np.eye(2, dtype=complex) / 2.0]
        + [thermal_qubit_state(config.beta) for _ in range(config.n_syndrome)]
    )


def reference_state(config: ModelConfig) -> np.ndarray:
    return np.eye(config.dim, dtype=complex) / config.dim


def haar_random_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    return psi


def density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def basis_state(bits: list[int]) -> np.ndarray:
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    state = np.zeros(2 ** len(bits), dtype=complex)
    state[index] = 1.0
    return state


def worst_state(config: ModelConfig) -> np.ndarray:
    return density_matrix(basis_state([0] * config.n_qubits))


def fast_state(config: ModelConfig) -> np.ndarray:
    if config.n_syndrome == 0:
        return density_matrix(np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0))
    zero = basis_state([0] * config.n_qubits)
    bell_like = basis_state([1, 1] + [0] * (config.n_qubits - 2))
    return density_matrix((zero + bell_like) / np.sqrt(2.0))


def trace_norm_hermitian(matrix: np.ndarray) -> float:
    hermitian = (matrix + matrix.conj().T) / 2.0
    return float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def single_qubit_reduced_state(
    rho: np.ndarray, n_qubits: int, qubit: int
) -> np.ndarray:
    perm = [qubit] + [idx for idx in range(n_qubits) if idx != qubit]
    tensor = rho.reshape([2] * (2 * n_qubits))
    permuted = np.transpose(
        tensor, axes=perm + [n_qubits + idx for idx in perm]
    ).reshape(2, 2 ** (n_qubits - 1), 2, 2 ** (n_qubits - 1))
    return np.trace(permuted, axis1=1, axis2=3)


def logical_distance(rho: np.ndarray, config: ModelConfig) -> float:
    reduced = single_qubit_reduced_state(rho, config.n_qubits, 0)
    return trace_norm_hermitian(reduced - np.eye(2) / 2.0)


@lru_cache(maxsize=None)
def _permutation_data(n_qubits: int, qubit: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    perm = tuple([qubit] + [idx for idx in range(n_qubits) if idx != qubit])
    inverse = tuple(np.argsort(perm).tolist())
    return perm, inverse


def _permute_density_matrix(rho: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    n_qubits = len(perm)
    tensor = rho.reshape([2] * (2 * n_qubits))
    return np.transpose(
        tensor, axes=list(perm) + [n_qubits + idx for idx in perm]
    ).reshape(rho.shape)


def replace_qubit_with_state(
    rho: np.ndarray, n_qubits: int, qubit: int, local_state: np.ndarray
) -> np.ndarray:
    perm, inverse = _permutation_data(n_qubits, qubit)
    rho_perm = _permute_density_matrix(rho, perm)
    dim_rest = 2 ** (n_qubits - 1)
    tensor = rho_perm.reshape(2, dim_rest, 2, dim_rest)
    reduced = tensor[0, :, 0, :] + tensor[1, :, 1, :]
    replaced = np.kron(local_state, reduced)
    return _permute_density_matrix(replaced, inverse)


def evolve_density_matrix(
    rho0: np.ndarray, t: float, config: ModelConfig
) -> np.ndarray:
    n_qubits = config.n_qubits
    rho = np.array(rho0, copy=True)

    logical_factor = np.exp(-config.leakage_rate * t)
    logical_target = np.eye(2, dtype=complex) / 2.0
    rho = logical_factor * rho + (1.0 - logical_factor) * replace_qubit_with_state(
        rho, n_qubits, 0, logical_target
    )

    syndrome_factor = np.exp(-config.syndrome_rate * t)
    syndrome_target = thermal_qubit_state(config.beta)
    for qubit in range(1, n_qubits):
        rho = syndrome_factor * rho + (
            1.0 - syndrome_factor
        ) * replace_qubit_with_state(rho, n_qubits, qubit, syndrome_target)

    return rho


def relaxation_curve(
    rho0: np.ndarray, times: np.ndarray, config: ModelConfig, sigma: np.ndarray
) -> np.ndarray:
    return np.array(
        [trace_norm_hermitian(evolve_density_matrix(rho0, t, config) - sigma) for t in times]
    )


def crossing_time(times: np.ndarray, curve: np.ndarray, epsilon: float) -> float:
    below = curve <= epsilon
    if not np.any(below):
        return float("inf")
    idx = int(np.argmax(below))
    if idx == 0:
        return float(times[0])
    t0, t1 = float(times[idx - 1]), float(times[idx])
    y0, y1 = float(curve[idx - 1]), float(curve[idx])
    fraction = (epsilon - y0) / (y1 - y0)
    return float(np.exp(np.log(t0) + fraction * (np.log(t1) - np.log(t0))))


def time_grid(config: ModelConfig, epsilon: float, num_times: int) -> np.ndarray:
    t_max = max(
        12.0,
        1.3 * np.log(2.0 / epsilon) / max(config.leakage_rate, 1e-9),
    )
    return np.geomspace(1.0e-3, t_max, num_times)


def run_bundle_experiment(
    config: ModelConfig,
    epsilon: float,
    num_samples: int = 64,
    num_times: int = 96,
    seed: int = 1234,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sigma = stationary_state(config)
    times = time_grid(config, epsilon, num_times)

    sample_curves = np.empty((num_samples, num_times), dtype=float)
    logical_distances = np.empty(num_samples, dtype=float)

    for idx in range(num_samples):
        rho0 = density_matrix(haar_random_state(config.dim, rng))
        logical_distances[idx] = logical_distance(rho0, config)
        sample_curves[idx] = relaxation_curve(rho0, times, config, sigma)

    mean_curve = np.mean(sample_curves, axis=0)
    band_lo, band_hi = np.quantile(sample_curves, [0.1, 0.9], axis=0)

    reference_curve = relaxation_curve(reference_state(config), times, config, sigma)
    worst_curve = relaxation_curve(worst_state(config), times, config, sigma)
    fast_curve = relaxation_curve(fast_state(config), times, config, sigma)

    sample_mixing = np.array(
        [crossing_time(times, curve, epsilon) for curve in sample_curves], dtype=float
    )

    summary = {
        "t_ref": crossing_time(times, reference_curve, epsilon),
        "t_mean": crossing_time(times, mean_curve, epsilon),
        "t_q50": float(np.quantile(sample_mixing, 0.5)),
        "t_q90": float(np.quantile(sample_mixing, 0.9)),
        "t_fast": crossing_time(times, fast_curve, epsilon),
        "t_worst": crossing_time(times, worst_curve, epsilon),
        "logical_q50": float(np.quantile(logical_distances, 0.5)),
        "logical_q90": float(np.quantile(logical_distances, 0.9)),
    }

    return {
        "config": config,
        "epsilon": epsilon,
        "times": times,
        "sample_curves": sample_curves,
        "mean_curve": mean_curve,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "reference_curve": reference_curve,
        "worst_curve": worst_curve,
        "fast_curve": fast_curve,
        "sample_mixing": sample_mixing,
        "logical_distances": logical_distances,
        "summary": summary,
    }


def run_scaling_study(
    sizes: list[int],
    epsilon: float,
    num_samples: int = 36,
    num_times: int = 88,
    seed: int = 2026,
    beta: float = 1.8,
    leakage_scale: float = 0.75,
    syndrome_rate: float = 2.0,
) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for offset, size in enumerate(sizes):
        config = ModelConfig(
            n_syndrome=size,
            beta=beta,
            leakage_scale=leakage_scale,
            syndrome_rate=syndrome_rate,
        )
        result = run_bundle_experiment(
            config=config,
            epsilon=epsilon,
            num_samples=num_samples,
            num_times=num_times,
            seed=seed + 97 * offset,
        )
        summary = result["summary"]
        records.append(
            {
                "n_syndrome": float(size),
                "dim": float(config.dim),
                "leakage_rate": float(config.leakage_rate),
                "t_ref": float(summary["t_ref"]),
                "t_mean": float(summary["t_mean"]),
                "t_q50": float(summary["t_q50"]),
                "t_q90": float(summary["t_q90"]),
                "t_fast": float(summary["t_fast"]),
                "t_worst": float(summary["t_worst"]),
                "logical_q50": float(summary["logical_q50"]),
                "logical_q90": float(summary["logical_q90"]),
            }
        )
    return records


def alpha_theory(n_syndrome: int, delta: float) -> float:
    dim_syn = 2 ** n_syndrome
    return float(np.sqrt(3.0 / (delta * (2 * dim_syn + 1))))


def save_schematic(config: ModelConfig, output_path: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.set_axis_off()

    logical_box = FancyBboxPatch(
        (0.05, 0.34),
        0.18,
        0.28,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=COLORS["logical"],
        edgecolor="none",
        alpha=0.95,
        transform=ax.transAxes,
    )
    ax.add_patch(logical_box)
    ax.text(
        0.14,
        0.48,
        "Logical qubit\n(encoded sector)",
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        transform=ax.transAxes,
    )

    start_x = 0.34
    width = 0.08
    gap = 0.025
    for idx in range(config.n_syndrome):
        x = start_x + idx * (width + gap)
        if x + width > 0.93:
            break
        syndrome_box = FancyBboxPatch(
            (x, 0.36),
            width,
            0.24,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=COLORS["syndrome"],
            edgecolor="none",
            alpha=0.95,
            transform=ax.transAxes,
        )
        ax.add_patch(syndrome_box)
        ax.text(
            x + width / 2,
            0.48,
            rf"$s_{{{idx + 1}}}$",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            transform=ax.transAxes,
        )
        arrow = FancyArrowPatch(
            (x + width / 2, 0.82),
            (x + width / 2, 0.64),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.4,
            color=COLORS["syndrome"],
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)

    logical_arrow = FancyArrowPatch(
        (0.14, 0.83),
        (0.14, 0.65),
        arrowstyle="-|>",
        mutation_scale=11,
        lw=1.5,
        color=COLORS["logical"],
        transform=ax.transAxes,
    )
    ax.add_patch(logical_arrow)

    ax.text(
        0.14,
        0.89,
        rf"Weak logical leakage: $\eta_L = e^{{-{config.leakage_scale:.2f} L}}$",
        ha="center",
        va="bottom",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.63,
        0.89,
        rf"Fast local thermalizers: $\mathcal{{R}}_j^{{(\beta)}}$ with $\beta={config.beta:.1f}$",
        ha="center",
        va="bottom",
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.14,
        r"$\sigma_L = \frac{\mathbb{I}_{\mathrm{log}}}{2} \otimes \tau_\beta^{\otimes L}$"
        "\n"
        r"Local jump operators keep the syndrome fast and the logical sector slow.",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_bundle_figure(result: dict[str, object], output_path: Path) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.6, 5.8))

    times = np.asarray(result["times"])
    epsilon = float(result["epsilon"])
    sample_curves = np.asarray(result["sample_curves"])
    mean_curve = np.asarray(result["mean_curve"])
    band_lo = np.asarray(result["band_lo"])
    band_hi = np.asarray(result["band_hi"])
    reference_curve = np.asarray(result["reference_curve"])
    worst_curve = np.asarray(result["worst_curve"])
    fast_curve = np.asarray(result["fast_curve"])
    config = result["config"]
    summary = result["summary"]

    for idx, curve in enumerate(sample_curves):
        ax.plot(
            times,
            curve,
            color=COLORS["gray"],
            lw=0.9,
            alpha=0.22 if idx < 24 else 0.12,
        )

    ax.fill_between(times, band_lo, band_hi, color=COLORS["band"], alpha=0.35)
    ax.plot(times, mean_curve, color=COLORS["mean"], lw=2.7)
    ax.plot(times, reference_curve, color=COLORS["reference"], lw=2.2, ls="--")
    ax.plot(times, worst_curve, color=COLORS["worst"], lw=2.5)
    ax.plot(times, fast_curve, color=COLORS["fast"], lw=2.4)
    ax.axhline(epsilon, color=COLORS["threshold"], lw=1.6, ls=":")

    markers = [
        ("$t_{0.9}$", float(summary["t_q90"]), COLORS["mean"]),
        ("$t_{\\mathrm{worst}}$", float(summary["t_worst"]), COLORS["worst"]),
    ]
    for label, xloc, color in markers:
        ax.axvline(xloc, color=color, lw=1.3, ls=":", alpha=0.85)
        ax.text(
            xloc,
            1.83 if "worst" in label else 1.55,
            label,
            rotation=90,
            ha="right",
            va="top",
            color=color,
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0.0, 2.05)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$g_t(\psi)=\|\Lambda_t(\rho_\psi)-\sigma_L\|_1$")
    ax.set_title(
        "Microscopic logical-sector Lindbladian: typical bundle vs rare slow tail"
    )

    text = (
        rf"$L={config.n_syndrome}$ syndrome qubits, $d={config.dim}$, "
        rf"$\eta_L={config.leakage_rate:.3f}$, $\epsilon={epsilon:.2f}$"
        "\n"
        rf"$t_{{\mathrm{{mean}}}}={summary['t_mean']:.2f}$, "
        rf"$t_{{0.9}}={summary['t_q90']:.2f}$, "
        rf"$t_{{\mathrm{{worst}}}}={summary['t_worst']:.2f}$"
    )
    ax.text(
        0.03,
        0.06,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    legend_handles = [
        Line2D([0], [0], color=COLORS["gray"], lw=1.2, alpha=0.7, label="Haar samples"),
        Line2D([0], [0], color=COLORS["mean"], lw=2.7, label="empirical mean"),
        Line2D([0], [0], color=COLORS["reference"], lw=2.2, ls="--", label="reference state"),
        Line2D([0], [0], color=COLORS["worst"], lw=2.5, label="slow logical state"),
        Line2D([0], [0], color=COLORS["fast"], lw=2.4, label="logical-neutral fast state"),
        Line2D([0], [0], color=COLORS["threshold"], lw=1.6, ls=":", label="threshold"),
    ]
    ax.legend(handles=legend_handles, ncols=2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_scaling_figure(records: list[dict[str, float]], output_path: Path) -> None:
    configure_matplotlib()
    sizes = np.array([record["n_syndrome"] for record in records], dtype=float)
    t_ref = np.array([record["t_ref"] for record in records])
    t_mean = np.array([record["t_mean"] for record in records])
    t_q90 = np.array([record["t_q90"] for record in records])
    t_fast = np.array([record["t_fast"] for record in records])
    t_worst = np.array([record["t_worst"] for record in records])
    gaps_mean = t_worst - t_mean
    gaps_q90 = t_worst - t_q90

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))

    axes[0].semilogy(sizes, t_fast, marker="o", color=COLORS["fast"], lw=2, label="fast state")
    axes[0].semilogy(
        sizes, t_ref, marker="s", color=COLORS["reference"], lw=2, ls="--", label="reference"
    )
    axes[0].semilogy(sizes, t_mean, marker="o", color=COLORS["mean"], lw=2, label="mean crossing")
    axes[0].semilogy(
        sizes, t_q90, marker="D", color="#7B4EA3", lw=2, label="90% quantile"
    )
    axes[0].semilogy(sizes, t_worst, marker="o", color=COLORS["worst"], lw=2.4, label="worst state")
    axes[0].set_xlabel("number of syndrome qubits L")
    axes[0].set_ylabel("mixing time")
    axes[0].set_title("Crossing times")
    axes[0].legend(loc="upper left")

    guide = gaps_q90[0] * np.exp(0.75 * (sizes - sizes[0]))
    axes[1].semilogy(
        sizes, gaps_mean, marker="o", color=COLORS["mean"], lw=2, label=r"$t_{\mathrm{worst}}-t_{\mathrm{mean}}$"
    )
    axes[1].semilogy(
        sizes, gaps_q90, marker="D", color="#7B4EA3", lw=2, label=r"$t_{\mathrm{worst}}-t_{0.9}$"
    )
    axes[1].semilogy(
        sizes, guide, color=COLORS["guide"], lw=1.4, ls=":", label=r"$\propto e^{0.75L}$ guide"
    )
    axes[1].set_xlabel("number of syndrome qubits L")
    axes[1].set_ylabel("gap")
    axes[1].set_title("Worst-minus-typical separation")
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.set_xticks(sizes)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_logical_overlap_figure(
    records: list[dict[str, float]], output_path: Path, delta: float = 0.1
) -> None:
    configure_matplotlib()
    sizes = np.array([record["n_syndrome"] for record in records], dtype=float)
    logical_q50 = np.array([record["logical_q50"] for record in records])
    logical_q90 = np.array([record["logical_q90"] for record in records])
    theory = np.array([alpha_theory(int(size), delta) for size in sizes])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogy(
        sizes,
        logical_q50,
        marker="o",
        color=COLORS["mean"],
        lw=2.1,
        label="Haar median",
    )
    ax.semilogy(
        sizes,
        logical_q90,
        marker="D",
        color="#7B4EA3",
        lw=2.1,
        label="Haar 90% quantile",
    )
    ax.semilogy(
        sizes,
        theory,
        color=COLORS["guide"],
        lw=1.5,
        ls=":",
        label=rf"$\alpha_L(\delta={delta:.1f})$ guide",
    )
    ax.set_xticks(sizes)
    ax.set_xlabel("number of syndrome qubits L")
    ax.set_ylabel(r"$\|\rho_{\mathrm{log}}-\mathbb{I}/2\|_1$")
    ax.set_title("Typical logical overlap is exponentially suppressed")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_bundle(result: dict[str, object]) -> dict[str, float]:
    summary = dict(result["summary"])
    return {key: float(value) for key, value in summary.items()}


def scaling_table(records: list[dict[str, float]]) -> str:
    headers = [
        "L",
        "dim",
        "t_fast",
        "t_ref",
        "t_mean",
        "t_q90",
        "t_worst",
        "logical_q90",
    ]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for record in records:
        lines.append(
            " | ".join(
                [
                    f"{int(record['n_syndrome'])}",
                    f"{int(record['dim'])}",
                    f"{record['t_fast']:.2f}",
                    f"{record['t_ref']:.2f}",
                    f"{record['t_mean']:.2f}",
                    f"{record['t_q90']:.2f}",
                    f"{record['t_worst']:.2f}",
                    f"{record['logical_q90']:.3f}",
                ]
            )
        )
    return "\n".join(lines)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    output_dir = ensure_output_dir(Path(__file__).resolve().parent / "outputs")
    epsilon = 0.35
    bundle_config = ModelConfig(
        n_syndrome=6,
        beta=1.8,
        leakage_scale=0.75,
        syndrome_rate=2.0,
    )
    bundle_result = run_bundle_experiment(
        config=bundle_config,
        epsilon=epsilon,
        num_samples=64,
        num_times=96,
        seed=2026,
    )
    records = run_scaling_study(
        sizes=[3, 4, 5, 6],
        epsilon=epsilon,
        num_samples=36,
        num_times=88,
        seed=2026,
    )

    save_schematic(
        bundle_config, output_dir / "appendixM_logical_sector_schematic.pdf"
    )
    save_bundle_figure(
        bundle_result, output_dir / "appendixM_logical_sector_bundle.pdf"
    )
    save_scaling_figure(
        records, output_dir / "appendixM_logical_sector_scaling.pdf"
    )
    save_logical_overlap_figure(
        records, output_dir / "appendixM_logical_sector_overlap.pdf", delta=0.1
    )


if __name__ == "__main__":
    main()
