from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.linalg import eig, expm

from diagnostics import threshold_stats
from metrics import crossing_time, monotone_envelope, trace_distance
from sampling import haar_state


SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
SP = SM.conj().T
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def lindblad_dissipator(operator: np.ndarray, rho: np.ndarray) -> np.ndarray:
    opdag_op = operator.conj().T @ operator
    return operator @ rho @ operator.conj().T - 0.5 * (opdag_op @ rho + rho @ opdag_op)


def one_site_generator_matrix(
    lam: float,
    beta: float = 1.2,
    omega: float = 1.0,
    gphi: float = 0.5,
) -> np.ndarray:
    pup = np.exp(-beta * omega) / (1.0 + np.exp(-beta * omega))
    pdn = 1.0 / (1.0 + np.exp(-beta * omega))
    gup = lam * pup
    gdn = lam * pdn

    jumps = [
        np.sqrt(gdn) * SM,
        np.sqrt(gup) * SP,
        np.sqrt(gphi * lam) * SZ,
    ]
    basis = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    ]

    generator = np.zeros((4, 4), dtype=np.complex128)
    for col, basis_state in enumerate(basis):
        image = np.zeros((2, 2), dtype=np.complex128)
        for jump in jumps:
            image += lindblad_dissipator(jump, basis_state)
        generator[:, col] = image.reshape(-1)
    return generator


@lru_cache(maxsize=None)
def interleave_perm(L: int) -> tuple[int, ...]:
    perm = []
    for idx in range(L):
        perm.extend([idx, L + idx])
    return tuple(perm)


def rho_to_pair_tensor(rho: np.ndarray, L: int) -> np.ndarray:
    tensor_std = rho.reshape((2,) * (2 * L))
    tensor_pair = np.transpose(tensor_std, interleave_perm(L))
    return tensor_pair.reshape((4,) * L)


def pair_tensor_to_rho(tensor_pair: np.ndarray, L: int) -> np.ndarray:
    perm = np.asarray(interleave_perm(L))
    inv_perm = np.argsort(perm)
    tensor_pair = tensor_pair.reshape((2, 2) * L)
    tensor_std = np.transpose(tensor_pair, axes=inv_perm)
    dim = 2 ** L
    return tensor_std.reshape(dim, dim)


def apply_local_superoperator(tensor_pair: np.ndarray, matrix: np.ndarray, site: int) -> np.ndarray:
    out = np.tensordot(matrix, tensor_pair, axes=(1, site))
    return np.moveaxis(out, 0, site)


def kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for operator in operators:
        out = np.kron(out, operator)
    return out


def thermal_qubit(beta: float = 1.2, omega: float = 1.0) -> np.ndarray:
    ground = 1.0 / (1.0 + np.exp(-beta * omega))
    excited = 1.0 - ground
    return np.diag([ground, excited]).astype(np.complex128)


def stationary_state(L: int, beta: float = 1.2, omega: float = 1.0) -> np.ndarray:
    return kron_all([thermal_qubit(beta=beta, omega=omega) for _ in range(L)])


def reference_state(L: int) -> np.ndarray:
    dim = 2 ** L
    return np.eye(dim, dtype=np.complex128) / dim


def site_rates(L: int, lambda_s: float) -> list[float]:
    return [float(lambda_s)] + [1.0] * (L - 1)


def step_superoperators(
    times: np.ndarray,
    lambdas: Sequence[float],
    beta: float = 1.2,
    omega: float = 1.0,
    gphi: float = 0.5,
) -> list[list[np.ndarray]]:
    deltas = np.diff(np.concatenate(([0.0], np.asarray(times, dtype=float))))
    local_generators = [
        one_site_generator_matrix(lam=float(lam), beta=beta, omega=omega, gphi=gphi)
        for lam in lambdas
    ]
    return [
        [expm(float(delta_t) * local_generator) for local_generator in local_generators]
        for delta_t in deltas
    ]


def relaxation_curve(
    rho0: np.ndarray,
    *,
    L: int,
    sigma: np.ndarray,
    steps: list[list[np.ndarray]],
) -> np.ndarray:
    tensor = rho_to_pair_tensor(rho0, L)
    curve = np.empty(len(steps), dtype=float)
    for idx, local_steps in enumerate(steps):
        for site, local_matrix in enumerate(local_steps):
            tensor = apply_local_superoperator(tensor, local_matrix, site)
        rho_t = pair_tensor_to_rho(tensor, L)
        curve[idx] = trace_distance(rho_t, sigma)
    return monotone_envelope(curve)


def time_grid(Delta: float, eps_floor: float, num_times: int = 72) -> np.ndarray:
    t_max = max(8.0, 4.5 * np.log(2.0 / eps_floor) / max(Delta, 1.0e-9))
    return np.concatenate(([0.0], np.geomspace(1.0e-3, t_max, num_times - 1)))


def local_gap_and_left_mode(
    lam: float,
    beta: float = 1.2,
    omega: float = 1.0,
    gphi: float = 0.5,
) -> tuple[float, np.ndarray]:
    local_generator = one_site_generator_matrix(lam=lam, beta=beta, omega=omega, gphi=gphi)
    evals, left_vecs = eig(local_generator.T)
    nonzero = np.where(np.real(evals) < -1.0e-10)[0]
    slow_idx = int(nonzero[np.argmax(np.real(evals[nonzero]))])
    slow_eval = evals[slow_idx]
    left_mode = left_vecs[:, slow_idx].reshape(2, 2)
    left_mode = 0.5 * (left_mode + left_mode.conj().T)
    left_mode = left_mode - np.trace(left_mode) * I2 / 2.0
    op_norm = np.linalg.norm(left_mode, ord=2)
    if op_norm > 0.0:
        left_mode = left_mode / op_norm
    if np.real(left_mode[0, 0]) < 0.0:
        left_mode = -left_mode
    return float(-np.real(slow_eval)), left_mode


def slow_boundary_state(left_mode: np.ndarray, beta: float, omega: float, L: int) -> np.ndarray:
    evals, evecs = np.linalg.eigh(left_mode)
    state = np.outer(evecs[:, np.argmax(evals)], evecs[:, np.argmax(evals)].conj())
    factors = [state] + [thermal_qubit(beta=beta, omega=omega) for _ in range(L - 1)]
    return kron_all(factors)


def run_boundary_case(
    *,
    L: int,
    lambda_s: float,
    eps_values: Sequence[float],
    delta: float,
    rng: np.random.Generator,
    beta: float = 1.2,
    omega: float = 1.0,
    gphi: float = 0.5,
    num_samples: int = 40,
    num_times: int = 64,
) -> dict[str, object]:
    eps_values = [float(eps) for eps in eps_values]
    eps_floor = float(min(eps_values))
    lambdas = site_rates(L, lambda_s)

    boundary_gap, A_B = local_gap_and_left_mode(lambda_s, beta=beta, omega=omega, gphi=gphi)
    bulk_gap, _ = local_gap_and_left_mode(1.0, beta=beta, omega=omega, gphi=gphi)
    Delta = float(min(boundary_gap, bulk_gap))

    times = time_grid(Delta=Delta, eps_floor=eps_floor, num_times=num_times)
    steps = step_superoperators(times, lambdas, beta=beta, omega=omega, gphi=gphi)

    dim = 2 ** L
    sigma = stationary_state(L, beta=beta, omega=omega)
    rho_ref = reference_state(L)
    rho_slow = slow_boundary_state(A_B, beta=beta, omega=omega, L=L)

    curves = np.empty((num_samples, len(times)), dtype=float)
    overlaps = np.empty(num_samples, dtype=float)
    for idx in range(num_samples):
        psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi = psi / np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())
        curves[idx] = relaxation_curve(rho0, L=L, sigma=sigma, steps=steps)
        X = psi.reshape(2, dim // 2)
        rhoB = X @ X.conj().T
        overlaps[idx] = float(np.real(np.trace(A_B @ rhoB)))

    h = relaxation_curve(rho_ref, L=L, sigma=sigma, steps=steps)
    slow_curve = relaxation_curve(rho_slow, L=L, sigma=sigma, steps=steps)

    basis_tmix: dict[float, list[float]] = {eps: [] for eps in eps_values}
    basis_worst_curve_by_eps: dict[float, np.ndarray | None] = {eps: None for eps in eps_values}
    basis_worst_index_by_eps: dict[float, int] = {eps: -1 for eps in eps_values}
    basis_worst_time_by_eps: dict[float, float] = {eps: -np.inf for eps in eps_values}
    for basis_index in range(dim):
        rho_basis = np.zeros((dim, dim), dtype=np.complex128)
        rho_basis[basis_index, basis_index] = 1.0
        curve_basis = relaxation_curve(rho_basis, L=L, sigma=sigma, steps=steps)
        for eps in eps_values:
            t_basis = float(crossing_time(curve_basis, times, eps))
            basis_tmix[eps].append(t_basis)
            if t_basis > basis_worst_time_by_eps[eps]:
                basis_worst_time_by_eps[eps] = t_basis
                basis_worst_index_by_eps[eps] = basis_index
                basis_worst_curve_by_eps[eps] = curve_basis.copy()

    eps_summary: dict[float, dict[str, object]] = {}
    abs_overlap = np.abs(overlaps)
    A_max = float(np.linalg.norm(A_B, ord=2))
    alpha_typ = float(np.quantile(abs_overlap, 1.0 - delta))
    for eps in eps_values:
        stats = threshold_stats(curves, times, eps)
        t_ref = float(crossing_time(h, times, eps))
        t_slow = float(crossing_time(slow_curve, times, eps))
        t_worst_basis = float(np.nanmax(np.asarray(basis_tmix[eps], dtype=float)))
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
            "t_slow": t_slow,
            "t_worst_basis": t_worst_basis,
            "worst_basis_index": int(basis_worst_index_by_eps[eps]),
            "worst_basis_curve": basis_worst_curve_by_eps[eps],
            "tmix_std": float(np.nanstd(stats["tmix"])),
            "V_over_m": float(stats["V"] / max(stats["m_cross"], 1.0e-12)),
            "gap_ratio": float(stats["s_cross"] / max(Delta, 1.0e-12)),
            "delta_t_pred": float(
                np.log(max(A_max, 1.0e-14) / max(alpha_typ, 1.0e-14)) / max(Delta, 1.0e-14)
            ),
            "delta_t_act": float(t_slow - np.nanquantile(stats["tmix"], 1.0 - delta)),
            "delta_t_act_basis": float(t_worst_basis - np.nanquantile(stats["tmix"], 1.0 - delta)),
        }

    return {
        "L": L,
        "lambda_s": float(lambda_s),
        "times": times,
        "curves": curves,
        "h": h,
        "slow_curve": slow_curve,
        "Delta": Delta,
        "A_B": A_B,
        "overlaps": overlaps,
        "eps_summary": eps_summary,
        "sigma": sigma,
    }


def run_boundary_bundle_only(
    *,
    L: int,
    lambda_s: float,
    eps: float,
    rng: np.random.Generator,
    beta: float = 1.2,
    omega: float = 1.0,
    gphi: float = 0.5,
    num_samples: int = 12,
    num_times: int = 40,
) -> dict[str, object]:
    lambdas = site_rates(L, lambda_s)
    Delta, _ = local_gap_and_left_mode(lambda_s, beta=beta, omega=omega, gphi=gphi)
    times = time_grid(Delta=Delta, eps_floor=eps, num_times=num_times)
    steps = step_superoperators(times, lambdas, beta=beta, omega=omega, gphi=gphi)

    dim = 2 ** L
    sigma = stationary_state(L, beta=beta, omega=omega)
    rho_ref = reference_state(L)

    curves = np.empty((num_samples, len(times)), dtype=float)
    for idx in range(num_samples):
        psi = haar_state(dim, rng)
        rho0 = np.outer(psi, psi.conj())
        curves[idx] = relaxation_curve(rho0, L=L, sigma=sigma, steps=steps)

    h = relaxation_curve(rho_ref, L=L, sigma=sigma, steps=steps)
    stats = threshold_stats(curves, times, eps)
    return {
        "L": L,
        "lambda_s": float(lambda_s),
        "Delta": Delta,
        "times": times,
        "curves": curves,
        "h": h,
        "mu": stats["mu"],
        "tstar": stats["tstar"],
        "eps": float(eps),
    }
