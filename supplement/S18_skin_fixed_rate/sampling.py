from __future__ import annotations

import numpy as np


def haar_state(d: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=d) + 1j * rng.normal(size=d)
    z = z.astype(np.complex128)
    return z / np.linalg.norm(z)


def haar_population_batch(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.exponential(scale=1.0, size=(n, d))
    return weights / weights.sum(axis=1, keepdims=True)


def boundary_overlap_sample(psi: np.ndarray, A_B: np.ndarray) -> float:
    dB = int(A_B.shape[0])
    dR = int(psi.size // dB)
    X = psi.reshape(dB, dR)
    rhoB = X @ X.conj().T
    return float(np.real(np.trace(A_B @ rhoB)))


def sample_boundary_overlaps(
    dim: int,
    n_samples: int,
    rng: np.random.Generator,
    A_B: np.ndarray,
) -> np.ndarray:
    overlaps = np.empty(n_samples, dtype=float)
    for idx in range(n_samples):
        overlaps[idx] = boundary_overlap_sample(haar_state(dim, rng), A_B)
    return overlaps
