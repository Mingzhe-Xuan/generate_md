"""Testable score/force/posterior identities; no assumption that a learned score is physical."""
import numpy as np


def centered_basis(n):
    """Cartesian orthonormal Helmert basis, removing only uniform translations."""
    if n < 2:
        raise ValueError('At least two atoms are needed.')
    h = np.zeros((n, n - 1))
    for k in range(n - 1):
        scale = np.sqrt((k + 1) * (k + 2))
        h[:k + 1, k] = 1 / scale
        h[k + 1, k] = -(k + 1) / scale
    return np.kron(h, np.eye(3))


def centered(v):
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    return (v - v.mean(0)).ravel()


def score_cartesian(score_fractional, cell):
    return centered(np.asarray(score_fractional) @ np.linalg.inv(cell).T)


def cartesian_noise_covariance(cell, fractional_std, n):
    cell = np.asarray(cell, dtype=float)
    q = centered_basis(n)
    covariance = np.kron(np.eye(n), fractional_std**2 * cell.T @ cell)
    return q.T @ covariance @ q


def fd_jacobian(plus, minus, step):
    if step <= 0:
        raise ValueError('Positive displacement required.')
    return (np.asarray(plus) - np.asarray(minus)).T / (2 * step)


def matrix_diagnostics(matrix):
    matrix = np.asarray(matrix, dtype=float)
    sym = .5 * (matrix + matrix.T)
    norm = float(np.linalg.norm(matrix))
    return dict(norm=norm, antisymmetric_fraction=float(np.linalg.norm(matrix - matrix.T) / (2 * max(norm, 1e-30))),
                symmetric_eigenvalues=np.linalg.eigvalsh(sym).tolist())


def harmonic_curvature(hessian, covariance, beta):
    """Exact Gaussian-smoothed Boltzmann curvature for an SPD harmonic well."""
    hessian, covariance = np.asarray(hessian), np.asarray(covariance)
    if beta <= 0 or np.linalg.eigvalsh(hessian).min() <= 0:
        raise ValueError('Positive beta and a strictly stable subspace are required.')
    return np.linalg.inv(np.linalg.inv(beta * hessian) + covariance)


def mixture_posterior(x, means, covariances, weights, noise_covariance, values=None):
    """Exact latent clean posterior for a finite Gaussian mixture plus Gaussian noise.

    A supplied property is constant within each component. The construction is a
    controlled model, not an estimate of MatterGen training-density weights.
    """
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    noise = np.asarray(noise_covariance, dtype=float)
    if np.any(weights <= 0) or np.linalg.eigvalsh(noise).min() <= 0:
        raise ValueError('Positive weights and positive-definite noise are required.')
    inverse = np.linalg.inv(covariances + noise)
    residual = x - means
    scores = -np.einsum('kij,kj->ki', inverse, residual)
    log_density = np.log(weights / weights.sum()) - .5 * (
        np.linalg.slogdet(covariances + noise)[1] + np.einsum('ki,kij,kj->k', residual, inverse, residual)
        + len(x) * np.log(2 * np.pi))
    maximum = float(log_density.max())
    responsibility = np.exp(log_density - maximum)
    normalizer = float(responsibility.sum())
    responsibility /= normalizer
    clean_means = means + np.einsum('kij,kjl,kl->ki', covariances, inverse, residual)
    clean_mean = responsibility @ clean_means
    score = responsibility @ scores
    result = dict(score=score, clean_mean=clean_mean, responsibility=responsibility,
                  log_density=maximum + np.log(normalizer))
    if values is not None:
        values = np.asarray(values, dtype=float)
        mean_value = float(responsibility @ values)
        cross_cov = np.sum(responsibility[:, None] * (clean_means - clean_mean) * (values - mean_value)[:, None], axis=0)
        result.update(property_mean=mean_value, property_gradient=np.linalg.solve(noise, cross_cov))
    return result


def remove_force_parallel(score, force):
    score, force = np.asarray(score), np.asarray(force)
    denominator = float(force @ force)
    return score - (score @ force) / denominator * force if denominator > 1e-30 else score.copy()


def cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-20 else None


def relative_error(first, second):
    return float(np.linalg.norm(np.asarray(first) - second) / max(np.linalg.norm(first), np.linalg.norm(second), 1e-30))
