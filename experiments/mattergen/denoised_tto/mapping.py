"""Symmetric lattice coordinates and explicit affine score-to-clean conversion."""
import numpy as np


def symmetric_cell(cell):
    cell = np.asarray(cell, dtype=float)
    values, vectors = np.linalg.eigh(cell @ cell.T)
    if cell.shape != (3, 3) or np.linalg.det(cell) <= 0 or values.min() <= 0:
        raise ValueError('A right-handed nonsingular cell is required.')
    return (vectors * np.sqrt(values)) @ vectors.T


def basis():
    result = np.zeros((6, 3, 3))
    for i in range(3):
        result[i, i, i] = 1.
    for k, (i, j) in enumerate(((1, 2), (0, 2), (0, 1)), 3):
        result[k, i, j] = result[k, j, i] = 2**-.5
    return result


def symmetric_noise(noise):
    noise = np.asarray(noise)
    diagonal = np.diag(np.diag(noise))
    return (noise + noise.T - 2*diagonal) / np.sqrt(2) + diagonal


def clean_estimate(noisy, score, std, coefficient=1., limit_mean=0.):
    """Array/tensor-generic inversion; caller handles wrapped positions."""
    return (noisy + std**2 * score - (1-coefficient)*limit_mean) / coefficient


class LatentGeometry:
    def __init__(self, numbers, cell, frac):
        self.numbers = np.asarray(numbers, dtype=int)
        self.cell = symmetric_cell(cell)
        self.frac = np.asarray(frac, dtype=float)
        self.n = len(self.numbers)
        self.size = 3*self.n + 6
        if self.n < 2 or self.frac.shape != (self.n, 3):
            raise ValueError('Invalid fixed-species geometry.')

    def project(self, q):
        result = np.asarray(q, dtype=float).copy()
        atomic = result[:-6].reshape(self.n, 3)
        atomic -= atomic.mean(0)
        return result

    def torch_pose(self, q):
        import torch
        reference = torch.as_tensor(self.cell, dtype=q.dtype, device=q.device)
        directions = torch.as_tensor(basis(), dtype=q.dtype, device=q.device)
        cell = reference + torch.einsum('a,aij->ij', q[-6:], directions)
        frac = torch.as_tensor(self.frac, dtype=q.dtype, device=q.device)
        return cell, frac + np.sqrt(self.n)*q[:-6].reshape(self.n, 3) @ torch.linalg.inv(cell)
