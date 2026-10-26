"""Units, cell rotations and coordinate conventions independent of model weights."""
import numpy as np
import pytest
from mapping import symmetric_cell, symmetric_noise, clean_estimate, LatentGeometry, basis


def test_symmetrization_preserves_row_lattice_metric_and_periodic_distances():
    cell = np.array([[5., 0, 0], [1.1, 6, 0], [.3, 1.4, 7]])
    result = symmetric_cell(cell)
    assert np.allclose(result, result.T)
    assert np.allclose(result @ result.T, cell @ cell.T)
    frac = np.array([[.2, .5, .6], [-.3, .1, .4]])
    assert np.allclose(np.linalg.norm(frac@cell, axis=1), np.linalg.norm(frac@result, axis=1))


def test_affine_vp_inverse_recovers_ideal_clean_cell_with_nonzero_prior_mean():
    clean = np.arange(9).reshape(3, 3)/10
    coefficient = .7
    limit = np.eye(3)*4
    eps = symmetric_noise(np.random.default_rng(42).normal(size=(3, 3)))
    std = .8
    noisy = coefficient*clean + (1-coefficient)*limit + std*eps
    assert np.allclose(clean_estimate(noisy, -eps/std, std, coefficient, limit), clean)
    assert not np.allclose(noisy + std**2*(-eps/std), clean)


def test_wrapped_ve_local_lift_roundtrip_across_periodic_boundary():
    clean = np.array([[.98, .02, .4]])
    eps = np.array([[.5, -.5, .3]])
    std = .1
    noisy = (clean+std*eps) % 1
    recovered = clean_estimate(noisy, -eps/std, std) % 1
    assert np.allclose(recovered, clean)


def test_noise_preserves_symmetry_and_per_component_variance():
    rng = np.random.default_rng(31)
    samples = np.array([symmetric_noise(x) for x in rng.normal(size=(20000, 3, 3))])
    assert np.array_equal(samples, samples.transpose(0, 2, 1))
    assert np.allclose(samples.var(0), 1., atol=.035)
    assert np.allclose(np.einsum('aij,bij->ab', basis(), basis()), np.eye(6))


def test_latent_projection_removes_only_common_translation():
    geom = LatentGeometry([14, 8], np.eye(3)*5, [[.2]*3, [.6]*3])
    q = np.arange(geom.size, dtype=float)
    projected = geom.project(q)
    assert np.allclose(projected[:-6].reshape(2, 3).mean(0), 0)
    assert np.array_equal(projected[-6:], q[-6:])
    assert np.array_equal(geom.numbers, [14, 8])


def test_left_handed_cell_is_rejected():
    with pytest.raises(ValueError):
        symmetric_cell(np.diag([-5., 5., 5.]))


def test_latent_cell_and_cartesian_pullback_match_finite_differences():
    import torch
    geom = LatentGeometry([14, 8], [[5, 0, 0], [1, 6, 0], [.2, .3, 7]], [[.2]*3, [.6]*3])
    q = torch.linspace(-.01, .01, geom.size, dtype=torch.float64, requires_grad=True)
    def objective(v):
        cell, frac = geom.torch_pose(v)
        return (frac@cell).square().sum() + .3*cell.square().sum()
    gradient = torch.autograd.grad(objective(q), q)[0]
    direction = torch.linspace(-1., 1., geom.size, dtype=q.dtype)
    direction /= direction.norm()
    eps = 1e-6
    fd = (objective(q+eps*direction)-objective(q-eps*direction))/(2*eps)
    torch.testing.assert_close(fd, gradient@direction, rtol=1e-6, atol=1e-6)


def test_decode_keeps_score_jacobian_and_can_isolate_stop_gradient_control():
    import torch
    from types import SimpleNamespace
    from adapter import Denoiser
    class Graph(SimpleNamespace):
        def replace(self, **kw):
            return Graph(**(vars(self) | kw))
    class SDE:
        def marginal_prob(self, x, *unused):
            return x, torch.full_like(x, .2)
        def _marginal_mean_coeff(self, t):
            return torch.full_like(t, .8)
        def get_limit_mean(self, x, graph):
            return torch.ones_like(x)*3.
    fake = Denoiser.__new__(Denoiser)
    fake.module = SimpleNamespace(corruption=SimpleNamespace(sdes={'pos': SDE(), 'cell': SDE()}))
    fake.scores = lambda g, t: g.replace(pos=-2*g.pos, cell=-3*g.cell)
    pos = torch.full((2, 3), .2, dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=pos.dtype)[None].requires_grad_()
    graph = Graph(pos=pos, cell=cell, batch=torch.zeros(2, dtype=torch.long), atomic_numbers=torch.tensor([14, 8]))
    t = torch.tensor([.1], dtype=pos.dtype)
    result = fake.decode(graph, t)
    grad = torch.autograd.grad(result.pos.sum()+result.cell.sum(), (pos, cell))
    stopped = fake.decode(graph, t, stop_score_gradient=True)
    stop_grad = torch.autograd.grad(stopped.pos.sum()+stopped.cell.sum(), (pos, cell))
    torch.testing.assert_close(grad[0], torch.full_like(pos, .92))
    torch.testing.assert_close(grad[1], torch.full_like(cell, 1.1))
    torch.testing.assert_close(stop_grad[0], torch.ones_like(pos))
    torch.testing.assert_close(stop_grad[1], torch.full_like(cell, 1.25))
    assert torch.equal(result.atomic_numbers, graph.atomic_numbers)


def test_conversion_control_captures_actual_raw_forward_and_removes_hook():
    import torch
    from types import SimpleNamespace
    from adapter import score_with_raw
    class ChangingForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, graph, t):
            self.calls += 1
            return graph + self.calls
    model = ChangingForward()
    def score_fn(graph, t):
        return model(graph, t)/.02
    module = SimpleNamespace(model=model, score_fn=score_fn)
    score, raw = score_with_raw(module, torch.tensor([1.]), torch.tensor([.1]))
    assert model.calls == 1 and not model._forward_hooks
    torch.testing.assert_close(score, raw/.02)
    def fail(graph, t):
        model(graph, t)
        raise RuntimeError('test failure')
    with pytest.raises(RuntimeError):
        score_with_raw(SimpleNamespace(model=model, score_fn=fail), torch.tensor([1.]), None)
    assert not model._forward_hooks
