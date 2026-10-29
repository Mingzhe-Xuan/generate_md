"""Analytic controls independent of pretrained-model behavior."""
import unittest
from pathlib import Path
import sys
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from relations import (centered_basis, cartesian_noise_covariance, fd_jacobian,
                       harmonic_curvature, mixture_posterior, remove_force_parallel, projected_adjoint_jacobian)
from prepare_anchors import symmetric_cell
from probe import save
from physics_client import Physics


class ForwardInverseIdentities(unittest.TestCase):
    def test_adjoint_jacobian_orientation_for_nonconservative_field(self):
        rng = np.random.default_rng(52)
        q = centered_basis(3)
        matrix = rng.normal(size=(9,9))
        adjoints = [matrix.T @ v for v in q.T]
        actual = projected_adjoint_jacobian(q, adjoints)
        np.testing.assert_allclose(actual, q.T @ matrix @ q, atol=1e-14)
        self.assertGreater(np.linalg.norm(actual-actual.T), 1.)

    def test_worker_keeps_virtualenv_executable_symlink(self):
        original = Path.resolve
        def emulate_symlink(path, *a, **kw):
            return Path('/system/python') if path.name == 'python' else original(path, *a, **kw)
        fake = SimpleNamespace(stdout=object(), poll=lambda: 0)
        rpc = SimpleNamespace(read_response=lambda *a: {'ready': True})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'rpc': rpc}), \
             patch('physics_client.subprocess.Popen', return_value=fake) as launch, \
             patch.object(Path, 'resolve', emulate_symlink):
            client = Physics(Path(tmp)/'rpc')
            self.assertEqual(launch.call_args.args[0][0], str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'))
            client.close()

    def test_actual_probe_record_handles_numpy_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result.json'
            save(path, dict(geometry=np.eye(3), energy=np.float64(-3.), NFE=np.int64(7)))
            value = json.loads(path.read_text())
            self.assertEqual(value['geometry'], np.eye(3).tolist())
            self.assertEqual(value['NFE'], 7)
            self.assertEqual(value['energy'], -3.)

    def test_module_names_do_not_shadow_standard_library(self):
        for path in Path(__file__).parent.glob('*.py'):
            self.assertNotIn(path.stem, sys.stdlib_module_names)

    def test_symmetric_cell_preserves_geometry_metric(self):
        cell = np.array([[3., .2, .1], [0., 4., .5], [0., 0., 6.]])
        result = symmetric_cell(cell)
        np.testing.assert_allclose(result @ result.T, cell @ cell.T, atol=1e-12)
        np.testing.assert_allclose(result, result.T, atol=1e-12)
        self.assertGreater(np.linalg.det(result), 0.)

    def test_translation_projection_and_orthonormality(self):
        for n in (2, 5, 8):
            q = centered_basis(n)
            np.testing.assert_allclose(q.T @ q, np.eye(3 * n - 3), atol=1e-14)
            np.testing.assert_allclose(q @ q.T, np.kron(np.eye(n) - np.ones((n, n))/n, np.eye(3)), atol=1e-14)

    def test_anisotropic_cartesian_noise_metric(self):
        cell = np.array([[3., .2, .1], [0., 4., .5], [0., 0., 6.]])
        covariance = cartesian_noise_covariance(cell, .1, 4)
        np.testing.assert_allclose(covariance, np.kron(np.eye(3), .01 * cell.T @ cell), atol=1e-14)

    def test_smoothed_harmonic_force_requires_covariance(self):
        h = np.array([[4., .7], [.7, 1.]])
        noise = np.array([[.4, .05], [.05, .1]])
        beta = 3.
        x = np.array([.3, -.2])
        covariance = np.linalg.inv(beta * h)
        actual = mixture_posterior(x, np.zeros((1, 2)), covariance[None], [1.], noise)
        predicted = harmonic_curvature(h, noise, beta)
        np.testing.assert_allclose(actual['score'], -predicted @ x, atol=1e-14)
        np.testing.assert_allclose(actual['score'], beta * (-h @ actual['clean_mean']), atol=1e-14)
        self.assertGreater(np.linalg.norm(actual['score'] - beta * (-h @ x)), .1)

    def test_tweedie_identity_for_mixture(self):
        means = np.array([[-1., .2], [1., -.3]])
        cov = np.array([np.eye(2) * .02, np.eye(2) * .1])
        noise = np.array([[.7, .1], [.1, .4]])
        x = np.array([.2, -.1])
        p = mixture_posterior(x, means, cov, [.3, .7], noise)
        np.testing.assert_allclose(p['clean_mean'], x + noise @ p['score'], atol=1e-14)

    def test_property_posterior_covariance_identity(self):
        means = np.array([[-1., .2], [1., -.3]])
        cov = np.array([np.eye(2) * .02, np.eye(2) * .1])
        noise = np.array([[.7, .1], [.1, .4]])
        x = np.array([.2, -.1])
        evaluate = lambda z: mixture_posterior(z, means, cov, [.3, .7], noise, [1., 3.])
        h = 1e-5
        numerical = np.array([(evaluate(x + h * v)['property_mean'] - evaluate(x - h * v)['property_mean']) / (2*h) for v in np.eye(2)])
        np.testing.assert_allclose(evaluate(x)['property_gradient'], numerical, rtol=1e-8, atol=1e-9)

    def test_conditional_score_is_weighted_clean_mean_shift(self):
        means = np.array([[-1., .2], [1., -.3]])
        cov = np.array([np.eye(2) * .02, np.eye(2) * .1])
        noise = np.array([[.7, .1], [.1, .4]])
        x = np.array([.2, -.1])
        weights = np.array([.3, .7])
        a = mixture_posterior(x, means, cov, weights, noise)
        b = mixture_posterior(x, means, cov, weights * [.1, .9], noise)
        np.testing.assert_allclose(b['score'] - a['score'], np.linalg.solve(noise, b['clean_mean'] - a['clean_mean']), atol=1e-14)

    def test_projection_can_destroy_integrability(self):
        k, h = np.diag([1., 4.]), np.diag([3., 1.])
        x, step = np.array([1., .7]), 1e-5
        field = lambda z: remove_force_parallel(-k @ z, -h @ z)
        plus = [field(x + step * v) for v in np.eye(2)]
        minus = [field(x - step * v) for v in np.eye(2)]
        jac = fd_jacobian(plus, minus, step)
        self.assertGreater(np.linalg.norm(jac - jac.T), .1)

    def test_energy_orthogonality_is_only_first_order(self):
        h = np.diag([3., 1.])
        x = np.array([1., .7])
        d = remove_force_parallel(-np.diag([1., 4.]) @ x, -h @ x)
        self.assertAlmostEqual(float((-h @ x) @ d), 0.)
        eta = .1
        actual = .5 * (x + eta*d) @ h @ (x + eta*d) - .5 * x @ h @ x
        self.assertGreater(actual, 0.)
        self.assertAlmostEqual(actual, .5 * eta**2 * d @ h @ d)

    def test_relaxation_pullback_erases_local_property_gradient(self):
        h = np.diag([1., 4.])
        gradient = np.array([2., 1.])
        norms = [np.linalg.norm(np.diag(np.exp(-np.diag(h) * t)) @ gradient) for t in (0., 1., 5., 20.)]
        self.assertTrue(all(a > b for a, b in zip(norms, norms[1:])))
        self.assertLess(norms[-1], 1e-8)


if __name__ == '__main__':
    unittest.main()
