"""Posterior uncertainty and forward-property geometry from the inverse score."""
import unittest
import numpy as np
from relations import (mixture_posterior, fd_jacobian,
    posterior_covariance_from_score_jacobian, linear_property_posterior_gradient)


class PosteriorGeometryTests(unittest.TestCase):
    def test_second_order_tweedie_matches_total_covariance(self):
        means = np.array([[-1., .2], [1., -.3]])
        cov = np.array([[[.1, .02], [.02, .2]], [[.3, -.04], [-.04, .1]]])
        noise = np.array([[.7, .1], [.1, .4]])
        x = np.array([.2, -.1]);step = 1e-5
        evaluate = lambda z: mixture_posterior(z, means, cov, [.3, .7], noise)
        plus = [evaluate(x + step * v)['score'] for v in np.eye(2)]
        minus = [evaluate(x - step * v)['score'] for v in np.eye(2)]
        jac = fd_jacobian(plus, minus, step)
        actual = evaluate(x)['clean_covariance']
        np.testing.assert_allclose(posterior_covariance_from_score_jacobian(jac, noise), actual, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(actual).min(), 0.)

    def test_linear_readout_pullback_and_covariance_gradient(self):
        means = np.array([[-1., .2], [1., -.3]])
        cov = np.array([[[.1, .02], [.02, .2]], [[.3, -.04], [-.04, .1]]])
        noise = np.array([[.7, .1], [.1, .4]])
        x = np.array([.2, -.1]);g = np.array([2., -1.]);step = 1e-5
        evaluate = lambda z: mixture_posterior(z, means, cov, [.3, .7], noise)
        plus = [evaluate(x + step * v)['score'] for v in np.eye(2)]
        minus = [evaluate(x - step * v)['score'] for v in np.eye(2)]
        jac = fd_jacobian(plus, minus, step)
        finite = np.array([(g @ evaluate(x + step*v)['clean_mean'] - g @ evaluate(x - step*v)['clean_mean'])/(2*step) for v in np.eye(2)])
        pullback = linear_property_posterior_gradient(jac, noise, g)
        covariance_gradient = np.linalg.solve(noise, evaluate(x)['clean_covariance'] @ g)
        np.testing.assert_allclose(pullback, finite, atol=1e-10)
        np.testing.assert_allclose(pullback, covariance_gradient, atol=1e-10)

    def test_invalid_proxy_is_not_repaired_into_covariance(self):
        invalid = np.array([[-2., 1.], [0., 0.]])
        proxy = posterior_covariance_from_score_jacobian(invalid, np.eye(2))
        self.assertGreater(np.linalg.norm(proxy-proxy.T), .1)
        self.assertLess(np.linalg.eigvalsh(.5*(proxy+proxy.T)).min(), 0.)


if __name__ == '__main__':unittest.main()
