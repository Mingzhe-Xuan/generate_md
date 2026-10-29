import unittest
import numpy as np
from precision_response import response_summary


class PrecisionResponseTests(unittest.TestCase):
    def test_nonsymmetric_linear_response_and_graph_flags(self):
        jac = np.array([[1., 2.], [-3., 4.]])
        v = np.array([.6, .8]);w = np.array([-.8, .6]);center = np.array([1., 2.])
        steps = [.001, .002]
        plus = np.array([center + h * jac @ v for h in steps])
        minus = np.array([center - h * jac @ v for h in steps])
        rows = response_summary(jac, v, w, plus, minus, steps, [True, False])
        self.assertLess(max(r['vector_AD_FD_relative_error'] for r in rows), 1e-12)
        self.assertTrue(rows[0]['same_graph'])
        self.assertFalse(rows[1]['same_graph'])
        self.assertAlmostEqual(rows[0]['scalar_AD'], rows[0]['scalar_FD'])

    def test_finite_step_nonlinearity_is_visible(self):
        steps = [.01, .1]
        v = w = np.ones(1)
        plus = np.array([[h + 10*h**3] for h in steps])
        minus = -plus
        rows = response_summary(np.ones((1, 1)), v, w, plus, minus, steps, [True, True])
        self.assertGreater(rows[1]['vector_AD_FD_relative_error'], rows[0]['vector_AD_FD_relative_error'])


if __name__ == '__main__':unittest.main()
