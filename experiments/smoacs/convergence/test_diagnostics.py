import unittest
import numpy as np
from diagnostics import objective,window_changes,normalized_direction


class DiagnosticsTests(unittest.TestCase):
    def test_original_objective_and_invalid_history(self):
        self.assertAlmostEqual(float(objective(2.5,-1.)),-1.)
        self.assertAlmostEqual(float(objective(2.8,-1.)),-.74)
        with self.assertRaises(ValueError):objective([2.5],[float('nan')])

    def test_plateau_and_decreasing_history(self):
        plateau=np.zeros((200,3))
        self.assertEqual(window_changes(plateau,179,199)['full_window_span_within_tolerance'],3)
        decreasing=np.repeat((-np.arange(200)*.001)[:,None],3,axis=1)
        report=window_changes(decreasing,179,199)
        self.assertEqual(report['updates_between'],20)
        self.assertEqual(report['decreased_gt_tolerance'],3)
        with self.assertRaises(ValueError):window_changes(decreasing,199,199)

    def test_endpoint_cancellation_is_not_a_stable_window(self):
        values=np.zeros((200,1));values[190]=1
        report=window_changes(values,179,199)
        self.assertEqual(report['endpoint_change_within_tolerance'],1)
        self.assertEqual(report['full_window_span_within_tolerance'],0)

    def test_normalized_gradient_direction_and_zero_gradient(self):
        x=np.array([1.,2.])
        direction=normalized_direction(2*x,.1)
        self.assertLess(float(np.dot(2*x,direction)),0)
        self.assertLess(float(np.sum((x+direction)**2)),float(np.sum(x**2)))
        np.testing.assert_array_equal(normalized_direction(np.zeros(2),.1),np.zeros(2))
        with self.assertRaises(ValueError):normalized_direction(np.array([np.inf]),.1)


if __name__=='__main__':unittest.main()
