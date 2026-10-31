import unittest
import numpy as np
from distribution import summarize, ranked_counts, binned


class GapDistribution(unittest.TestCase):
    def test_exact_modes_and_ties(self):
        self.assertEqual(ranked_counts([0., 0., 1., 2., 2.]),
                         [dict(value_eV=0., count=2), dict(value_eV=2., count=2), dict(value_eV=1., count=1)])

    def test_positive_subset_and_window_counts(self):
        s = summarize([0., 0., .1, .2, 2.3, 2.5, 2.7])
        self.assertEqual(s['zero_count'], 2)
        self.assertEqual(s['positive']['n'], 5)
        self.assertEqual(next(r['count'] for r in s['target_windows']
                              if r['target_eV'] == 2.5 and r['tolerance_eV'] == .2), 3)
        self.assertEqual(summarize([0., 0.])['positive']['exact_top'], [])

    def test_bin_edges_and_roundoff(self):
        r = binned([0., .099, .1, .3, .1+.2], .1)
        self.assertEqual(r, [dict(lower_eV=0., upper_eV=.1, count=2),
                             dict(lower_eV=.3, upper_eV=.4, count=2),
                             dict(lower_eV=.1, upper_eV=.2, count=1)])

    def test_invalid_labels_fail(self):
        for x in ([], [-.1], [np.nan], [np.inf], [[0.]]):
            with self.assertRaises(ValueError):
                summarize(x)


if __name__ == '__main__':
    unittest.main()
