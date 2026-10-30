"""Small meaningful controls for missingness, paired cohorts and support."""
import unittest
import numpy as np
from metrics import summary, fixed_cohort, geometry


def row(arm, parent, step, gap, masked=0):
    return dict(arm=arm, parent=parent, step=step, stage='predictor', gap=gap,
                target=2.5, masked=masked, num_atoms=4)


class MetricsTests(unittest.TestCase):
    def test_missing_is_not_zero_and_pairs_match(self):
        data = [row('cfg2', 0, 500, 2.5), row('cfg2', 1, 500, 4.5),
                row('cfg0', 0, 500, None, 1), row('cfg0', 1, 500, 3.5)]
        a, b = summary(data)
        self.assertEqual((a['n'], b['n'], a['paired_n']), (2, 1, 1))
        self.assertEqual((a['mae'], a['paired_mae'], b['paired_mae']), (1., 2., 1.))
        self.assertEqual(b['mask_fraction'], .125)

    def test_all_missing_remains_missing(self):
        results = summary([row(arm, 0, 0, None, 4) for arm in ('cfg2', 'cfg0')])
        self.assertTrue(all(r['mae'] is None and r['paired_mae'] is None and r['n'] == 0 for r in results))

    def test_fixed_cohort_intersects_all_times_and_arms(self):
        data = [row(arm, parent, step, 2.5) for arm in ('cfg2', 'cfg0') for parent in (0, 1) for step in (750, 900)]
        data[-1]['gap'] = None
        self.assertEqual(fixed_cohort(data, 750), [0])

    def test_eligibility_masks_collision_and_model_support(self):
        cell = np.eye(3)*5; frac = np.array([[0., 0., 0.], [.5, .5, .5]])
        self.assertIn('masked_types', geometry(np.array([1, 101]), cell, frac, 98)[0])
        self.assertIn('unsupported_types', geometry(np.array([1, 97]), cell, frac, 94)[0])
        self.assertEqual(geometry(np.array([1, 8]), cell, frac, 98)[0], [])
        self.assertIn('collision_guard', geometry(np.array([1, 8]), cell, frac*0, 98)[0])

    def test_bad_cells_and_nonfinite_positions(self):
        self.assertIn('invalid_cell', geometry(np.array([8]), np.eye(3)*.01, np.zeros((1, 3)), 98)[0])
        self.assertIn('nonfinite_geometry', geometry(np.array([8]), np.eye(3)*5, np.full((1, 3), np.nan), 98)[0])


if __name__ == '__main__': unittest.main()
