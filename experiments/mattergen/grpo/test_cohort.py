import unittest
import numpy as np
from prepare import target, group, split, eligibility


class CohortContracts(unittest.TestCase):
    def test_closed_DFT_target_window(self):
        for gap in (2.4, 2.5, 2.6): self.assertTrue(target(gap))
        for gap in (0., 2.399, 2.601): self.assertFalse(target(gap))
        with self.assertRaises(ValueError): target(float('nan'))

    def test_group_split_survives_order_and_cell_replication(self):
        key = group(['Na', 'Cl'])
        self.assertEqual(key, group(['Cl', 'Na', 'Na', 'Cl']))
        self.assertEqual(split(key), split(group(['Na', 'Cl'])))
        self.assertIn(split(key), ('train', 'validation', 'test'))

    def test_support_and_geometric_rejections_are_explicit(self):
        c = np.eye(3)
        self.assertIsNone(eligibility([97], c, [[0, 0, 0]]))
        self.assertEqual(eligibility([99], c, [[0, 0, 0]]), 'ALIGNN_element_support')
        self.assertEqual(eligibility([1]*21, c, np.zeros((21, 3))), 'atom_count')
        self.assertEqual(eligibility([1], c*0, [[0, 0, 0]]), 'cell_orientation_or_volume')
        self.assertEqual(eligibility([1], c, [[float('inf'), 0, 0]]), 'geometry')


if __name__ == '__main__': unittest.main()
