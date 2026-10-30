import unittest
import numpy as np
from phase import events


class PhaseTests(unittest.TestCase):
    def test_unknown_types_and_final_composition_event(self):
        numbers = np.array([[101, 101], [8, 101], [8, 1], [8, 1], [8, 1]])
        steps = [0, 0, 5, 10, 1000]; stages = ['prior', 'predictor', 'predictor', 'predictor', 'terminal']
        rr = [dict(gap=gap, target=2.5) for gap in (None, None, None, 1., 2.)]
        result = events(numbers, steps, stages, rr)
        self.assertEqual((result['first_types_known'], result['first_evaluable'], result['final_composition_sustained_from']), (5, 10, 5))
        self.assertEqual(result['terminal_minus_first_error'], -1.)
        self.assertEqual((result['fixed_species_adjacent_pairs'], result['fixed_species_error_increase_pairs']), (1, 0))

    def test_species_changes_and_nonmonotonic_errors(self):
        numbers = np.array([[101], [8], [1], [1]])
        rr = [dict(gap=gap, target=2.5) for gap in (None, 2., 1., 2.5)]
        result = events(numbers, [0, 0, 5, 1000], ['prior', 'predictor', 'predictor', 'terminal'], rr)
        self.assertEqual(result['final_composition_sustained_from'], 5)
        self.assertEqual((result['target_error_increase_pairs'], result['fixed_species_error_increase_pairs']), (1, 0))
        self.assertFalse(result['first_vs_terminal_same_species'])

    def test_force_reduction_does_not_imply_property_improvement(self):
        rr = [dict(gap=None, target=2.5, force_rms=None), dict(gap=2., target=2.5, force_rms=3.),
              dict(gap=1., target=2.5, force_rms=.1)]
        result = events(np.array([[101], [8], [8]]), [0, 950, 1000], ['prior', 'predictor', 'terminal'], rr)
        self.assertEqual(result['terminal_minus_first_error'], 1.)
        self.assertAlmostEqual(result['terminal_minus_first_force_rms'], -2.9)


if __name__ == '__main__': unittest.main()
