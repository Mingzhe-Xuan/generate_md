import unittest
from summarize import count_statuses, dft_readiness


class SummaryTests(unittest.TestCase):
    def test_potential_variants_and_unconverged_inputs_block_dft(self):
        self.assertEqual(dft_readiness(['Li_sv'], ['Li'], 'converged'),
                         ('blocked_missing_potentials',['Li_sv']))
        self.assertEqual(dft_readiness(['Li_sv'], ['Li_sv'], 'unconverged'),
                         ('blocked_relaxation',[]))
        self.assertEqual(dft_readiness(['Li_sv'], ['Li_sv'], 'converged'), ('not_run',[]))

    def test_failures_and_missing_stay_in_denominator(self):
        rows = [dict(candidate_id='a',status='converged'),dict(candidate_id='b',status='failed'),
                dict(candidate_id='c',status='missing'),dict(candidate_id='d',status='unconverged')]
        result = count_statuses(rows, 4)
        self.assertEqual(result['relaxation_converged_fraction'], 0.25)
        self.assertEqual(sum(result['counts'].values()), 4)
        with self.assertRaises(ValueError):
            count_statuses(rows[:3], 4)
        with self.assertRaises(ValueError):
            count_statuses([rows[0]]*4, 4)


if __name__ == '__main__':
    unittest.main()
