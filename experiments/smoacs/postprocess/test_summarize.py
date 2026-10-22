import unittest
from summarize import count_statuses


class SummaryTests(unittest.TestCase):
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
