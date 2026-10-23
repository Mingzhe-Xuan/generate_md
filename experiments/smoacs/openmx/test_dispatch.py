import unittest
from dispatch import select_rows


class DispatchTests(unittest.TestCase):
    def test_no_production_before_calibration(self):
        with self.assertRaises(ValueError):
            select_rows([], {'status':'numerical_convergence_failed'}, {})

    def test_unknown_sun_and_uncovered_candidates_stay_in_manifest(self):
        rows = [dict(candidate_id=f'mp-{i}.vasp', sun=False) for i in range(128)]
        rows[0]['sun'] = True
        rows[1]['sun'] = True
        rows[2]['sun'] = None
        calibration = dict(status='completed', numerical_convergence_passed=True, basis_level=1)
        ids, statuses = select_rows(rows, calibration, {'element_covered_ids':['mp-0.vasp']})
        self.assertEqual(ids, ['mp-0.vasp'])
        self.assertEqual(len(statuses), 128)
        self.assertEqual(statuses[1]['status'], 'blocked_missing_pbe19_resources')
        self.assertEqual(statuses[2]['status'], 'sun_unknown')


if __name__ == '__main__':
    unittest.main()
