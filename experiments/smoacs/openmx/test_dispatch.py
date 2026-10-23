import unittest
from dispatch import select_rows, calibration_profile


class DispatchTests(unittest.TestCase):
    def test_production_preserves_certified_larger_basis(self):
        profile = calibration_profile(dict(status='completed', numerical_convergence_passed=True,
                                            basis_level=2, mesh_increment=0))
        self.assertEqual(profile, dict(basis_level=2, mesh_increment=0))

    def test_no_production_before_calibration(self):
        rows = [dict(candidate_id=f'mp-{i}.vasp', sun=False) for i in range(128)]
        good = dict(status='completed', numerical_convergence_passed=True, basis_level=1, mesh_increment=0)
        for change in (dict(status='numerical_convergence_failed'),
                       dict(numerical_convergence_passed=False), dict(basis_level=0), dict(mesh_increment=2)):
            with self.assertRaisesRegex(ValueError, 'Production blocked'):
                select_rows(rows, dict(good, **change), {'element_covered_ids':[]})

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
