"""Independent audit rejects tampering and keeps unfavorable numerical cases."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analyze import audit_case, digest, fit_cohorts, physical_diagnostics
from relations import centered_basis


class AnalysisTests(unittest.TestCase):
    def test_unstable_modes_are_not_replaced(self):
        h = np.diag([-1., 0., 3.])
        result = physical_diagnostics(np.stack([h, h]))
        self.assertEqual(result['negative_modes'], 1)
        self.assertEqual(result['soft_modes'], 1)
        self.assertFalse(result['full_SPD'])
        self.assertTrue(result['numerical_QC'])

    def test_failed_qc_retained_in_all_case_cohort(self):
        records = [dict(id=s, system=s, t=.1, H=np.eye(2), C=np.eye(2),
                        observed=np.eye(2), native_QC=s != 'C', joint_QC=s != 'C', SPD=True)
                   for s in ('A', 'B', 'C')]
        cohorts = {r['cohort']: r for r in fit_cohorts(records)}
        self.assertEqual(cohorts['all_native']['cases'], 3)
        self.assertEqual(cohorts['all_native']['status'], 'computed')
        self.assertEqual(cohorts['native_QC']['cases'], 2)
        self.assertEqual(cohorts['native_QC']['status'], 'insufficient_chemical_systems')
        self.assertEqual(cohorts['full_SPD_joint_QC']['excluded'], ['C'])

    def test_raw_matrix_reconstruction_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            q = centered_basis(2)
            h = np.diag([2., 3., 4.])
            jac = np.array([[-1., 2., 0.], [0., -3., 1.], [0., 0., -4.]])
            raw = {}
            for label in ('anchor', 'probe'):
                for step in (.001, .002):
                    plus = -step * (q @ h).T
                    raw[label + '_' + str(step)] = np.stack([plus, -plus], axis=1)
            score_fields = []
            for step in (.001, .002):
                plus = step * (q @ jac).T
                score_fields.append(np.stack([plus, -plus], axis=1))
            np.savez(folder / 'vectors.npz', Q=q, cell=np.eye(3), numbers=[14, 14],
                     H_anchor=np.stack([h, h]), H_probe=np.stack([h, h]),
                     score_fields=np.array([score_fields]), fractional_std=[.1],
                     score_jacobian=np.array([[jac, jac]]), score_jacobian_ad=np.array([jac]), **raw)
            adjoints = (q @ jac.T).T
            np.savez(folder / 'time-0.05.npz', adjoints=adjoints, J_ad=jac)
            result = dict(status='completed', id='Si-00', system='Si', num_atoms=2, seconds=1., NFE=20,
                          vectors_sha256=digest(folder / 'vectors.npz'),
                          times=[dict(t=.05, native_derivative_QC_passed=False)])
            (folder / 'result.json').write_text(json.dumps(result))
            case, records = audit_case(folder, dict(policy=dict(FD_steps_A=[.001, .002])))
            self.assertLess(case['max_raw_reconstruction_error'], 1e-12)
            self.assertGreater(case['times'][0]['score_curvature']['antisymmetric_fraction'], .1)
            np.testing.assert_allclose(records[0]['observed'], -jac)
            self.assertFalse(records[0]['joint_QC'])
            with (folder / 'vectors.npz').open('ab') as f:f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):audit_case(folder, dict())


if __name__ == '__main__':unittest.main()
