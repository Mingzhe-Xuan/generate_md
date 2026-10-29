"""Synthetic laws and explicit cross-validation leakage regression."""
import unittest
import numpy as np
from curvature_fit import fit,heldout_systems
from relations import harmonic_curvature


def examples(kind='thermal'):
    records=[]
    for group in range(4):
        for replica in range(2):
            h=np.array([[2.+group,.2],[.2, .5+.3*replica]])
            c=np.array([[.1+.01*group,.02],[.02,.2+.03*replica]])
            observed=harmonic_curvature(h,c,7.) if kind=='thermal' else 3.*np.linalg.inv(c)
            records.append(dict(id=f'{group}-{replica}',system=str(group),H=h,C=c,observed=observed))
    return records


class CurvatureCrossValidation(unittest.TestCase):
    def test_exact_noise_law_on_heldout_systems(self):
        result=heldout_systems(examples('noise'),'scaled_noise')
        self.assertAlmostEqual(result['R2_zero_baseline'],1.)
        for fold in result['folds']:
            self.assertAlmostEqual(fold['parameter'],3.)
            self.assertNotIn(fold['system'],fold['training_systems'])

    def test_noncommuting_harmonic_law_predicts_heldout_cases(self):
        result=heldout_systems(examples(),'blurred_H')
        self.assertGreater(result['R2_zero_baseline'],.999)
        for fold in result['folds']:
            self.assertLess(abs(fold['parameter']/7.-1),.02)

    def test_heldout_targets_cannot_change_their_fitted_parameter(self):
        records=examples()
        before=heldout_systems(records,'scaled_H')['folds'][0]
        for row in records:
            if row['system']=='0':row['observed']=-100*row['observed']
        after=heldout_systems(records,'scaled_H')['folds'][0]
        self.assertEqual(before['parameter'],after['parameter'])
        self.assertNotEqual(before['errors'],after['errors'])

    def test_negative_r2_is_not_clipped(self):
        records=examples('noise')
        for row in records:
            row['observed']=row['observed']*.001
        result=heldout_systems(records,'noise')
        self.assertLess(result['R2_zero_baseline'],0.)


if __name__=='__main__':unittest.main()
