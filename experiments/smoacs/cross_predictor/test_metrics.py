import unittest
from metrics import classify,numerical,agreement,index_rows


class AgreementTests(unittest.TestCase):
    def test_thresholds_and_missing_are_not_failure(self):
        self.assertTrue(classify(2.3,-.50001,True)['joint'])
        self.assertTrue(classify(2.7,-1.,True)['gap'])
        self.assertFalse(classify(2.5,-.5,True)['eform'])
        self.assertFalse(classify(2.5,-1.,False)['joint'])
        self.assertIsNone(classify(None,None,True)['joint'])
        with self.assertRaises(ValueError):classify(float('nan'),-1.,True)

    def test_numeric_identity_bias_and_constant_correlation(self):
        result=numerical([1,2,3],[2,3,4])
        self.assertEqual(result['mae_between_predictors'],1.)
        self.assertEqual(result['rmse_between_predictors'],1.)
        self.assertEqual(result['bias_alignn_minus_crystalformer'],1.)
        self.assertAlmostEqual(result['pearson'],1.)
        self.assertIsNone(numerical([1,1],[2,2])['pearson'])
        self.assertEqual(numerical([],[])['n'],0)

    def test_confusion_retention_and_unknown_denominator(self):
        rows=[]
        for gap_cf,gap_al in ((2.5,2.5),(2.5,1.),(1.,2.5),(1.,1.),(2.5,None)):
            rows.append(dict(status='completed' if gap_al is not None else 'prediction_failed',
                crystalformer_gap=gap_cf,alignn_gap=gap_al,crystalformer_eform=-1.,alignn_eform=-1. if gap_al is not None else None,
                crystalformer_flags=classify(gap_cf,-1.,True),alignn_flags=classify(gap_al,-1. if gap_al is not None else None,True)))
        result=agreement(rows)
        c=result['classification']['joint']
        self.assertEqual(result['denominator'],5)
        self.assertEqual([c[k] for k in ('both_pass','crystalformer_only','alignn_only','neither_pass','unknown')],[1]*5)
        self.assertAlmostEqual(c['confirmed_retention_of_crystalformer_pass'],1/3)
        self.assertEqual(result['numerical']['gap']['n'],4)
        self.assertEqual(c['classification_agreement'],.5)

    def test_id_mismatch_rejected(self):
        with self.assertRaises(ValueError):index_rows([{'candidate_id':'a'},{'candidate_id':'a'}],['a','b'])
        with self.assertRaises(ValueError):index_rows([{'candidate_id':'other'}],['a'])


if __name__=='__main__':unittest.main()
