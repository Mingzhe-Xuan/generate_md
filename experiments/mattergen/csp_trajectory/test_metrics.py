import unittest
from metrics import sequence,analyze,distribution


class MetricsTest(unittest.TestCase):
    def test_gap_does_not_bridge(self):
        r=sequence([0,1,2,3,4],[.8,.7,None,.1,.2])
        self.assertEqual(r['adjacent_pairs'],2)
        self.assertEqual(r['drops']['0.05'],1)
        self.assertAlmostEqual(r['maximum_contiguous_drawdown'],.1)

    def test_peak_not_just_single_drop(self):
        r=sequence(range(4),[.9,.85,.8,.95])
        self.assertAlmostEqual(r['maximum_contiguous_drawdown'],.1)
        self.assertEqual(r['drops']['0.01'],2)

    def test_stationary_unknown(self):
        self.assertEqual(sequence([0,1],[None,None])['adjacent_pairs'],0)
        self.assertIsNone(distribution([])['mean'])

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):sequence([0,0],[.1,.2])

    def test_cohort_and_smoothing(self):
        rows=[dict(trajectory=i,system=str(i),batch=0,stage='predictor',step=s,
            cos_score_force=(None if i==1 and s==900 else s/1000),cos_drift_force=s/1000)
            for i in range(2) for s in range(1000)]
        r=analyze(rows)
        self.assertEqual(r['fixed_cohorts'][0]['cos_score_force']['trajectories'],[0])
        self.assertEqual(r['fixed_cohorts'][2]['cos_score_force']['n'],2)
        self.assertEqual(r['paths'][1]['cos_score_force']['blocks25']['adjacent_pairs'],37)
        self.assertEqual(r['summary']['cos_drift_force']['drops']['0.0'],0)


if __name__=='__main__':unittest.main()
