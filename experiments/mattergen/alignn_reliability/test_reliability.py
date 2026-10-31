import tempfile
from pathlib import Path
import unittest
import numpy as np
from core import GAP_SHA, labels, composition, role, standardize_fit, metrics, calibrate
from head import Head


class Contracts(unittest.TestCase):
    def test_error_thresholds_and_boundaries(self):
        self.assertEqual(labels([1., 1., 1.], [1.1, 1.2, 1.5], .2).tolist(), [1, 1, 0])
        self.assertEqual(labels([0.], [max(0., -.1)], .1).tolist(), [1])
        with self.assertRaises(ValueError): labels([1.], [np.nan], .2)

    def test_group_invariant_to_atom_order_and_formula_units(self):
        a = composition(['Na','Cl']); b = composition(['Cl','Na','Na','Cl'])
        self.assertEqual(a, b); self.assertEqual(role(a,'val'), role(b,'val'))
        self.assertEqual(role(a,'test'), 'test')
        with self.assertRaises(ValueError): role(a,'train')

    def test_scaler_does_not_fit_evaluation_outlier(self):
        x=np.array([[1., 5.], [3., 5.]])
        mean, scale=standardize_fit(x)
        self.assertTrue(np.array_equal(mean, [2., 5.]))
        self.assertTrue(np.array_equal(scale, [1., 1.]))
        self.assertEqual(((np.array([[102., 5.]])-mean)/scale)[0,0],100.)

    def test_selective_metrics_and_bad_class(self):
        result=metrics([1,0,1],[.95,.05,.8],[.05,1.,.1])
        self.assertEqual(result['auroc'],1.); self.assertEqual(result['ap_bad'],1.)
        self.assertEqual(result['accept_0.9']['n'],1)
        self.assertEqual(result['accept_0.9']['correct_fraction'],1.)
        self.assertLess(result['coverage_curve'][0]['mae'],result['mae'])

    def test_empty_and_single_class(self):
        self.assertEqual(metrics([],[],[]),dict(n=0))
        r=metrics([1,1],[.1,.2],[.1,.1]);self.assertIsNone(r['auroc'])
        self.assertEqual(r['accept_0.9']['n'],0);self.assertIsNone(r['accept_0.9']['mae'])

    def test_calibration_preserves_ranking(self):
        fit=calibrate(np.array([-3.,-2.,-1.,1.,2.,3.]),np.array([0,1,0,1,0,1]))
        self.assertGreater(fit['a'],0)
        self.assertEqual(calibrate([0,1],[1,1])['status'],'single_class_identity')

    def test_head_serialization_and_identity(self):
        arrays=dict(kind=np.array('linear'),input=np.array('h'),mean=np.zeros(2),scale=np.ones(2),
            weight=np.array([1.,-1.]),bias=np.array(0.),predictor_sha256=np.array(GAP_SHA),cal_a=np.array(1.),cal_b=np.array(0.))
        h=Head(arrays);x=np.array([[2.,0.],[0.,2.]])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'head.npz';h.save(path)
            np.testing.assert_allclose(Head.load(path).predict_proba(x),h.predict_proba(x))
            h.arrays['predictor_sha256']=np.array('wrong');h.save(path)
            with self.assertRaises(ValueError):Head.load(path)

    def test_mlp_and_prediction_only_inference(self):
        h=Head(dict(kind=np.array('mlp'),input=np.array('h'),mean=np.zeros(2),scale=np.ones(2),
            w1=np.eye(2,dtype=np.float32),b1=np.zeros(2,dtype=np.float32),
            w2=np.array([[1.,1.]],dtype=np.float32),b2=np.zeros(1,dtype=np.float32)))
        self.assertEqual(h.logits([[2.,-3.]])[0],2.)
        with self.assertRaises(ValueError):h.logits([[1.]])
        raw=Head(dict(kind=np.array('linear'),input=np.array('raw_prediction'),mean=np.zeros(1),scale=np.ones(1),weight=np.ones(1),bias=np.array(0.)))
        with self.assertRaises(ValueError):raw.predict_proba([[1.]])
        self.assertEqual(raw.logits([[1.]],raw_prediction=[2.])[0],2.)


if __name__=='__main__':unittest.main()
