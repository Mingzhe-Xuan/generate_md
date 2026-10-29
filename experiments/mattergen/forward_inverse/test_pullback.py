import unittest
import numpy as np
from pullback_math import rms_direction, response


class PullbackTests(unittest.TestCase):
    def test_translation_removed_and_atomic_rms_one(self):
        x=np.array([[1.,2.,3.],[-2.,1.,2.],[.4,.7,-.3]])
        d=rms_direction(x,3)
        np.testing.assert_allclose(d.mean(0),0.,atol=1e-15)
        self.assertAlmostEqual(np.sqrt(np.mean(np.sum(d*d,axis=1))),1.)
        self.assertIsNone(rms_direction(np.ones((3,3)),3))

    def test_linear_property_contracting_map_and_pullback(self):
        n=2;amplitude=.02
        direction=rms_direction(np.array([[1.,2.,0.],[-1.,-2.,0.]]),n)
        gradients=np.array([[[1.,0.,0.],[-1.,0.,0.]],[[0.,2.,0.],[0.,-2.,0.]]])
        raw=gradients.reshape(2,-1)@direction.ravel()
        shrink=.04
        record=response(amplitude*raw,-amplitude*raw,shrink*amplitude*raw,-shrink*amplitude*raw,
                        gradients,gradients,direction,2*amplitude*shrink*direction,amplitude)
        np.testing.assert_allclose(record['raw_FD'],record['raw_AD'],atol=1e-12)
        np.testing.assert_allclose(record['relaxed_FD'],record['finite_map_pullback'],atol=1e-12)
        np.testing.assert_allclose(record['absolute_response_retention'],[shrink,shrink],atol=1e-12)
        self.assertAlmostEqual(record['geometry_response_gain'],shrink)

    def test_zero_property_response_ratio_remains_undefined(self):
        direction=rms_direction([[1.,0.,0.],[-1.,0.,0.]],2)
        record=response(np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2),np.zeros((2,2,3)),
                        np.zeros((2,2,3)),direction,np.zeros((2,3)),.01)
        self.assertEqual(record['absolute_response_retention'],[None,None])


if __name__=='__main__':unittest.main()
