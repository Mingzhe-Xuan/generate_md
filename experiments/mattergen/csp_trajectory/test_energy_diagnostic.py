import unittest
import numpy as np
from energy_diagnostic import sigma_fractional,omitted_blocks,largest_window_drop


class DiagnosticTests(unittest.TestCase):
    def test_smooth_noise_at_midpoint(self):
        values=sigma_fractional(np.array([499,500,501]))
        self.assertAlmostEqual(values[1],np.sqrt(.01*5)/2)
        self.assertAlmostEqual(values[1]/values[0],values[2]/values[1])
        self.assertTrue(np.all(np.diff(values)<0))

    def test_block_omission_is_display_rule(self):
        a=np.ones((2,50));a[0,:6]=np.nan;a[1,:5]=np.nan
        np.testing.assert_array_equal(omitted_blocks(a),[[True,False],[False,False]])

    def test_window_drop_and_missing_rejection(self):
        a=np.r_[np.ones(100),np.zeros(100)]
        result=largest_window_drop(a,width=20,minimum=16)
        self.assertEqual(result['median_change_eV_atom'],-1.)
        self.assertTrue(90<=result['center_step']<=110)
        self.assertIsNone(largest_window_drop(np.full(200,np.nan),width=20,minimum=16))


if __name__=='__main__':unittest.main()
