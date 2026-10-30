import unittest
import numpy as np
from energy_trajectory import per_atom,record_energy,block_medians,fixed_cohort,changes


class EnergyTests(unittest.TestCase):
    def test_per_atom_not_total_energy(self):
        self.assertEqual(per_atom(-16.,8),-2.)
        with self.assertRaises(ValueError):per_atom(1.,0)
        with self.assertRaises(ValueError):per_atom(float('nan'),8)

    def test_missing_blocks_are_not_interpolated(self):
        a=np.arange(100,dtype=float);a[:6]=np.nan;a[25:30]=np.nan
        x,y,n=block_medians(a)
        self.assertEqual(n.tolist(),[19,20,25,25]);self.assertTrue(np.isnan(y[0]))
        self.assertEqual(y[1],39.5);self.assertEqual(x.tolist(),[12.,37.,62.,87.])

    def test_energy_does_not_require_defined_force_cosine(self):
        self.assertEqual(record_energy(dict(status='stationary',cos_score_force=None,energy_eV=-16.),8),-2.)
        self.assertIsNone(record_energy(dict(status='unknown',energy_eV=None),8))

    def test_unknown_breaks_adjacent_changes(self):
        result=changes([3.,2.,np.nan,0.,1.])
        self.assertEqual(result['adjacent_pairs'],2)
        self.assertEqual(result['increases'],1);self.assertEqual(result['decreases'],1)

    def test_fixed_cohort_and_final_reference(self):
        a=np.array([[4.,3.,2.,1.],[8.,np.nan,3.,4.]])
        relative=a-a[:,-1,None]
        ids,v=fixed_cohort(relative,1)
        self.assertEqual(ids.tolist(),[0]);np.testing.assert_equal(v,[[2.,1.,0.]])
        self.assertEqual(relative[1,2],-1.)


if __name__=='__main__':unittest.main()
