from types import SimpleNamespace
import unittest
import torch
from boundary import Boundary
from comparison import eligible,classify,native_force_norm


class CoreTests(unittest.TestCase):
    def test_boundary_exact_indices_values_and_rng(self):
        def native(x):return x+torch.randn_like(x),x.clone()
        module=SimpleNamespace(_mask_replace=native);captures=[]
        torch.manual_seed(123);expected=[native(torch.tensor([float(i)])) for i in range(2000)]
        torch.manual_seed(123)
        with Boundary(module,torch,lambda i,s,m:captures.append((i,s.clone(),m.clone()))) as obs:
            actual=[module._mask_replace(torch.tensor([float(i)])) for i in range(2000)]
        self.assertEqual(obs.calls,2000);self.assertEqual([c[0] for c in captures],[800,1000])
        self.assertEqual(float(captures[0][2]),1599.)
        for pair,reference in zip(actual,expected):
            for a,b in zip(pair,reference):torch.testing.assert_close(a,b)
        self.assertIs(module._mask_replace,native)

    def test_capture_rng_mutation_rejected_and_restored(self):
        native=lambda:(torch.zeros(1),torch.zeros(1));module=SimpleNamespace(_mask_replace=native)
        with self.assertRaisesRegex(RuntimeError,'RNG'):
            with Boundary(module,torch,lambda *args:torch.randn(1)):
                for i in range(1600):module._mask_replace()
        self.assertIs(module._mask_replace,native)

    def test_both_force_criteria(self):
        row=dict(status='converged',converged=True,max_raw_force=.009,max_filter_force=.009)
        self.assertTrue(eligible(row));self.assertFalse(eligible(dict(row,max_raw_force=.02)))
        self.assertFalse(eligible(dict(row,max_filter_force=float('nan'))))
        self.assertFalse(eligible(dict(row,status='unconverged')))

    def test_serialized_force_norm_keeps_native_float32_arithmetic(self):
        import numpy as np
        forces=np.array([[.003,.004,.007]],dtype=np.float32)
        native=float(np.linalg.norm(forces,axis=1).max())
        self.assertEqual(native_force_norm(forces.tolist()),native)
        self.assertNotEqual(float(np.linalg.norm(np.array(forces.tolist()),axis=1).max()),native)
        with self.assertRaises(ValueError):native_force_norm([[.1,.2,.3]])

    def test_refinement_keeps_coarse_endpoints_above_raw_threshold(self):
        from refine import continuation_pose
        pose=dict(numbers=[13,13],cell=[[5,0,0],[0,5,0],[0,0,5]],frac=[[0,0,0],[.5,.5,.5]])
        row=dict(force_eligible=False,result=dict(pose,status='converged',max_raw_force=.02))
        self.assertEqual(continuation_pose(row),pose)
        with self.assertRaises(ValueError):continuation_pose(dict(result=dict(pose,status='failed')))

    def test_energy_and_geometry_are_separate(self):
        self.assertFalse(classify(0.,False,True)['consistent_endpoint'])
        self.assertFalse(classify(.002,True,True)['consistent_endpoint'])
        self.assertTrue(classify(.0001,True,True)['consistent_endpoint'])
        self.assertEqual(classify(-.02,False,False)['energy_winner'],'early')

    def test_geometry_translation_permutation_and_distortion(self):
        from pymatgen.core import Structure,Lattice
        from geometry import compare
        a=Structure(Lattice.cubic(5.64),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
        b=Structure(a.lattice,['Cl','Na'],[[1.7,.8,-.1],[1.2,.3,-.6]])
        result=compare(a,b)
        self.assertTrue(result['strict_match']);self.assertLess(result['aligned_internal_RMS_A'],1e-8)
        self.assertIsNone(result['ordered'])
        b=Structure(a.lattice,['Na','Cl'],[[0,0,0],[.25,.25,.25]])
        self.assertFalse(compare(a,b)['strict_match'])
        b=Structure(Lattice.cubic(6.3),['Na','Cl'],a.frac_coords)
        self.assertFalse(compare(a,b)['strict_match'])

    def test_endpoint_support_and_integer_basis_invariance(self):
        import numpy as np
        from geometry import support
        numbers=[11,17];cell=np.eye(3)*3.;frac=np.array([[0,0,0],[.5,.5,.5]])
        self.assertEqual(support(numbers,cell,frac)['reasons'],[])
        transform=np.array([[1,0,0],[100,1,0],[0,0,1]])
        self.assertEqual(support(numbers,transform@cell,frac@np.linalg.inv(transform))['reasons'],[])
        self.assertIn('density_guard_volume_per_atom_below_2A3',support(numbers,np.eye(3),frac)['reasons'])
        self.assertIn('collision_below_0.5A',support(numbers,cell,[[0,0,0],[.01,0,0]])['reasons'])

    def test_same_crystal_in_different_eight_atom_supercells(self):
        from pymatgen.core import Lattice,Structure
        from physical import match
        base=Structure.from_spacegroup('Fm-3m',Lattice.cubic(4.),['Al'],[[0,0,0]])
        first=base.copy();first.make_supercell([2,1,1])
        second=base.copy();second.make_supercell([[1,1,0],[-1,1,0],[0,0,1]])
        self.assertEqual(len(first),8);self.assertEqual(len(second),8)
        self.assertTrue(match(first,second)['physical_strict_match'])
        second.scale_lattice(second.volume*1.1**3)
        self.assertFalse(match(first,second)['physical_strict_match'])

    def test_filter_force_reconstruction_at_changed_cell(self):
        import numpy as np
        from ase import Atoms
        from ase.filters import ExpCellFilter
        from ase.calculators.singlepoint import SinglePointCalculator
        atoms=Atoms('NaCl',cell=np.eye(3)*5.64,scaled_positions=[[0,0,0],[.5,.5,.5]],pbc=True)
        first=ExpCellFilter(atoms)
        atoms.set_cell(np.diag([5.8,5.4,5.7]),scale_atoms=True)
        forces=np.array([[.02,.03,.01],[-.02,-.03,-.01]])
        stress=np.arange(6)*.001
        atoms.calc=SinglePointCalculator(atoms,energy=-5.,forces=forces,stress=stress)
        expected=first.get_forces()
        clone=atoms.copy();clone.calc=SinglePointCalculator(clone,energy=-5.,forces=forces,stress=stress)
        rebuilt=ExpCellFilter(clone);rebuilt.orig_cell=np.eye(3)*5.64
        np.testing.assert_allclose(rebuilt.get_forces(),expected,atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
