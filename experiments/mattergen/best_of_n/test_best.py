import importlib.util
from pathlib import Path
import unittest
import numpy as np

from policy import AMPLITUDES,best,displacement,eligible


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class BestPolicyTests(unittest.TestCase):
    def row(self,index,gap=2.55,energy=-1,force=.005):
        return dict(branch_index=index,crystalformer_gap=gap,crystalformer_eform=energy,structure_valid=True,
            relaxation=dict(status='converged',converged=True,max_filter_force=force,max_raw_atomic_force=force,final_energy_eV=-10,num_atoms=4))

    def test_seeded_exact_RMS_bounded_zero_translation(self):
        self.assertEqual([AMPLITUDES.count(v) for v in (.05,.10,.20)],[2,6,2])
        for n in (4,8,12,20):
            for value in (.05,.10,.20):
                delta=displacement(n,value,[42,n,10])
                np.testing.assert_array_equal(delta,displacement(n,value,[42,n,10]))
                np.testing.assert_allclose(delta.mean(0),0,atol=1e-15)
                self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),value)
                self.assertLessEqual(np.linalg.norm(delta,axis=1).max(),2*value)
        np.testing.assert_array_equal(displacement(4,0,42),np.zeros((4,3)))

    def test_force_and_energy_gates_precede_gap_ranking(self):
        invalid_force=self.row(1,2.5,force=.02)
        invalid_energy=self.row(2,2.5,energy=-.5)
        winner=self.row(4,2.55)
        self.assertIs(best([invalid_force,invalid_energy,self.row(3,2.6),winner]),winner)
        self.assertTrue(eligible(self.row(1,force=.01)))
        row=self.row(1)
        row['relaxation']['max_raw_atomic_force']=.02
        self.assertFalse(eligible(row))
        row=self.row(1)
        row['relaxation']['converged']=False
        self.assertFalse(eligible(row))

    def test_baseline_excluded_prefix_and_no_ALIGNN_leak(self):
        first=self.row(1,2.8)
        second=self.row(2,2.51)
        zero=self.row(0,2.5)
        self.assertIs(best([zero,first,second],prefix=1),first)
        self.assertIs(best([zero,first,second]),second)
        first['alignn_gap']=2.5
        second['alignn_gap']=100
        self.assertIs(best([zero,first,second]),second)
        self.assertIsNone(best([zero]))
        self.assertIsNone(best([self.row(1,force=.02)]))

    def test_ties_are_deterministic_and_energy_secondary(self):
        a,b=self.row(2),self.row(1)
        self.assertIs(best([a,b]),b)
        a['relaxation']['final_energy_eV']=-20
        self.assertIs(best([b,a]),a)

    def test_no_eligible_parent_and_missing_independent_predictor_keep_coverage(self):
        report=module('best_test_report',Path(__file__).with_name('report.py'))
        row=self.row(1)
        row.update(crystalformer_flags=dict(gap=True,eform=True,joint=True),alignn_gap=None,alignn_eform=None,
            alignn_flags=dict(gap=None,eform=None,joint=None))
        result=report.selected_summary([row,None,None,None],4)
        self.assertEqual((result['selected'],result['no_eligible']),(1,3))
        self.assertEqual(result['crystalformer']['unknown'],3)
        self.assertEqual(result['alignn']['unknown'],4)
        self.assertIsNone(result['alignn']['MAE'])
        paired=report.paired_errors({'x':row},{'x':row},['x'])
        self.assertEqual(paired['known_parent_ids']['alignn'],[])
        self.assertIsNone(paired['MAE']['alignn']['CF20'])


    def test_per_candidate_gradients_do_not_depend_on_branch_count(self):
        import torch
        path=Path(__file__).resolve().parents[2]/'smoacs/adam_cycles/microbatch.py'
        accumulate=module('best_test_microbatch',path).accumulate
        gradients=[]
        for n in (2,4):
            p=torch.nn.Parameter(torch.arange(1,n+1,dtype=torch.float64))
            _,g=accumulate([p],[(i,i+1) for i in range(n)],torch.ones(n,dtype=torch.bool),1,lambda a,b:p[a:b]**2)
            gradients.append(g[0][:2])
        torch.testing.assert_close(gradients[0],gradients[1])

    def test_periodic_collision_and_parent_displacement_projection(self):
        import torch
        geometry=module('best_test_geometry',Path(__file__).resolve().parents[1]/'cf_templates/branch_geometry.py')
        cell=torch.eye(3)*4
        self.assertFalse(geometry.periodic_separated(cell,torch.tensor([[.01,0,0],[.99,0,0]])))
        self.assertTrue(geometry.periodic_separated(cell,torch.tensor([[.1,0,0],[.5,0,0]])))
        ref=torch.tensor([[.2,.2,.2],[.6,.6,.6]])
        coords=torch.nn.Parameter(ref+.4)
        context=dict(targets=[torch.ones(1,3),torch.ones(1,3),coords],data={},batch=torch.tensor([0,0]),slices=[slice(0,2)],
            lower=torch.ones(1,3)*2,upper=torch.ones(1,3)*10,angle_lower=torch.ones(1,3)*60,angle_upper=torch.ones(1,3)*120)
        geometry.install(context,ref.numpy(),cell[None].numpy(),np.array([[4,4,4]]),np.array([[90,90,90]]))
        context['project_after_update']()
        self.assertLessEqual(float(torch.linalg.vector_norm((coords.detach()-ref)@cell,dim=1).max()),.400001)
        torch.testing.assert_close(context['lower'],torch.ones(1,3)*3.92)
        torch.testing.assert_close(context['angle_upper'],torch.ones(1,3)*92)


if __name__=='__main__':
    unittest.main()
