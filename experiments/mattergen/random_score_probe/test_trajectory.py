import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch


def load(name):
    spec=importlib.util.spec_from_file_location('random_probe_test_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

m=load('trajectory_math')
Observer=load('observer').Observer


class RandomProbeTests(unittest.TestCase):
    def test_observer_keeps_rng_values_and_predictor_selection(self):
        def native(x,t):return x+torch.randn_like(x)
        sampler=SimpleNamespace(_score_fn=native,_n_steps_corrector=1,_correctors={'pos':object()})
        captured=[]
        obs=Observer(sampler,[0,2],lambda x,t,y,stage,step:captured.append((stage,step,x.clone(),y.clone())),torch)
        torch.manual_seed(55)
        baseline=[native(torch.ones(2),torch.tensor([t])) for t in range(6)]
        torch.manual_seed(55)
        trial=[obs(torch.ones(2),torch.tensor([t])) for t in range(6)]
        for a,b in zip(baseline,trial):torch.testing.assert_close(a,b)
        self.assertEqual([(r[0],r[1]) for r in captured],[('prior',0),('predictor',0),('predictor',2)])
        self.assertEqual(obs.calls,6)
        with self.assertRaises(ValueError):Observer(SimpleNamespace(_score_fn=native,_n_steps_corrector=2,_correctors={}),[],None,torch)

    def test_rng_mutating_observer_rejected(self):
        sampler=SimpleNamespace(_score_fn=lambda x,t:x,_n_steps_corrector=1,_correctors={'a':1})
        obs=Observer(sampler,[0],lambda *args:torch.randn(1),torch)
        with self.assertRaisesRegex(RuntimeError,'RNG'):obs(torch.ones(1),torch.ones(1))

    def test_masks_unsupported_cell_and_density(self):
        self.assertIn('masked_atom_types',m.eligibility([101,8],np.eye(3)*5))
        self.assertIn('unsupported_atom_types',m.eligibility([95,8],np.eye(3)*5))
        self.assertIn('invalid_or_nearly_singular_cell',m.eligibility([8,8],np.diag([0,5,5])))
        self.assertIn('density_guard_volume_per_atom_below_2A3',m.eligibility([8,8],np.eye(3)))
        self.assertEqual(m.eligibility([8,8],np.eye(3)*5),[])

    def test_covector_displacement_and_rotation(self):
        cell=np.array([[3.,.2,.1],[.4,5.,.3],[.2,.1,7.]])
        force=np.array([[1.,2.,3.],[-1.,-2.,-3.]])
        score=force@cell.T
        r=m.comparison(score,cell,force)
        self.assertAlmostEqual(r['cos_score_force'],1.)
        self.assertLess(r['cos_drift_force'],1.)
        rot=np.array([[0.,1,0],[-1,0,0],[0,0,1]])
        rotated=m.comparison(score,cell@rot,force@rot)
        self.assertAlmostEqual(rotated['cos_score_force'],r['cos_score_force'])
        self.assertAlmostEqual(rotated['cos_drift_force'],r['cos_drift_force'])
        self.assertLess(r['score_energy_directional_derivative'],0.)
        self.assertIsNone(m.comparison(score,cell,np.zeros_like(force))['cos_score_force'])

    def test_coverage_unknown_not_zero(self):
        rows=[dict(stage='prior',step=0,t=1.,trajectory=i,status='unknown',masked_atoms=4,num_atoms=4) for i in range(16)]
        rows += [dict(stage='predictor',step=999,t=.001,trajectory=i,status='compared',masked_atoms=0,num_atoms=4,
                      cos_score_force=.6,cos_drift_force=.8) for i in range(16)]
        summary=m.summarize(rows)
        self.assertEqual(summary[0]['cos_score_force']['n'],0)
        self.assertIsNone(summary[0]['cos_score_force']['mean'])
        self.assertEqual(summary[1]['cos_score_force']['n'],16)
        self.assertAlmostEqual(summary[1]['cos_score_force']['mean'],.6)
        self.assertEqual(summary[0]['masked_atom_fraction_mean'],1.)


if __name__=='__main__':unittest.main()
