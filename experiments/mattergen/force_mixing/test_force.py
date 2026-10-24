import importlib.util
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'third_party/SMOACS'))
from geometry import combine,energy_surrogate,geometry,property_cartesian_force
from protocol import pose_hash


class ForceMixingTests(unittest.TestCase):
    def test_report_helpers_retain_unknowns_and_paired_ids(self):
        path=Path(__file__).with_name('report.py')
        spec=importlib.util.spec_from_file_location('force_report_test',path)
        report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
        util=report.helpers()
        row=dict(candidate_id='a',crystalformer_gap=2.5,alignn_gap=2.6,
            crystalformer_flags=dict(gap=True,eform=True,joint=True),alignn_flags=dict(gap=True,eform=True,joint=True))
        missing=dict(candidate_id='b',crystalformer_gap=None,alignn_gap=None,
            crystalformer_flags=dict(gap=None,eform=None,joint=None),alignn_flags=dict(gap=None,eform=None,joint=None))
        summary=util.summarize([row,missing])
        self.assertEqual(summary['denominator'],2)
        self.assertEqual(summary['alignn']['unknown'],1)
        self.assertEqual(util.paired([row,missing],[missing,row],['a','b'])['alignn']['n'],1)

    def context(self):
        abc=torch.tensor([[4.2,4.5,4.8],[3.8,4.4,5.1]],dtype=torch.float64)
        angle=torch.tensor([[82.,94.,103.],[91.,88.,97.]],dtype=torch.float64)
        frac=torch.tensor([[.12,.14,.17],[.41,.47,.43],[.21,.13,.18],[.48,.42,.39]],dtype=torch.float64)
        p=[torch.nn.Parameter(t.clone()) for t in (torch.ones_like(abc),(angle-60)/60,frac,torch.zeros(4,98,dtype=torch.float64),torch.zeros(2,10,dtype=torch.float64))]
        return dict(targets=p,base=abc,lower=abc*.8,upper=abc*1.2,angle_lower=angle-10,angle_upper=angle+10,
            batch=torch.tensor([0,0,1,1]))

    def potential(self,cells,frac):
        delta=frac.reshape(2,2,3)[:,1]-frac.reshape(2,2,3)[:,0]
        r=torch.bmm(delta[:,None,:],cells).squeeze(1)
        volume=torch.linalg.det(cells)
        energy=.35*(r*r).sum(1)+.0015*(volume-80.)**2
        forces=torch.stack((.7*r,-.7*r),dim=1).reshape(-1,3)
        gh=.7*delta[:,:,None]*r[:,None,:]+(.003*(volume-80.)*volume)[:,None,None]*torch.linalg.inv(cells).transpose(-1,-2)
        stress=cells.transpose(-1,-2)@gh/volume[:,None,None]
        return energy,forces,stress

    def test_force_and_stress_pullback_matches_exact_energy_gradient(self):
        c=self.context();h,s=geometry(c)
        energy,force,stress=self.potential(h,s)
        expected=torch.autograd.grad(energy.mean(),c['targets'][:3],retain_graph=True)
        pulled=energy_surrogate(h,s,c['batch'],energy,force,stress).mean()
        pulled=pulled+c['targets'][3].sum()*0+c['targets'][4].sum()*0
        actual=torch.autograd.grad(pulled,c['targets'])
        for a,b in zip(actual[:3],expected):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
        self.assertTrue(all(torch.count_nonzero(g)==0 for g in actual[3:]))
        torch.testing.assert_close(property_cartesian_force(expected[2],h,c['batch'],2),force)
        self.assertTrue(torch.allclose(stress,stress.transpose(-1,-2),atol=1e-12))

    def test_native_variable_chain_rule_with_central_finite_differences(self):
        c=self.context();h,s=geometry(c)
        e,f,stress=self.potential(h,s)
        gradients=torch.autograd.grad(energy_surrogate(h,s,c['batch'],e,f,stress).sum(),c['targets'][:3])
        for block in range(3):
            p=c['targets'][block]
            direction=torch.arange(1,p.numel()+1,dtype=p.dtype).reshape(p.shape)
            direction/=direction.norm()
            before=p.detach().clone()
            values=[]
            with torch.no_grad():
                for sign in (1,-1):
                    p.copy_(before+sign*1e-6*direction)
                    values.append(float(self.potential(*geometry(c))[0].sum()))
                p.copy_(before)
            self.assertAlmostEqual((values[0]-values[1])/2e-6,float((gradients[block]*direction).sum()),places=6)

    def test_mixing_sign_scale_and_zero_limits(self):
        physical=[torch.tensor([2.,-3.]),torch.tensor([0.,4.])]
        property_gradient=[torch.tensor([-4.,5.]),torch.tensor([2.,0.])]
        for weight in (.1,.2,.3):
            actual=combine(physical,property_gradient,weight)
            for a,p,g in zip(actual,physical,property_gradient):torch.testing.assert_close(-a,-p+weight*(-g))
        for a,b in zip(combine(physical,property_gradient,0.),physical):torch.testing.assert_close(a,b)
        for a,g in zip(combine([torch.zeros_like(x) for x in physical],property_gradient,.2),property_gradient):torch.testing.assert_close(a,.2*g)
        with self.assertRaises(ValueError):combine(physical,property_gradient,-.1)

    def test_no_affine_atomic_force_double_count(self):
        c=self.context();h,s=geometry(c);energy,force,stress=self.potential(h,s)
        correct=torch.autograd.grad(energy.sum(),c['targets'][0],retain_graph=True)[0]
        pos=torch.bmm(s[:,None,:],h[c['batch']]).squeeze(1)
        wrong=energy_surrogate(h,s,c['batch'],energy,force,stress).sum()-(force.detach()*pos).sum()
        duplicate=torch.autograd.grad(wrong,c['targets'][0])[0]
        self.assertGreater(float((duplicate-correct).abs().max()),.1)

    def test_pose_hash_ignores_order_of_keys_but_rejects_stale_geometry(self):
        a=dict(cell=[[4.,0,0],[0,4.,0],[0,0,4.]],frac=[[.1,.2,.3]],numbers=[14],candidate_id='a')
        b=dict(reversed(list(a.items())))
        self.assertEqual(pose_hash(a),pose_hash(b))
        b['frac']=[[.1,.2,.31]]
        self.assertNotEqual(pose_hash(a),pose_hash(b))

    def test_entry_resolves_geometry_and_controller_dependencies(self):
        spec=importlib.util.spec_from_file_location('force_entry_test',Path(__file__).with_name('run.py'))
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        from solver import gram_check
        self.assertTrue(gram_check(torch.eye(3).numpy(),torch.eye(3).numpy())['passed'])


if __name__=='__main__':unittest.main()
