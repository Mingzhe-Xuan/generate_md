from pathlib import Path
import tempfile
import unittest

import torch

from compose import PCGrad,project_pair


class PCGradTests(unittest.TestCase):
    def test_known_conflict_and_both_original_references(self):
        p=torch.tensor([1.,0.]);g=torch.tensor([-1.,1.])
        pp,gp,conflict=project_pair(p,g)
        self.assertTrue(conflict)
        torch.testing.assert_close(pp,torch.tensor([.5,.5],dtype=torch.float64))
        torch.testing.assert_close(gp,torch.tensor([0.,1.],dtype=torch.float64))
        self.assertAlmostEqual(float(pp@g.double()),0.)
        self.assertAlmostEqual(float(gp@p.double()),0.)
        self.assertGreaterEqual(float(pp@gp),0.)
        self.assertGreaterEqual(float((pp+gp)@p.double()),0.)
        self.assertGreaterEqual(float((pp+gp)@g.double()),0.)

    def test_nonconflict_zero_and_antiparallel_limits(self):
        for p,g in [([1.,2.],[2.,1.]),([1.,0.],[0.,2.]),([0.,0.],[2.,1.])]:
            p,g=torch.tensor(p),torch.tensor(g)
            pp,gp,conflict=project_pair(p,g)
            self.assertFalse(conflict)
            torch.testing.assert_close(pp,p.double());torch.testing.assert_close(gp,g.double())
        pp,gp,conflict=project_pair(torch.tensor([3.,0.]),torch.tensor([-1.,0.]))
        self.assertTrue(conflict);self.assertEqual(float(pp.norm()+gp.norm()),0.)

    def test_inputs_not_modified_and_nonfinite_rejected(self):
        p=torch.tensor([1.,0.]);g=torch.tensor([-1.,2.]);a,b=p.clone(),g.clone()
        project_pair(p,g);torch.testing.assert_close(p,a);torch.testing.assert_close(g,b)
        with self.assertRaises(ValueError):project_pair(p,torch.tensor([float('nan'),0.]))

    def context(self,path):
        return dict(output=Path(path),ids=['a','b'],slices=[slice(0,1),slice(1,3)])

    def gradients(self):
        p=[torch.zeros(2,3),torch.zeros(2,3),torch.zeros(3,3),torch.zeros(3,98),torch.zeros(2,10)]
        g=[v.clone() for v in p]
        p[0][0,0]=1;g[0][0,0]=-1;g[0][0,1]=1
        p[0][1,0]=100;g[0][1,0]=100
        return p,g

    def test_adapter_is_per_structure_and_keeps_nonconflict_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            c=self.context(directory);fn=PCGrad(c,.2);p,g=self.gradients()
            result=fn(p,g,torch.tensor([True,True]))
            expected_p,expected_g,_=project_pair(p[0][0],.2*g[0][0])
            torch.testing.assert_close(result[0][0],(expected_p+expected_g).float())
            torch.testing.assert_close(result[0][1],p[0][1]+.2*g[0][1],rtol=0,atol=0)
            self.assertTrue(all(torch.count_nonzero(v)==0 for v in result[3:]))
            self.assertEqual(fn.conflicts,1)

    def test_inactive_case_and_fixed_species_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            fn=PCGrad(self.context(directory),.3);p,g=self.gradients()
            result=fn(p,g,torch.tensor([False,True]))
            torch.testing.assert_close(result[0][0],p[0][0]+.3*g[0][0],rtol=0,atol=0)
            self.assertEqual(fn.conflicts,0)
            g[3][0,0]=1
            with self.assertRaises(ValueError):fn(p,g,torch.tensor([True,True]))


if __name__=='__main__':unittest.main()
