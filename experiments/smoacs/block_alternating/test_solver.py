import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from solver import solve,projected_gradient,gram_check,hinge_subgradient


class SolverTests(unittest.TestCase):
    def test_hinge_stationarity_and_nonstationary_rejection(self):
        gradient,coefficient=hinge_subgradient([.5],2.46,[2.],[1.],[0.],[1.])
        np.testing.assert_allclose(gradient,0.,atol=1e-12);self.assertAlmostEqual(coefficient,-.5)
        gradient,_=hinge_subgradient([.5],2.54,[2.],[1.],[0.],[1.])
        self.assertGreaterEqual(abs(gradient[0]),1.)
        with self.assertRaises(ValueError):hinge_subgradient([.5],2.5,[2.],[1.],[0.],[1.])

    def test_gaussian_forward_and_distance_gradients(self):
        import sys
        from pathlib import Path
        import torch
        from gaussian import ExactGaussianAverage
        sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'third_party/SMOACS/src'))
        from crystalformer.models.indexed_lattice_multi_head_attention import AverageGaussiansBySoftmaxXYFunc_
        x=torch.tensor([.3,1.2,2.4],dtype=torch.float64).reshape(1,1,3,1).requires_grad_()
        y=torch.tensor([-.7,-1.1],dtype=torch.float64).reshape(1,2,1,1).requires_grad_()
        w=torch.arange(16,dtype=torch.float64).reshape(2,2,4).div(10).requires_grad_()
        for projection in (None,w):
            original=AverageGaussiansBySoftmaxXYFunc_.apply(3.,4,x,y,-2,projection,1.,'',False)
            exact=ExactGaussianAverage.apply(3.,4,x,y,-2,projection)
            torch.testing.assert_close(original,exact,rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(lambda x,y,w:ExactGaussianAverage.apply(3.,4,x,y,-2,w),(x,y,w)))
        zero=torch.zeros_like(x,requires_grad=True)
        ExactGaussianAverage.apply(3.,4,zero,y,-2).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_interior_stationarity(self):
        x,report=solve(lambda x:(float(((x-.4)**2).sum()),2*(x-.4)),[.9,.1],[0,0],[1,1])
        self.assertTrue(report['converged']);np.testing.assert_allclose(x,.4,atol=1e-4)

    def test_boundary_stationarity_reports_raw_gradient(self):
        x,report=solve(lambda x:(float(((x+2)**2).sum()),2*(x+2)),[.5],[0],[1])
        self.assertTrue(report['converged']);self.assertGreater(report['raw_gradient'],1)
        self.assertEqual(projected_gradient([0],[-2],[0],[1])[0],-2)

    def test_budget_is_not_convergence(self):
        _,report=solve(lambda x:(float((x[0]-1)**4+100*(x[1]+2)**4),
                                 np.array([4*(x[0]-1)**3,400*(x[1]+2)**3])),
                       [10,10],[-20,-20],[20,20],maxiter=1)
        self.assertFalse(report['converged'])

    def test_false_optimizer_stop_restarts_after_actual_descent(self):
        def stopped(fun,x,**kwargs):return SimpleNamespace(x=x,nit=0,message='false ftol stop')
        with patch('solver.minimize',side_effect=stopped):
            x,report=solve(lambda x:(float(((x-.01)**2).sum()),2*(x-.01)),[0.],[-1.],[1.])
        self.assertTrue(report['converged']);self.assertAlmostEqual(x[0],.01)
        self.assertTrue(any('measured descent' in message for message in report['optimizer_message']))

    def test_shared_evaluation_budget_is_enforced(self):
        _,report=solve(lambda x:(float(((x-.4)**2).sum()),2*(x-.4)),[.9],[0],[1],maxfun=1)
        self.assertFalse(report['converged']);self.assertEqual(report['evaluations'],1)

    def test_float32_large_cell_and_actual_distortion(self):
        cell=np.array([[73.55354789,0,0],[-71.320,.0+17.98,0],[-89.53,-2.,13.]])
        reconstructed=cell.astype(np.float32)
        self.assertTrue(gram_check(cell,reconstructed)['passed'])
        self.assertTrue(gram_check(cell,cell@np.array([[0,1,0],[1,0,0],[0,0,-1]]))['passed'])
        self.assertFalse(gram_check(cell,cell*1.001)['passed'])


if __name__=='__main__':unittest.main()
