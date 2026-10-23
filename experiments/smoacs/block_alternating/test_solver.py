import unittest
import numpy as np
from solver import solve,projected_gradient,gram_check


class SolverTests(unittest.TestCase):
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

    def test_float32_large_cell_and_actual_distortion(self):
        cell=np.array([[73.55354789,0,0],[-71.320,.0+17.98,0],[-89.53,-2.,13.]])
        reconstructed=cell.astype(np.float32)
        self.assertTrue(gram_check(cell,reconstructed)['passed'])
        self.assertTrue(gram_check(cell,cell@np.array([[0,1,0],[1,0,0],[0,0,-1]]))['passed'])
        self.assertFalse(gram_check(cell,cell*1.001)['passed'])


if __name__=='__main__':unittest.main()
