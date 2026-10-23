import unittest
import numpy as np
from policy import schedule,stop_status,geometry_change


class PolicyTests(unittest.TestCase):
    def test_schedule_has_nonzero_terminal_learning_rate(self):
        self.assertEqual(schedule(0),(.01,1.))
        self.assertAlmostEqual(schedule(199)[0],.0001495)
        self.assertEqual(schedule(200),(.0001,.1))
        self.assertEqual(schedule(300),(.0001,.1))

    def test_true_joint_and_cycle_are_distinct(self):
        self.assertEqual(stop_status(.0001,0,.0001,5,True,True),'jointly_converged')
        self.assertIsNone(stop_status(1,1,0,20,True,True))
        self.assertEqual(stop_status(1,0,0,10,True,True),'nonjoint_physical_cycle')
        self.assertIsNone(stop_status(0,0,0,20,False,True))
        self.assertIsNone(stop_status(0,0,0,20,True,False))

    def test_periodicity_and_translation(self):
        cell=np.eye(3)*3;coords=np.array([[0.,0,0],[.5,.5,.5]])
        rms,relative=geometry_change(cell,coords,cell,coords+np.array([1.1,.1,.1]))
        self.assertAlmostEqual(rms,0.)
        self.assertEqual(relative,0.)


if __name__=='__main__':unittest.main()
