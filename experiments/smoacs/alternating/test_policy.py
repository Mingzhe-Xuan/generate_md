import unittest
import io
import tempfile
from pathlib import Path
import numpy as np
from policy import schedule,stop_status,geometry_change


class PolicyTests(unittest.TestCase):
    def test_rpc_keeps_library_banners_out_of_protocol(self):
        from run import read_response
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stdout.log'
            result=read_response(io.StringIO('library banner\n__AB_RPC__{"ready":true}\n'),path)
            self.assertEqual(result,{'ready':True})
            self.assertIn('library banner',path.read_text())
            with self.assertRaises(RuntimeError):read_response(io.StringIO(''),path)

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
        self.assertIsNone(stop_status(.0001,0,.0001,5,True,True,a_rms=.01))
        self.assertIsNone(stop_status(1,0,0,10,True,True,atom_step=.001))

    def test_periodicity_and_translation(self):
        cell=np.eye(3)*3;coords=np.array([[0.,0,0],[.5,.5,.5]])
        rms,relative=geometry_change(cell,coords,cell,coords+np.array([1.1,.1,.1]))
        self.assertAlmostEqual(rms,0.)
        self.assertEqual(relative,0.)


if __name__=='__main__':unittest.main()
