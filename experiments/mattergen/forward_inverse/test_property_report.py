import unittest
import numpy as np
from property_report import controls_valid,check_endpoint,interval_by_parent,summarize_rows


class PropertyReportTests(unittest.TestCase):
    def test_convergence_invariants_reject_favorable_but_invalid_endpoint(self):
        row=dict(status='converged',numbers=[14,14],cell=np.eye(3).tolist(),forces_eV_A=[[.002,0,0],[0,0,0]])
        with self.assertRaisesRegex(ValueError,'Raw force'):check_endpoint(row,[14,14],np.eye(3),.001)
        row['forces_eV_A']=[[0,0,0],[0,0,0]];row['cell'][0][0]=1.001
        with self.assertRaisesRegex(ValueError,'Fixed cell'):check_endpoint(row,[14,14],np.eye(3),.001)

    def test_gradient_control_checks_each_property_and_graph(self):
        controls=[dict(direction='random',same_graph=True,AD=[1.,0.],FD=[1.,0.],fixed_graph_FD=[1.,0.]) for _ in range(2)]
        policy=dict(FD_steps_A=[1e-4,5e-5],FD_abs=1e-5,FD_rel=.002)
        case=dict(controls=controls)
        self.assertTrue(controls_valid(case,'random',0,policy))
        controls[0]['FD'][1]=.1
        self.assertTrue(controls_valid(case,'random',0,policy));self.assertFalse(controls_valid(case,'random',1,policy))
        controls[0]['same_graph']=False
        self.assertFalse(controls_valid(case,'random',0,policy))

    def test_parent_intervals_and_missing_cohorts(self):
        self.assertIsNone(interval_by_parent({'a':1.}))
        self.assertEqual(interval_by_parent({'a':.1,'b':.1}),[.1,.1])
        policy=dict(directions=['random'],amplitudes_RMS_A=[.01],tolerances_eV_A=[.001],property_order=['gap','ef'])
        row=dict(id='a',direction='random',amplitude_RMS_A=.01,tolerance_eV_A=.001,gradient_QC=[True,False],
                 absolute_response_retention=[None,.1],raw_FD=[0.,1.],relaxed_FD=[0.,.1],geometry_response_gain=.1)
        groups=summarize_rows([row],policy)
        self.assertEqual(groups[0]['ratio_defined_parents'],0)
        self.assertEqual(groups[-1]['eligible_parents'],0)
        self.assertIsNone(groups[-1]['median_absolute_response_retention'])
        with self.assertRaisesRegex(ValueError,'Repeated parent'):summarize_rows([row,row],policy)


if __name__=='__main__':unittest.main()
