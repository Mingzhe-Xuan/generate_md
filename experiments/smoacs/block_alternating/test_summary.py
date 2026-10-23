import unittest
from summarize import summarize


class SummaryTests(unittest.TestCase):
    def test_partial_budget_is_not_joint_convergence(self):
        report=summarize(dict(protocol=dict(ids=['a','b','c','d']),candidates=[dict(candidate_id='a',status='A_unconverged')]),{}, {})
        self.assertEqual(report['denominator'],4);self.assertFalse(report['processing_completed'])
        self.assertEqual(report['joint_confirmed'],0)

    def test_joint_requires_all_independent_residual_checks(self):
        row=dict(candidate_id='a',status='jointly_converged',a_report=dict(converged=True,projected_gradient=.0001),
            b_result=dict(status='converged',max_filter_force=.009),b_endpoint_a_projected_gradient=.0001,
            b_endpoint_a_autograd_gradient=.0001,
            stable_rounds=3,a_geometry_rms=0.,a_cell_relative=0.,b_geometry_rms=0.,b_cell_relative=0.,feedback_check=dict(passed=True))
        data=dict(protocol=dict(ids=['a','b','c','d']),candidates=[row])
        self.assertEqual(summarize(data,{}, {})['joint_confirmed'],1)
        row['b_result']['max_filter_force']=.02
        with self.assertRaises(ValueError):summarize(data,{}, {})
        row['status']='active';data['candidates'].append(row.copy())
        with self.assertRaises(ValueError):summarize(data,{}, {})


if __name__=='__main__':unittest.main()
