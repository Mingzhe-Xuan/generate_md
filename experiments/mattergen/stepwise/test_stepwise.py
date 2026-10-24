import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('stepwise_comparison',Path(__file__).with_name('report.py'))
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class StepwiseReportTests(unittest.TestCase):
    def test_missing_and_duplicate_ids_rejected(self):
        row=dict(candidate_id='a',stage='after')
        with self.assertRaises(ValueError):
            report.select([row],['a','b'],'after')
        with self.assertRaises(ValueError):
            report.select([row,row],['a'],'after')

    def test_unknowns_and_real_error_metrics(self):
        rows=[]
        for gap,energy,valid in [(2.5,-1,True),(0,-.1,True),(None,None,None)]:
            row=dict(structure_valid=valid)
            for model in ('crystalformer','alignn'):
                row.update({model+'_gap':gap,model+'_eform':energy,
                    model+'_flags':dict(gap=None if gap is None else 2.3<=gap<=2.7,
                        eform=None if energy is None else energy<-.5,
                        joint=None if gap is None else 2.3<=gap<=2.7 and energy<-.5 and valid)})
            rows.append(row)
        result=report.summarize(rows)
        self.assertEqual(result['crystalformer_joint'],dict(passed=1,unknown=1,denominator=3))
        self.assertEqual(result['crystalformer_gap_MAE'],1.25)
        self.assertAlmostEqual(result['alignn_gap_RMSE'],math.sqrt(3.125))

    def test_global_budget_force_and_feedback_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ids=list('abcd')
            def write(name,value):
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            protocol=dict(ids=ids,cycles=200,steps_per_A=1,lr_period=200,total_A_budget=200,
                lr_cycle_policy='global',template_initialization=dict(species_policy='fixed_per_site_onehot'))
            cycles=[]
            for cycle in range(1,201):
                cycles.append(dict(cycle=cycle,candidates=[dict(candidate_id=key,a_updates=1,status='completed',
                    b_result=dict(status='converged',max_filter_force=.005),B=dict(sha256=key+str(cycle)),
                    preceding_B_sha256=key+str(cycle-1) if cycle>1 else None) for key in ids]))
            data=dict(protocol=protocol,cycles=cycles)
            write('results.json',data)
            write('complete.json',dict(completed=True))
            write('gpu-audit.json',dict(passed=True))
            write('fixed-species-audit.json',dict(passed=True))
            write('evaluation-cycle200/cross-prediction/predictions.json',[])
            write('evaluation-cycle200/cross-prediction/metadata.json',dict(crystalformer_control=dict(control_passed=True),controls=[dict(max_errors=[0]),dict(max_errors=[0])]))
            self.assertEqual(report.validated_new_arm(root,200,ids)[1],dict.fromkeys(ids,200))
            cycles[10]['candidates'][0]['preceding_B_sha256']='wrong'
            write('results.json',data)
            with self.assertRaisesRegex(ValueError,'preceding'):
                report.validated_new_arm(root,200,ids)
            cycles[10]['candidates'][0]['preceding_B_sha256']='a10'
            cycles[10]['candidates'][0]['b_result']['max_filter_force']=.02
            write('results.json',data)
            with self.assertRaisesRegex(ValueError,'force-converged'):
                report.validated_new_arm(root,200,ids)
            cycles[10]['candidates'][0]['status']='B_unconverged'
            write('results.json',data)
            with self.assertRaisesRegex(ValueError,'advanced'):
                report.validated_new_arm(root,200,ids)
