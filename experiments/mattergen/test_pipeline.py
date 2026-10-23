import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fetch import digest
from report import audit


class PipelineTests(unittest.TestCase):
    def test_full128_report_retains_failed_B_and_unknown_hull(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selection=root/'selection';ids=[f'mattergen-{i:04d}.vasp' for i in range(128)]
            def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            source_spec=json.loads(Path(__file__).with_name('sources.json').read_text())
            write(root/'generated/complete.json',dict(count=128,completed=True,sampling_steps=1000,source=source_spec,conditioning={'dft_band_gap':2.5},guidance=2.))
            write(root/'generated/gpu-audit.json',dict(all_cuda=True,score_calls=4000,finite_positions_and_cells=True))
            rows=[];sun=[];candidates=[]
            for i,identifier in enumerate(ids):
                source=root/'structures'/identifier;source.parent.mkdir(exist_ok=True);source.write_text('fixture'+identifier)
                candidates.append(dict(candidate_id=identifier,source=str(source),sha256=digest(source)))
                b=root/'relaxation'/Path(identifier).stem/'relaxed.vasp';b.parent.mkdir(parents=True);b.write_text('relaxed fixture'+identifier)
                write(b.with_name('result.json'),dict(input_sha256=digest(source),status='converged' if i<127 else 'failed',relaxed_path=str(b),relaxed_sha256=digest(b)))
                for stage in ('before','after'):
                    known=stage=='before' or i<127
                    rows.append(dict(candidate_id=identifier,stage=stage,status='completed' if known else 'not_eligible_relaxation',input_sha256=digest(source),geometry_sha256=digest(b),structure_valid=known if known else None,
                        crystalformer_flags=dict(gap=known if known else None,eform=known if known else None,joint=known if known else None),
                        alignn_flags=dict(gap=False if known else None,eform=known if known else None,joint=False if known else None)))
                sun.append(dict(candidate_id=identifier,stable=True if i<126 else None,unique=True if i<127 else None,novel=False if i==0 else True if i<127 else None,sun=False if i in (0,127) else True if i<126 else None))
            write(root/'inventory-249/inventory.json',dict(denominator=128,candidates=candidates,generation_manifest_sha256=digest(root/'generated/complete.json')))
            write(root/'cross-prediction/metadata.json',dict(denominator=128,controls=[{},{}],crystalformer_control=dict(control_passed=True)))
            write(root/'cross-prediction/predictions.json',rows);write(root/'SUN/candidates.json',sun);write(root/'SUN/summary.json',{})
            write(selection/'mattersim-inventory.json',{})
            result=audit(root,selection)
            self.assertTrue(result['verified_end_to_end']);self.assertEqual(result['after_B']['crystalformer_gap']['pass_count'],127)
            self.assertEqual(result['after_B']['crystalformer_gap']['denominator'],128)
            self.assertEqual(result['SUN']['sun']['pass_count'],125);self.assertEqual(result['SUN']['stable']['unknown_count'],2)
            rows[-1]['candidate_id']=ids[0];write(root/'cross-prediction/predictions.json',rows)
            with self.assertRaises(ValueError):audit(root,selection)

    def test_conditional_restoration_refuses_unverified_report_and_foreign_job(self):
        import resume_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);experiment=root/'results/smoacs/full128';checkpoint=experiment/'alternating/checkpoint.pt'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'saved optimizer')
            manifest=root/'pause.json';manifest.write_text(json.dumps(dict(experiment_root=str(experiment),checkpoint_sha256=digest(checkpoint),held_jobs=[312])))
            report=root/'report.json';report.write_text(json.dumps(dict(denominator=128,verified_end_to_end=False)))
            with patch.object(resume_jobs.subprocess,'check_output') as called:
                with self.assertRaises(ValueError):resume_jobs.release(manifest,report)
                called.assert_not_called()
            report.write_text(json.dumps(dict(denominator=128,verified_end_to_end=True,reference_selection='same',prediction_sha256='pred',sun_sha256='sun')))
            with patch.object(resume_jobs,'audit',return_value=dict(prediction_sha256='pred',sun_sha256='sun')),\
                 patch('pathlib.Path.cwd',return_value=root),patch.object(os,'getuid',return_value=1010,create=True),\
                 patch.object(resume_jobs.subprocess,'check_output',return_value='JobName=full128-AB UserId=other(999) WorkDir='+str(root)+' JobState=PENDING Reason=JobHeldUser'),\
                 patch.object(resume_jobs.subprocess,'run') as called:
                with self.assertRaisesRegex(ValueError,'owner differs'):resume_jobs.release(manifest,report)
                called.assert_not_called()

    def test_submission_idempotency_and_dependencies(self):
        path=Path(__file__).with_name('submit.py');spec=importlib.util.spec_from_file_location('mattergen_submit_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'pause.json';manifest.write_text('{}');output=root/'results/mattergen/test'
            with patch('pathlib.Path.cwd',return_value=root),patch('sys.argv',['submit','--output',str(output),'--setup-job','318','--paused-manifest',str(manifest)]),\
                 patch.object(module.subprocess,'check_output',side_effect=['100\n','101\n','102\n']) as called:
                module.main();module.main();self.assertEqual(called.call_count,3)
                self.assertIn('--dependency=afterok:318',called.call_args_list[0].args[0])
                self.assertIn('--dependency=afterok:100',called.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:101',called.call_args_list[2].args[0])


if __name__=='__main__':unittest.main()
