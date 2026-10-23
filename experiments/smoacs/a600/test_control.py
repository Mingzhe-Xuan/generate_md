import importlib.util
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adam_cycles'))
from adam_policy import cosine_fraction
spec=importlib.util.spec_from_file_location('gap_compare',Path(__file__).with_name('compare.py'))
compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)


class ControlTests(unittest.TestCase):
    def test_single_cycle_export_never_requires_cycle3(self):
        import export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=['a','b','c','d'];rows=[]
            for identifier in ids:
                source=root/(identifier+'.vasp');source.write_text('immutable geometry')
                point=dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),num_atoms=2,
                    formula='NaCl',elements=['Na','Cl'],gap=2.5,eform=-1.,valid=True,neutral=True,separated=True)
                rows.append(dict(candidate_id=identifier,A=point,a_updates=600,status='not_eligible'))
            (root/'complete.json').write_text(json.dumps(dict(completed=True)))
            (root/'results.json').write_text(json.dumps(dict(protocol=dict(ids=ids),cycles=[dict(cycle=1,candidates=rows)])))
            with patch.dict(os.environ,SLURM_JOB_ID='unit'),patch('sys.argv',['export','--run',str(root),'--cycles','1']):export.main()
            self.assertTrue((root/'evaluation-cycle1/inventory-249/inventory.json').exists())
            self.assertFalse((root/'evaluation-cycle3').exists())

    def test_both_controls_and_comparison_dependencies(self):
        spec=importlib.util.spec_from_file_location('a600_submit',Path(__file__).with_name('submit.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'results/smoacs/control'
            with patch('pathlib.Path.cwd',return_value=root),patch('sys.argv',['submit','--output',str(target)]),\
                 patch.object(module.subprocess,'check_output',side_effect=['10','11','12','13','14']) as call:
                module.main();module.main();self.assertEqual(call.call_count,5)
                self.assertIn('--dependency=afterok:11:13',call.call_args_list[-1].args[0])
                self.assertEqual(call.call_args_list[0].kwargs['env']['SMOACS_A600_SCHEDULE'],'per_phase')
                self.assertEqual(call.call_args_list[2].kwargs['env']['SMOACS_A600_SCHEDULE'],'restart200')

    def test_600_schedule_and_matching_restarts(self):
        self.assertEqual(cosine_fraction(0,600),1.)
        self.assertAlmostEqual(cosine_fraction(300,600),.5)
        self.assertGreater(cosine_fraction(599,600),0.)
        self.assertEqual(cosine_fraction(600,600),0.)
        self.assertEqual([cosine_fraction(i%200,200) for i in range(600)],
                         [cosine_fraction(i,200) for _ in range(3) for i in range(200)])

    def test_errors_are_to_target_not_model_difference(self):
        rows=[dict(candidate_id='a',stage='after',crystalformer_gap=2.),dict(candidate_id='b',stage='after',crystalformer_gap=None)]
        values=compare.errors(rows,['a','b'],'crystalformer','after');r=compare.describe(values)
        self.assertEqual(r['mean_absolute_target_error'],.5);self.assertEqual(r['unknown'],1)
        self.assertEqual(r['denominator'],2)
        with self.assertRaises(ValueError):compare.errors(rows+rows,['a','b'],'crystalformer','after')


if __name__=='__main__':unittest.main()
