import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SHARED=Path(__file__).resolve().parents[1]/'adam_cycles'
sys.path.append(str(SHARED))
from microbatch import accumulate,Microbatch
from report import conjunction,indexed,property_stats,sun_stats


class FullComparisonTests(unittest.TestCase):
    def test_full128_export_keeps_failed_and_feedback_failed_rows(self):
        import export
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=[str(i)+'.vasp' for i in range(128)];rows=[]
            source=root/'A.vasp';source.write_text('test A');b=root/'B.vasp';b.write_text('test B')
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            for i in ids:
                a=dict(source=str(source),sha256=sha,num_atoms=2,formula='NaCl',elements=['Na','Cl'],gap=2.5,eform=-1.,valid=True,neutral=True,separated=True)
                rows.append(dict(candidate_id=i,A=a,a_updates=600,status='not_eligible'))
            rows[0].update(status='B_converged_feedback_failed',b_result=dict(candidate_id=ids[0],status='converged',input_sha256=sha,
                relaxed_path=str(b),relaxed_sha256=hashlib.sha256(b.read_bytes()).hexdigest()))
            (root/'complete.json').write_text(json.dumps(dict(completed=True)))
            (root/'results.json').write_text(json.dumps(dict(protocol=dict(ids=ids),cycles=[dict(cycle=1,candidates=rows)])))
            with patch.dict(os.environ,SLURM_JOB_ID='test'),patch('sys.argv',['export','--run',str(root),'--cycles','1']):export.main()
            inventory=json.loads((root/'evaluation-cycle1/inventory-249/inventory.json').read_text())
            self.assertEqual(inventory['denominator'],128);self.assertEqual(len(inventory['candidates']),128)
            self.assertEqual(json.loads((root/'reference-scope.json').read_text())['denominator'],1)

    def test_streamed_global_gradient_and_adam_match_unbatched_with_failures(self):
        import torch
        torch.manual_seed(42)
        for stride in (8,16):
            reference=torch.nn.Parameter(torch.randn(128,3,dtype=torch.float64))
            streamed=torch.nn.Parameter(reference.detach().clone())
            active=torch.ones(128,dtype=torch.bool);active[1::7]=False
            a=torch.optim.Adam([reference],lr=.002);b=torch.optim.Adam([streamed],lr=.002)
            for _ in range(3):
                loss=(reference.sin()+reference.square()).sum(1)
                reference.grad=torch.autograd.grad(loss[active].sum()/128,[reference])[0]
                _,gradients=accumulate([streamed],[(i,min(i+stride,128)) for i in range(0,128,stride)],active,128,
                    lambda i,j:(streamed[i:j].sin()+streamed[i:j].square()).sum(1))
                torch.testing.assert_close(gradients[0],reference.grad,rtol=1e-12,atol=1e-12)
                streamed.grad=gradients[0];a.step();b.step()
                torch.testing.assert_close(streamed,reference,rtol=1e-12,atol=1e-12)

    def test_full_denominator_unknowns_and_joint(self):
        ids=[str(i) for i in range(128)]
        rows=[dict(candidate_id=i,structure_valid=True,crystalformer_flags=dict(gap=True,eform=False,joint=False),
            alignn_flags=dict(gap=False,eform=True,joint=False)) for i in ids]
        rows[-1].update(structure_valid=None,crystalformer_flags=dict(gap=None,eform=None,joint=None),alignn_flags=dict(gap=None,eform=None,joint=None))
        stats=property_stats(rows,ids)
        self.assertEqual(stats['crystalformer_gap']['pass_count'],127)
        self.assertEqual(stats['crystalformer_gap']['denominator'],128)
        self.assertEqual(stats['crystalformer_gap']['unknown_count'],1)
        self.assertEqual(stats['crystalformer_joint']['pass_count'],0)
        self.assertEqual(stats['crystalformer_gap']['possible_rate_bounds'],[127/128,1.])
        sun=[dict(candidate_id=i,stable=True,unique=True,novel=None,sun=None) for i in ids]
        result=sun_stats(sun,rows,ids)
        self.assertEqual(result['sun']['unknown_count'],128)
        self.assertEqual(result['sun_and_crystalformer_joint']['unknown_count'],1)
        self.assertIs(conjunction([False,None]),False)
        with self.assertRaises(ValueError):indexed(rows[:-1],ids)
        with self.assertRaises(ValueError):indexed(rows[:-1]+[rows[0]],ids)

    def test_matrix_scaled_gram_tolerance_not_componentwise_roundoff(self):
        import numpy as np
        sys.path.insert(0,str(SHARED.parent/'block_alternating'))
        from solver import gram_check
        original=np.diag([90.,80.,75.]);reconstructed=original.astype('float32')
        reconstructed[0,1]=1e-6
        self.assertTrue(gram_check(reconstructed,original)['passed'])
        self.assertFalse(gram_check(original*1.001,original)['passed'])

    def test_microbatch_forward_rejects_retained_graph(self):
        context=dict(ids=list(range(128)))
        with self.assertRaises(RuntimeError):Microbatch(context).forward(.01)

    def test_seven_job_dependencies_are_idempotent(self):
        path=Path(__file__).with_name('submit.py');spec=importlib.util.spec_from_file_location('full_submit_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'results/smoacs/test-full'
            with patch('pathlib.Path.cwd',return_value=root),patch('sys.argv',['submit','--output',str(output)]),\
                patch.object(module.subprocess,'check_output',side_effect=[str(i)+'\n' for i in range(100,107)]) as call:
                module.main();module.main()
                self.assertEqual(call.call_count,7)
                self.assertIn('--dependency=afterok:101:103:105',call.call_args_list[-1].args[0])
                self.assertEqual(json.loads(output.with_name(output.name+'-jobs.json').read_text())['alternating'],102)


if __name__=='__main__':unittest.main()
