import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import repair_relaxation as repair


class RepairTests(unittest.TestCase):
    def test_preserves_A_and_successes_rechecks_only_supported_cuda_failures(self):
        for steps in (200, 600):
            with self.subTest(steps=steps):
                self.exercise_repair(steps)

    def exercise_repair(self, steps):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            root = workspace/'results/mattergen/full/neutral'
            root.mkdir(parents=True)
            for name in ('A-trace.jsonl', 'checkpoint.pt', 'protocol.json'):
                (root/name).write_text('protected '+name)
            protected = {name: repair.digest(root/name) for name in ('A-trace.jsonl', 'checkpoint.pt', 'protocol.json')}
            rows, states = [], []
            for i, (z, status) in enumerate([(8, 'failed'), (98, 'failed'), (8, 'converged'), (8, 'unconverged')]):
                key = f'mattergen-{i:04d}.vasp'
                source = root/'A'/key
                source.parent.mkdir(exist_ok=True)
                source.write_text('immutable A '+key)
                sha = repair.digest(source)
                folder = root/'cycle-1/B'/Path(key).stem
                folder.mkdir(parents=True)
                record = dict(candidate_id=key, input_sha256=sha, checkpoint_sha256=repair.MODEL_SHA,
                    status=status, maxstep=.05, step_limit=1600)
                if status == 'failed':
                    record['error'] = 'CUDA error: device-side assert triggered'
                else:
                    (folder/'relaxed.vasp').write_text('preserved B '+key)
                    record.update(relaxed_path=str(folder/'relaxed.vasp'), relaxed_sha256=repair.digest(folder/'relaxed.vasp'))
                repair.write(folder/'result.json', record)
                rows.append(dict(candidate_id=key, A=dict(source=str(source), sha256=sha, atomic_numbers=[z],
                    num_atoms=1, formula='O1' if z==8 else 'Cf1', elements=['O'] if z==8 else ['Cf'],
                    gap=2.5, eform=-1., valid=False, neutral=False, separated=True),
                    b_result=record, status='completed' if status=='converged' else 'B_'+status, a_updates=steps))
                states.append(dict(candidate_id=key, status='B_'+status))
            repair.write(root/'results.json', dict(phase='complete', protocol=dict(cycles=1, steps_per_A=steps, ids=[r['candidate_id'] for r in rows]),
                cycles=[dict(cycle=1, candidates=rows)], states=states))
            repair.write(root/'complete.json', dict(completed=True, cycles=1, denominator=4))
            (root/'evaluation-cycle1').mkdir()
            (root/'evaluation-cycle1/keep-evidence').write_text('old evaluator preserved')
            preserved_B = [repair.digest(root/'cycle-1/B'/f'mattergen-{i:04d}'/'result.json') for i in (2, 3)]
            calls = []

            def fake_start(_root, _tag):
                worker = SimpleNamespace(stdin=io.StringIO(), poll=lambda: 0)
                worker.stdout = worker
                return worker, io.StringIO(), dict(ready=True, checkpoint_sha256=repair.MODEL_SHA, model_max_z=94)

            def fake_response(worker, _log):
                task = json.loads(worker.stdin.getvalue())['tasks'][0]
                calls.append(task['candidate_id'])
                path = Path(task['output'])/'relaxed.vasp'
                path.write_text('isolated fresh B')
                return dict(results=[dict(candidate_id=task['candidate_id'], input_sha256=task['expected_sha256'],
                    checkpoint_sha256=repair.MODEL_SHA, status='converged', converged=True,
                    relaxed_path=str(path), relaxed_sha256=repair.digest(path),
                    maxstep=.05, step_limit=1600, max_filter_force=.005)])

            with patch.dict(os.environ, SLURM_JOB_ID='42'), patch('pathlib.Path.cwd', return_value=workspace), \
                 patch('sys.argv', ['repair', '--run', str(root), '--workers', '1']), \
                 patch.object(repair, 'start_worker', side_effect=fake_start) as start, \
                 patch.object(repair, 'read_response', side_effect=fake_response):
                repair.main()
                self.assertEqual(start.call_count, 2)
                spec = importlib.util.spec_from_file_location('posthoc_export', repair.SMOACS/'adam_cycles/export.py')
                exporter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(exporter)
                with patch('sys.argv', ['export', '--run', str(root), '--cycles', '1']):
                    exporter.main()
                repair.main()
                self.assertEqual(start.call_count, 2)
            self.assertEqual(calls, ['mattergen-0000.vasp'])
            for name, sha in protected.items():
                self.assertEqual(repair.digest(root/name), sha)
            self.assertEqual(preserved_B, [repair.digest(root/'cycle-1/B'/f'mattergen-{i:04d}'/'result.json') for i in (2, 3)])
            final = repair.load(root/'results.json')['cycles'][0]['candidates']
            self.assertEqual(final[0]['status'], 'B_converged_posthoc_repair')
            self.assertEqual(final[1]['b_result']['error_type'], 'unsupported_species')
            self.assertFalse(final[1]['b_result']['evaluated'])
            self.assertTrue((root/'evaluation-cycle1-before-cuda-repair-42/keep-evidence').exists())
            self.assertTrue((root/'B-before-cuda-repair-42/mattergen-0000/result.json').exists())
            exported = repair.load(root/'evaluation-cycle1/inventory-249/inventory.json')
            self.assertEqual(exported['denominator'], 4)
            self.assertIsNotNone(exported['posthoc_B_repair'])


if __name__ == '__main__':
    unittest.main()
