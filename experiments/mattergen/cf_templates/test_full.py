from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from full_report import rate, validate_partition, main as make_full_report
from monitor import parse_gpu, task_labels
from template_policy import select_cohort


class FullTests(unittest.TestCase):
    def test_all128_have_one_explicit_strategy(self):
        rows = [dict(candidate_id=str(i), eligible=i<65, num_atoms=4) for i in range(128)]
        groups = {m: [r['candidate_id'] for r in select_cohort(rows, m)] for m in ('neutral', 'unmasked')}
        self.assertEqual([len(groups[m]) for m in ('neutral', 'unmasked')], [65, 63])
        validate_partition([str(i) for i in range(128)], groups)
        groups['unmasked'][0] = '0'
        with self.assertRaises(ValueError):
            validate_partition([str(i) for i in range(128)], groups)

    def test_unknowns_remain_in_full_rate(self):
        self.assertEqual(rate([True, False, None]), dict(passed=1, unknown=1, denominator=3,
            confirmed_rate=1/3, possible_rate_bounds=[1/3, 2/3]))
        with self.assertRaises(ValueError):
            rate([1, 0])

    def test_gpu_units_and_only_task_labels_are_retained(self):
        self.assertEqual(parse_gpu('6800, 32607, 80\n')['total_mib'], 32607)
        labels = task_labels(b'SLURM_JOB_ID=123\0SECRET=not-recorded\0MATTERGEN_CF_COHORT=neutral\0')
        self.assertEqual(labels, dict(SLURM_JOB_ID='123', MATTERGEN_CF_COHORT='neutral'))
        with self.assertRaises(ValueError):
            parse_gpu('40000, 32607, 50')

    def test_unmasked_forward_really_disables_oxidation(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'smoacs/adam_cycles'))
        from microbatch import Microbatch
        targets = [torch.nn.Parameter(torch.ones(shape, dtype=torch.float64))
                   for shape in [(2, 3), (2, 3), (4, 3), (4, 98), (2, 10)]]
        seen = []

        def predictor(**kwargs):
            xs = kwargs['optimization_targets']
            seen.append(xs[4])
            signal = sum(t.sum() for t in xs[:4])
            count = len(kwargs['fixed_inputs']['size'])
            return signal.expand(count), -signal.expand(count), {}

        size = torch.tensor([2, 2])
        batch = torch.tensor([0, 0, 1, 1])
        context = dict(ids=['a', 'b'], targets=targets, size=size, batch=batch,
            slices=[slice(0, 2), slice(2, 4)], data=dict(size=size, batch=batch),
            base=torch.ones(2, 3), lower=torch.ones(2, 3), upper=torch.ones(2, 3)*10,
            angle_lower=torch.ones(2, 3)*60, angle_upper=torch.ones(2, 3)*120,
            gap_model=None, ef_model=None, author=SimpleNamespace(forward_propagation_Crystalformer=predictor),
            unconstrained_elements=True)
        # Use a nonzero objective slope rather than a cancelling gap/Ef fixture.
        def predictor_nonzero(**kwargs):
            gap, energy, out = predictor(**kwargs)
            return gap, energy*.5, out
        context['author'].forward_propagation_Crystalformer = predictor_nonzero
        loss, grads = Microbatch(context, size=1).gradients(.01, torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(seen, [None, None])
        self.assertTrue(torch.equal(grads[4], torch.zeros_like(grads[4])))
        self.assertTrue(all(torch.isfinite(g).all() and torch.count_nonzero(g)>0 for g in grads[:4]))

    def test_merged_report_keeps_failed_case_and_strategy_denominators(self):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, source = base/'full', base/'source'
            ids = [f'mattergen-{i:04d}.vasp' for i in range(128)]
            original = [dict(candidate_id=key, elements=['H'], original_valid=str(i<65)) for i, key in enumerate(ids)]
            write(source/'inventory-249/inventory.json', dict(denominator=128, candidates=original))
            source_sha = hashlib.sha256((source/'inventory-249/inventory.json').read_bytes()).hexdigest()
            baseline = []

            def prediction(key, stage, valid, missing=False):
                flags = dict(gap=None if missing else True, eform=None if missing else True, joint=None if missing else valid)
                return dict(candidate_id=key, stage=stage, structure_valid=None if missing else valid,
                    crystalformer_gap=None if missing else 2.5, crystalformer_eform=None if missing else -1,
                    alignn_gap=None if missing else 2.5, alignn_eform=None if missing else -1,
                    crystalformer_flags=flags, alignn_flags=flags)

            for i, key in enumerate(ids):
                baseline += [prediction(key, stage, i<65) for stage in ('before', 'after')]
                write(source/'relaxation'/Path(key).stem/'result.json', dict(status='converged'))
            write(source/'cross-prediction/predictions.json', baseline)
            for strategy, chosen in [('neutral', ids[:65]), ('unmasked', ids[65:])]:
                arm = root/strategy
                write(arm/'comparison.json', dict(ids=chosen, strategy=strategy, gpu_audit=dict(passed=True)))
                write(arm/'initialization.json', dict(source_inventory_sha256=source_sha,
                    selected=[dict(candidate_id=i, eligible=strategy=='neutral') for i in chosen]))
                write(arm/'torch-memory.json', dict(max_reserved_bytes=1024))
                candidates, predictions = [], []
                for key in chosen:
                    a_path = arm/'A'/key
                    a_path.parent.mkdir(parents=True, exist_ok=True)
                    a_path.write_text('immutable A '+key)
                    a_sha = hashlib.sha256(a_path.read_bytes()).hexdigest()
                    candidates.append(dict(candidate_id=key, source=str(a_path), sha256=a_sha, elements=['H']))
                    folder = arm/'evaluation-cycle1/relaxation'/Path(key).stem
                    failed = key == ids[64]
                    record = dict(input_sha256=a_sha, status='failed' if failed else 'converged')
                    if not failed:
                        b_path = folder/'relaxed.vasp'
                        b_path.parent.mkdir(parents=True, exist_ok=True)
                        b_path.write_text('immutable B '+key)
                        record['relaxed_sha256'] = hashlib.sha256(b_path.read_bytes()).hexdigest()
                    write(folder/'result.json', record)
                    predictions += [prediction(key, 'before', True), prediction(key, 'after', True, failed)]
                write(arm/'evaluation-cycle1/inventory-249/inventory.json', dict(candidates=candidates))
                write(arm/'evaluation-cycle1/cross-prediction/predictions.json', predictions)
                write(arm/'evaluation-cycle1/cross-prediction/metadata.json', dict(denominator=len(chosen),
                    crystalformer_control=dict(control_passed=True), controls=[dict(max_errors=[0]), dict(max_errors=[0])]))
            with patch('sys.argv', ['full_report', '--run', str(root), '--source', str(source)]):
                make_full_report()
            report = json.loads((root/'comparison.json').read_text())
            self.assertEqual(report['strategy_counts'], dict(neutral=65, unmasked=63))
            self.assertEqual(report['relaxation_status_counts'], dict(converged=127, failed=1))
            self.assertEqual(report['groups']['all128']['properties']['MatterGen_CF600_B']['crystalformer_joint'],
                             dict(passed=127, unknown=1, denominator=128))
            self.assertEqual(report['groups']['neutral']['properties']['MatterGen_CF600_B']['crystalformer_joint']['denominator'], 65)
            self.assertEqual(report['groups']['unmasked']['properties']['MatterGen_B']['crystalformer_joint']['passed'], 0)
            self.assertFalse(report['SUN_complete'])


if __name__ == '__main__':
    unittest.main()
