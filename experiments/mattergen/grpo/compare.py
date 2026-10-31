"""Untouched matched-context comparison of GRPO and official conditional MatterGen."""
import argparse
from collections import Counter
from functools import reduce
from math import gcd
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from models import Model, tensor_state_hash
from prepare import sha
from rollout import generate, structures, Evaluator
from train import property_metrics
from smoke import write
from checkpoint_io import load_checkpoint

ARMS = ('base', 'base_native', 'grpo_best', 'grpo_best_native', 'grpo_last', 'conditional_matched', 'conditional_native')


def checkpoint_for_arm(arm):
    if arm not in ARMS: raise ValueError('Unknown comparison arm.')
    if arm.startswith('grpo_best'): return 'best.pt'
    if arm == 'grpo_last': return 'latest.pt'
    return None


def composition_summary(poses, valid_ids):
    counts = Counter()
    for row in poses:
        if row['index'] not in valid_ids: continue
        formula = Counter(int(z) for z in row['numbers']); common = reduce(gcd, formula.values())
        key = '|'.join(f'{z}:{formula[z]//common}' for z in sorted(formula))
        counts[key] += 1
    n = sum(counts.values())
    probabilities = np.array(list(counts.values()), float)/n if n else np.array([])
    return dict(evaluable=n, distinct_reduced_compositions=len(counts),
        distinct_composition_fraction=len(counts)/n if n else None,
        entropy_nats=float(-np.sum(probabilities*np.log(probabilities))),
        top_compositions=[dict(atomic_number_counts=k, n=v) for k, v in counts.most_common(10)],
        interpretation='Composition diversity only, not structure uniqueness or SUN.')


def counts_for_test(cohort, n, seed):
    pool = [len(r['numbers']) for r in cohort if r['role'] == 'test']
    if not pool: raise ValueError('No heldout context rows.')
    return np.random.default_rng(seed).choice(pool, size=n).tolist()


def audit_arm(events, poses, counts):
    wanted = list(range(len(counts)))
    if sorted(r['index'] for r in events) != wanted or sorted(r['index'] for r in poses) != wanted:
        raise ValueError('Missing/duplicate final candidates; full requested denominator is mandatory.')
    for row in events:
        if row['batch'] != row['index']//8:
            raise ValueError('Sampling cluster labels differ from generation layout.')
    for row in poses:
        if len(row['numbers']) != counts[row['index']]:
            raise ValueError('Final atom counts differ from the matched context plan.')


def audit_relaxation_response(response, selected):
    wanted = [row['candidate_id'] for row in selected]
    actual = [row['candidate_id'] for row in response['results']]
    if len(set(actual)) != len(actual) or sorted(actual) != sorted(wanted):
        raise ValueError('Relaxation response cohort differs.')
    for row in response['results']:
        if (row.get('infrastructure_failure')
                or row.get('error_type') in ('cuda_context_error', 'cuda_out_of_memory')
                or 'Poisoned CUDA context' in row.get('error', '')):
            raise RuntimeError('Physics infrastructure failed; preserve results and repair before comparison: '
                               f"{row['candidate_id']}: {row.get('error_type', row.get('error', 'unknown'))}")


def paired_cluster_interval(left, right, seed=20261103, repeats=2000):
    if len(left) != len(right) or not left: raise ValueError('Paired rollout counts differ.')
    groups = sorted({r['batch'] for r in left})
    if groups != sorted({r['batch'] for r in right}): raise ValueError('Paired cluster IDs differ.')
    pairs = []
    for group in groups:
        a = [r for r in left if r['batch'] == group]; b = [r for r in right if r['batch'] == group]
        if [r['index'] for r in a] != [r['index'] for r in b]: raise ValueError('Paired contexts differ.')
        pairs.append((sum(r.get('pass_gap', False) for r in a)-sum(r.get('pass_gap', False) for r in b), len(a)))
    delta = sum(p[0] for p in pairs)/sum(p[1] for p in pairs)
    if len(groups) < 2 or len({p[0]/p[1] for p in pairs}) == 1:
        return dict(delta=delta, interval=None, clusters=len(groups), note='Insufficient between-cluster variation for informative empirical bootstrap.')
    a = np.asarray(pairs); indices = np.random.default_rng(seed).integers(len(groups), size=(repeats, len(groups)))
    samples = a[indices].sum(1); values = samples[:, 0]/samples[:, 1]
    return dict(delta=delta, interval=np.quantile(values, [.025, .975]).tolist(), clusters=len(groups),
        method='Paired sampling-batch bootstrap; native corrector has batch coupling; exploratory single-training-seed comparison.')


def row_json(row):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}


def native_generate(model, counts, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed % 2**32)
    started = time.perf_counter()
    with torch.no_grad(): _, terminal_mean = model.sampler.sample(model.batch(counts), mask={})
    return dict(terminal=model.state(terminal_mean), seconds=time.perf_counter()-started,
                sampler=f'Official1000-step PC, native adaptive corrector and final mean, CFG{model.metadata["guidance"]}')


def generation_comparison(root, out, count, evaluator):
    training = json.loads((root/'training/complete.json').read_text())
    if not training['passed'] or not training['frozen_base_unchanged']:
        raise ValueError('Completed checked GRPO run required.')
    cohort_audit = json.loads((root/'data/cohort-audit.json').read_text())
    if sha(root/'data/cohort.jsonl') != cohort_audit['cohort_sha256']:
        raise ValueError('Heldout context source changed.')
    cohort = [json.loads(line) for line in (root/'data/cohort.jsonl').read_text().splitlines()]
    seed = training['identity']['final_test_seed']; counts = counts_for_test(cohort, count, seed)
    plan = dict(count=count, batch_size=8, counts=counts, seed=seed, arms=list(ARMS), target=2.5, tolerance=.1,
        best_checkpoint_sha256=sha(root/'training/best.pt'), last_checkpoint_sha256=sha(root/'training/latest.pt'),
        cohort_sha256=cohort_audit['cohort_sha256'],
        policy_code_sha256={name: sha(Path(__file__).with_name(name)) for name in ('policy.py', 'models.py', 'rollout.py')},
        comparison_code_sha256=sha(__file__), checkpoint_reader_sha256=sha(Path(__file__).with_name('checkpoint_io.py')),
        training_complete_sha256=sha(root/'training/complete.json'),
        limitations='Same ALIGNN used for training reward and evaluation; surrogate adherence only. '
                     'Real cohort supplies atom counts, not template geometry. Training/checkpoint selection never uses final test seeds.')
    if plan['policy_code_sha256'] != {name: training['identity']['code_sha256'][name] for name in plan['policy_code_sha256']}:
        raise ValueError('Evaluation policy code differs from trained policy.')
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text()) != plan:
        raise ValueError('Existing comparison plan differs; use a separate experiment.')
    write(out/'plan.json', plan)
    for arm in ARMS:
        folder = out/arm; folder.mkdir(exist_ok=True)
        model = Model(root, kind='conditional' if arm.startswith('conditional') else 'base')
        checkpoint_iteration = None
        if arm.startswith('grpo'):
            model.lora(4)
            path = root/'training'/checkpoint_for_arm(arm)
            saved = load_checkpoint(path)
            if saved['identity'] != training['identity']: raise ValueError('Wrong GRPO checkpoint.')
            model.load_adaptation(saved['adaptation']); checkpoint_iteration = saved['iteration']
        model.module.requires_grad_(False); model.module.eval()
        fingerprint = tensor_state_hash(model.module); records = []; poses = []
        for batch_index, offset in enumerate(range(0, count, 8)):
            batch_folder = folder/f'batch-{batch_index:03d}'; batch_folder.mkdir(exist_ok=True)
            complete = batch_folder/'complete.json'
            if complete.exists():
                batch = json.loads(complete.read_text())
                if batch['model_state_hash'] != fingerprint or batch['terminal_sha256'] != sha(batch_folder/'terminal.pt'):
                    raise ValueError('Stale completed comparison batch.')
                if batch['counts'] != counts[offset:offset+8]: raise ValueError('Changed comparison contexts.')
                records.extend(batch['events']); poses.extend(batch['structures']); continue
            sampled = native_generate(model, counts[offset:offset+8], seed+batch_index) if arm.endswith('_native') else generate(model, counts[offset:offset+8], seed+batch_index)
            terminal = sampled['terminal']; rows = structures(terminal)
            events = [evaluator(row) for row in rows]
            for local, (row, event) in enumerate(zip(rows, events)):
                row['index'] = offset+local; event.update(index=offset+local, batch=batch_index)
            torch.save(terminal, batch_folder/'terminal.pt')
            batch = dict(events=events, structures=[row_json(r) for r in rows], counts=counts[offset:offset+8],
                model_state_hash=fingerprint, terminal_sha256=sha(batch_folder/'terminal.pt'), sampler=sampled['sampler'],
                generation_seconds=sampled['seconds'], seed=seed+batch_index)
            write(complete, batch); records.extend(events); poses.extend(batch['structures'])
            write(out/'progress.json', dict(phase='generation', arm=arm, completed=len(records), requested=count))
        audit_arm(records, poses, counts)
        result = dict(arm=arm, metrics=property_metrics(records), events=records, structures=poses,
            model=model.metadata, checkpoint_iteration=checkpoint_iteration, model_state_hash=fingerprint,
            composition_diversity=composition_summary(poses, {r['index'] for r in records if r['valid']}))
        write(folder/'raw.json', result)
        del model; gc.collect(); torch.cuda.empty_cache()
    return plan


def relaxation_comparison(root, out, evaluator):
    from ase import Atoms
    from ase.io import read, write as ase_write
    smoacs = Path(__file__).resolve().parents[2]/'smoacs'
    sys.path.insert(0, str(smoacs/'alternating')); from rpc import read_response
    task_by_id = {}; originals = {}; worker = None
    for arm in ARMS:
        raw = json.loads((out/arm/'raw.json').read_text()); originals[arm] = raw
        poses = {r['index']: r for r in raw['structures']}
        folder = out/arm/'poses'; folder.mkdir(exist_ok=True)
        for event in raw['events']:
            if not event['valid']: continue
            row = poses[event['index']]; path = folder/f'{event["index"]:04d}.vasp'
            if not path.exists():
                ase_write(path, Atoms(numbers=row['numbers'], cell=row['cell'], scaled_positions=row['frac'], pbc=True), format='vasp', direct=True, sort=False)
            candidate = f'{arm}-{event["index"]:04d}.vasp'
            task_by_id[candidate] = dict(candidate_id=candidate, source=str(path.resolve()),
                output=str((out/arm/f'relaxation/{event["index"]:04d}').resolve()),
                arm=arm, index=event['index'], expected_input_sha256=sha(path))
    log = (out/f'physics-{os.environ["SLURM_JOB_ID"]}.err').open('w')
    results = {}
    try:
        worker = subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'), str(smoacs/'alternating/worker.py'),
            '--checkpoint', str(Path.cwd()/'results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        ready = read_response(worker.stdout, out/'physics.stdout')
        if not ready['ready']: raise RuntimeError('Canonical B worker not ready.')
        tasks = list(task_by_id.values())
        for offset in range(0, len(tasks), 8):
            selected = tasks[offset:offset+8]
            worker.stdin.write(json.dumps(dict(tasks=selected))+'\n'); worker.stdin.flush()
            response = read_response(worker.stdout, out/'physics.stdout')
            audit_relaxation_response(response, selected)
            for row in response['results']:
                if row['input_sha256'] != task_by_id[row['candidate_id']]['expected_input_sha256']:
                    raise ValueError('Relaxation input geometry changed.')
                results[row['candidate_id']] = row
            write(out/'progress.json', dict(phase='common_relaxation', completed=len(results), requested=len(tasks)))
    finally:
        if worker is not None and worker.poll() is None:
            try: worker.stdin.write('{"stop":true}\n'); worker.stdin.flush(); worker.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired): worker.terminate(); worker.wait(timeout=15)
        log.close()
    for arm in ARMS:
        events = []
        for raw in originals[arm]['events']:
            event = dict(index=raw['index'], batch=raw['batch'], valid=False, reward=-1., force_pass=False)
            key = f'{arm}-{raw["index"]:04d}.vasp'; result = results.get(key)
            if result is None:
                event['error'] = raw.get('error', 'No relaxable raw structure.')
            else:
                event['relaxation'] = result
                if result.get('relaxed_path'):
                    atoms = read(result['relaxed_path'], format='vasp')
                    prediction = evaluator(dict(index=raw['index'], numbers=atoms.numbers, cell=atoms.cell.array,
                                                frac=atoms.get_scaled_positions()))
                    event.update(prediction); event['batch'] = raw['batch']
                    event['force_pass'] = result.get('status') == 'converged' and result['max_raw_atomic_force'] <= .01+1e-6
            events.append(event)
        result = dict(arm=arm, metrics=property_metrics(events), events=events,
            raw_force_converged=sum(r['force_pass'] for r in events),
            joint_gap_force_pass=sum(r.get('pass_gap', False) and r['force_pass'] for r in events),
            worker=ready, protocol='Pinned canonical MatterSim5M FixSymmetry/ExpCellFilter BFGS, fmax0.01,maxstep0.05,cap1600; raw force checked separately; not DFT stability.')
        write(out/arm/'relaxed.json', result)


def report(out):
    rows = {}; lines = ['# MatterGen GRPO versus bandgap-conditioned checkpoint', '',
        'Independent final rollout seeds; same frozen ALIGNN used as reward and evaluator. Surrogate adherence, not DFT validation.', '',
        '| Arm | Raw valid | Raw gap pass | Raw valid MAE/eV | Relaxed gap pass | Relaxed valid MAE/eV | Gap AND force pass |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        raw = json.loads((out/arm/'raw.json').read_text()); relaxed = json.loads((out/arm/'relaxed.json').read_text())
        rows[arm] = dict(raw=raw['metrics'], relaxed=relaxed['metrics'],
            force_converged=relaxed['raw_force_converged'], joint_gap_force_pass=relaxed['joint_gap_force_pass'],
            checkpoint_iteration=raw['checkpoint_iteration'], composition_diversity=raw['composition_diversity'])
        a = raw['metrics']; b = relaxed['metrics']; n = a['n']
        fmt = lambda v: 'undefined' if v is None else f'{v:.4f}'
        lines.append(f'| {arm} | {a["valid"]}/{n} | {a["gap_pass"]}/{n} | {fmt(a["gap_MAE_valid_eV"])} | '
                     f'{b["gap_pass"]}/{n} | {fmt(b["gap_MAE_valid_eV"])} | {relaxed["joint_gap_force_pass"]}/{n} |')
    deltas = {}
    for stage, name in (('raw', 'raw.json'), ('relaxed', 'relaxed.json')):
        left = json.loads((out/'grpo_best'/name).read_text())['events']
        for baseline in ('base', 'conditional_matched', 'conditional_native'):
            right = json.loads((out/baseline/name).read_text())['events']
            deltas[stage+'-GRPO-minus-'+baseline] = paired_cluster_interval(left, right)
        left_native = json.loads((out/'grpo_best_native'/name).read_text())['events']
        for baseline in ('base_native', 'conditional_native'):
            right = json.loads((out/baseline/name).read_text())['events']
            deltas[stage+'-GRPO-native-minus-'+baseline] = paired_cluster_interval(left_native, right)
    summary = dict(passed=True, rows=rows, success_rate_differences=deltas,
        plan_sha256=sha(out/'plan.json'),
        limitations='One training seed; native PC batch coupling handled by cluster resampling. Invalid/unsupported/unconverged cases retained. '
                     'ALIGNN is reward and evaluator, no DFT/SUN/independent accuracy claim. Real target structures supply atom-count distribution only.')
    write(out/'comparison.json', summary)
    lines += ['', 'grpo_best is selected only on validation; grpo_last reports the final trained checkpoint. '
              'conditional_matched uses the same explicit GRPO sampler. base_native, grpo_best_native and conditional_native '
              'all use official native PC with their respective checkpoint, isolating weight gains under native inference. '
              'The GRPO checkpoint for native inference is still selected only by the prespecified training-kernel validation, not final native tests.', '',
              'All gap-pass denominators are the full requested generation count. MAE is conditional on evaluable structures; read alongside validity. '
              'Force pass requires canonical relaxation convergence and raw atomic force <=0.01eV/Angstrom. Formation energy/phonons/DFT stability not certified.', '']
    lines += ['Evaluability checks use model support, finite geometry, volume1..200Angstrom^3/atom and periodic separation>=0.6Angstrom; '
              'this is not the paper\'s full validity/SUN definition.', '', '| Arm | Distinct reduced compositions / evaluable | Composition entropy/nats |',
              '|---|---:|---:|']
    for arm in ARMS:
        d = rows[arm]['composition_diversity']
        lines.append(f'| {arm} | {d["distinct_reduced_compositions"]}/{d["evaluable"]} | {d["entropy_nats"]:.4f} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary), flush=True)


def main(root, count):
    out = root/'comparison'; out.mkdir(exist_ok=True)
    evaluator = Evaluator(out/f'property-loader-{os.environ["SLURM_JOB_ID"]}')
    generation_comparison(root, out, count, evaluator)
    relaxation_comparison(root, out, evaluator)
    report(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--count', type=int, default=256); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Generation/evaluation/relaxation must run in Slurm.')
    if a.count < 8 or a.count % 8: p.error('Comparison count must be a positive multiple of8.')
    main(a.run.resolve(), a.count)
