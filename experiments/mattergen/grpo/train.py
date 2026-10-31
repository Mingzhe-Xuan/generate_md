"""Resumable online GRPO from the verified pretrained MatterGen base."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from models import Model, tensor_state_hash
from policy import advantages, action_logp, action_kl, grpo_loss
from prepare import sha
from rollout import generate, structures, Evaluator
from smoke import write


def subset_record(record, indices):
    counts = record['state']['num_atoms']
    bounds = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    atoms = torch.cat([torch.arange(bounds[i], bounds[i+1]) for i in indices])
    graphs = torch.tensor(indices, dtype=torch.long)
    state = {k: v[graphs if k in ('cell', 'num_atoms') else atoms].clone() for k, v in record['state'].items()}
    action = {k: v[graphs if k == 'cell' else atoms].clone() for k, v in record['action'].items()}
    return dict(state=state, action=action, old_logp=record['old_logp'][graphs].clone(),
                indices=graphs, t=record['t'], dt=record['dt'], stage=record['stage'], index=record['index'])


def microbatches(counts, max_graphs=4, max_atoms=64):
    selected = []; atoms = 0
    for index, count in enumerate(counts):
        if selected and (len(selected) >= max_graphs or atoms+count > max_atoms):
            yield selected; selected = []; atoms = 0
        selected.append(index); atoms += count
    if selected: yield selected


def property_metrics(events):
    valid = [r for r in events if r['valid']]
    n = len(events)
    return dict(n=n, valid=len(valid), valid_fraction=len(valid)/n,
        gap_pass=sum(r.get('pass_gap', False) for r in events),
        gap_pass_fraction=sum(r.get('pass_gap', False) for r in events)/n,
        gap_MAE_valid_eV=float(np.mean([r['error_eV'] for r in valid])) if valid else None,
        mean_reward=float(np.mean([r['reward'] for r in events])))


def validation_key(metrics):
    return (metrics['gap_pass_fraction'], metrics['valid_fraction'],
            -metrics['gap_MAE_valid_eV'] if metrics['gap_MAE_valid_eV'] is not None else -float('inf'))


def save_checkpoint(path, model, optimizer, iteration, rng, identity, best):
    temp = path.with_suffix('.tmp')
    torch.save(dict(adaptation=model.adaptation_state(), optimizer=optimizer.state_dict(), iteration=iteration,
        numpy_rng=rng.bit_generator.state, identity=identity, best=best,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), temp)
    temp.replace(path)


def validation(model, pool, evaluator, seed, n, out):
    if (out/'complete.json').exists():
        result = json.loads((out/'complete.json').read_text())
        if result['model_state_hash'] != tensor_state_hash(model.module):
            raise ValueError('Existing validation belongs to another policy state.')
        return result
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts = rng.choice([len(r['numbers']) for r in pool], size=n).tolist()
    events = []
    for batch_index, offset in enumerate(range(0, n, 8)):
        sampled = generate(model, counts[offset:offset+8], seed+batch_index)
        batch_events = [evaluator(r) for r in structures(sampled['terminal'])]
        for local, event in enumerate(batch_events): event['index'] = offset+local
        events.extend(batch_events)
        torch.save(sampled['terminal'], out/f'batch-{batch_index:03d}.pt')
        write(out/'progress.json', dict(completed=len(events), requested=n))
    result = dict(metrics=property_metrics(events), events=events, counts=counts, seed=seed,
                  model_state_hash=tensor_state_hash(model.module))
    write(out/'complete.json', result)
    return result


def policy_update(model, reference, rollout, a, optimizer, inner_updates=2):
    diagnostics = []
    total = len(rollout['counts']); records = rollout['records']
    for inner in range(inner_updates):
        optimizer.zero_grad(); logs = []; max_replay = 0.
        for record in records:
            for indices in microbatches(rollout['counts']):
                piece = subset_record(record, indices); n = len(indices)
                x = model.restore(piece['state']); t = torch.full((n,), piece['t'], device='cuda'); dt = torch.tensor(piece['dt'], device='cuda')
                action = {k: v.to('cuda') for k, v in piece['action'].items()}
                distribution = model.kernel.distributions(x, t, dt, piece['stage'])
                lp = action_logp(distribution, action, x.batch, n); old = piece['old_logp'].to('cuda')
                if inner == 0:
                    error = float((lp.detach()-old).abs().max()); max_replay = max(max_replay, error)
                    if error > .01:
                        raise ValueError(f'Unchanged-policy microbatch replay error{error}>0.01; stop before update.')
                with torch.no_grad(): ref = reference.kernel.distributions(x, t, dt, piece['stage'])
                dims = 3*x.num_atoms.double()+6+(x.num_atoms if piece['stage']=='predictor' else 0)
                kl = action_kl(distribution, ref, x.batch, n)/dims
                loss, log = grpo_loss(lp, old, a[piece['indices'].to('cuda')], kl, beta=.01)
                (loss*(n/total)/len(records)).backward(); logs.append(log)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.module.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(norm): raise ValueError('Nonfinite GRPO gradient.')
        if norm > 0: optimizer.step()
        diagnostics.append(dict(inner=inner, gradient_norm=float(norm), max_replay_error=max_replay,
            mean_reference_kl=float(np.mean([r['reference_kl'] for r in logs])),
            mean_clip_fraction=float(np.mean([r['clip_fraction'] for r in logs])),
            ratio_min=min(r['ratio_min'] for r in logs), ratio_max=max(r['ratio_max'] for r in logs)))
    return diagnostics


def main(args):
    root = args.run.resolve(); out = root/'training'; out.mkdir(parents=True, exist_ok=True)
    smoke = json.loads(args.smoke.read_text())
    if not smoke['passed'] or not smoke['frozen_base_unchanged'] or smoke['full_rollout_steps'] != 1000 or smoke.get('policy_updates', 0) != 2:
        raise ValueError('Passing complete CUDA/likelihood/real-reward smoke required before training.')
    code = {name: sha(Path(__file__).with_name(name)) for name in ('policy.py', 'models.py', 'rollout.py', 'train.py')}
    if any(smoke['code_sha256'][name] != code[name] for name in code):
        raise ValueError('Policy code differs from the passing smoke.')
    cohort = json.loads((root/'data/cohort-audit.json').read_text())
    if not cohort['passed'] or sha(root/'data/cohort.jsonl') != cohort['cohort_sha256']:
        raise ValueError('Training context cohort changed.')
    rows = [json.loads(line) for line in (root/'data/cohort.jsonl').read_text().splitlines()]
    train = [r for r in rows if r['role']=='train']; val = [r for r in rows if r['role']=='validation']
    identity = dict(seed=args.seed, iterations=args.iterations, groups=args.groups, group_size=args.group_size,
        records=args.records, learning_rate=args.lr, inner_updates=2, validation_samples=args.validation_samples,
        validation_every=args.validation_every, code_sha256=code, cohort_sha256=cohort['cohort_sha256'],
        sampler=smoke['sampler'], base=smoke['trainable']['exact_base_tensor_provenance'],
        reward='Frozen ALIGNN raw gap exp(-0.5*((gap-2.5)/0.5)^2); invalid=-1',
        training_data_use='Target-range real train-cohort atom-count contexts only; no supervised warmup or template editing.',
        final_test_seed=args.seed+10000000,
        limitations='ALIGNN serves as both reward and evaluator; adherence is not DFT accuracy. Native sampler baseline required.')
    plan_path = out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != identity:
        raise ValueError('Resume plan/code/data mismatch.')
    write(plan_path, identity)
    torch.set_num_threads(2)
    model = Model(root); torch.manual_seed(args.seed); model.lora(4); reference = Model(root)
    params = [p for p in model.module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.)
    frozen = tensor_state_hash(model.module, frozen_only=True)
    rng = np.random.default_rng(args.seed)
    latest = out/'latest.pt'; best = None; completed = 0
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        if state['identity'] != identity: raise ValueError('Checkpoint identity differs.')
        model.load_adaptation(state['adaptation']); optimizer.load_state_dict(state['optimizer'])
        completed = state['iteration']; rng.bit_generator.state = state['numpy_rng']; best = state['best']
    else:
        save_checkpoint(latest, model, optimizer, 0, rng, identity, best)
    evaluator = Evaluator(out/f'property-loader-{os.environ["SLURM_JOB_ID"]}')
    start = time.perf_counter()
    if completed == 0:
        result = validation(model, val, evaluator, args.seed+1000000, args.validation_samples, out/'validation/0000')
        best = dict(iteration=0, metrics=result['metrics'])
        save_checkpoint(out/'best.pt', model, optimizer, 0, rng, identity, best)
        save_checkpoint(latest, model, optimizer, 0, rng, identity, best)
    for iteration in range(completed+1, args.iterations+1):
        folder = out/f'rollouts/{iteration:04d}'; folder.mkdir(parents=True, exist_ok=True)
        rollout_path = folder/'rollout.pt'
        if rollout_path.exists():
            sampled = torch.load(rollout_path, map_location='cpu', weights_only=True)
            if sampled['policy_state_hash'] != tensor_state_hash(model.module):
                raise ValueError('Saved rollout does not belong to the restored old policy.')
            rng.bit_generator.state = sampled['post_context_rng']
        else:
            group_counts = rng.choice([len(r['numbers']) for r in train], size=args.groups).tolist()
            counts = [n for n in group_counts for _ in range(args.group_size)]
            indices = sorted(rng.choice(np.arange(1000), size=args.records, replace=False).tolist())
            sampled = generate(model, counts, args.seed+1009*iteration, record_indices=indices,
                progress=lambda e: write(out/'progress.json', dict(iteration=iteration, requested=args.iterations, phase='rollout', **e)))
            sampled.update(policy_state_hash=tensor_state_hash(model.module), post_context_rng=rng.bit_generator.state)
            temp = rollout_path.with_suffix('.tmp'); torch.save(sampled, temp); temp.replace(rollout_path)
        events_path = folder/'rewards.json'
        if events_path.exists(): events = json.loads(events_path.read_text())
        else:
            events = [evaluator(r) for r in structures(sampled['terminal'])]; write(events_path, events)
        a = advantages(torch.tensor([r['reward'] for r in events], device='cuda', dtype=torch.float64), args.group_size)
        lr = args.lr*(.1+.9*.5*(1+np.cos(np.pi*(iteration-1)/max(1, args.iterations-1))))
        for group in optimizer.param_groups: group['lr'] = lr
        if bool((a.abs()>0).any()):
            logs = policy_update(model, reference, sampled, a, optimizer)
        else: logs = []; print('No reward variation; counted rollout, skipped policy update.', flush=True)
        event = dict(iteration=iteration, generated=args.groups*args.group_size, cumulative_generated=iteration*args.groups*args.group_size,
            metrics=property_metrics(events), updates=logs, lr=lr, rollout_seconds=sampled['seconds'], elapsed_seconds=time.perf_counter()-start)
        if iteration % args.validation_every == 0 or iteration == args.iterations:
            result = validation(model, val, evaluator, args.seed+1000000, args.validation_samples, out/f'validation/{iteration:04d}')
            event['validation'] = result['metrics']
            if validation_key(result['metrics']) > validation_key(best['metrics']):
                best = dict(iteration=iteration, metrics=result['metrics'])
                save_checkpoint(out/'best.pt', model, optimizer, iteration, rng, identity, best)
            if tensor_state_hash(model.module, frozen_only=True) != frozen:
                raise ValueError('Frozen base parameters changed during online training.')
        save_checkpoint(latest, model, optimizer, iteration, rng, identity, best)
        write(folder/'complete.json', event); write(out/'progress.json', dict(phase='iteration_complete', **event))
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(event, allow_nan=False)+'\n')
        print(json.dumps(event), flush=True)
    if tensor_state_hash(model.module, frozen_only=True) != frozen: raise ValueError('Frozen base changed at training end.')
    if not any(bool((value != 0).any()) for name, value in model.adaptation_state().items() if name.endswith('.b')):
        raise ValueError('No effective LoRA policy update; online GRPO training is not complete.')
    write(out/'complete.json', dict(passed=True, iterations=args.iterations,
        generated_training_candidates=args.iterations*args.groups*args.group_size, best=best,
        best_checkpoint_sha256=sha(out/'best.pt'), latest_checkpoint_sha256=sha(latest), frozen_base_unchanged=True,
        identity=identity, slurm_job_id=os.environ['SLURM_JOB_ID'], seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        interpretation='Online GRPO complete; untouched generation comparison remains a separate required stage.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--smoke', type=Path, required=True); p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--groups', type=int, default=4); p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--records', type=int, default=8); p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--lr', type=float, default=2e-6); p.add_argument('--validation-samples', type=int, default=64)
    p.add_argument('--validation-every', type=int, default=25); args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Online generation/training/evaluation must run in Slurm.')
    if args.group_size < 2 or args.iterations < 1 or args.groups < 1 or not 1 <= args.records <= 1000:
        p.error('Invalid GRPO rollout budget.')
    main(args)
