"""Full-1000-level CUDA rollout, likelihood replay and real-reward GRPO update."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from models import Model, tensor_state_hash
from policy import action_logp, action_kl, advantages, grpo_loss
from prepare import sha
from rollout import generate, structures, Evaluator


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def main(root):
    out = root/f'smoke-{os.environ["SLURM_JOB_ID"]}'; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    policy = Model(root); policy.lora(4); reference = Model(root)
    frozen = tensor_state_hash(policy.module, frozen_only=True)
    started = time.perf_counter()
    rollout = generate(policy, [4]*4, 20261101, record_indices=(50, 200, 500, 800, 950),
        progress=lambda event: (write(out/'progress.json', event), print(json.dumps(event), flush=True)))
    torch.save(rollout, out/'rollout.pt')
    evaluator = Evaluator(out/'property-loader')
    events = [evaluator(row) for row in structures(rollout['terminal'])]
    write(out/'rewards.json', events)
    a = advantages(torch.tensor([r['reward'] for r in events], device='cuda', dtype=torch.float64), 4)
    if not bool((a.abs() > 0).any()):
        raise ValueError('No within-group real reward variance; preserve rollout and investigate before training.')
    params = [p for p in policy.module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.)
    before = {n: p.detach().clone() for n, p in policy.module.named_parameters() if p.requires_grad}
    optimizer.zero_grad()
    replay = []; losses = []
    for record in rollout['records']:
        x = policy.restore(record['state']); t = torch.full((4,), record['t'], device='cuda'); dt = torch.tensor(record['dt'], device='cuda')
        action = {k: v.to('cuda') for k, v in record['action'].items()}
        distribution = policy.kernel.distributions(x, t, dt, record['stage'])
        lp = action_logp(distribution, action, x.batch, 4)
        error = float((lp.detach()-record['old_logp'].to('cuda')).abs().max())
        replay.append(dict(index=record['index'], stage=record['stage'], max_old_logp_error=error))
        if error > .01:
            write(out/'replay-failure.json', replay)
            raise ValueError('Unchanged-policy replay logprob error exceeds0.01.')
        with torch.no_grad(): ref = reference.kernel.distributions(x, t, dt, record['stage'])
        kl = action_kl(distribution, ref, x.batch, 4)/(3*x.num_atoms.double()+6+(x.num_atoms if record['stage']=='predictor' else 0))
        loss, stats = grpo_loss(lp, record['old_logp'].to('cuda'), a, kl, beta=.01)
        (loss/len(rollout['records'])).backward(); losses.append(stats)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
    if not torch.isfinite(norm) or norm <= 0: raise ValueError('No finite nonzero policy gradient.')
    optimizer.step()
    changed = max(float((p.detach()-before[n]).abs().max()) for n, p in policy.module.named_parameters() if p.requires_grad)
    if changed <= 0 or tensor_state_hash(policy.module, frozen_only=True) != frozen:
        raise ValueError('No adaptation update or frozen pretrained parameters changed.')
    torch.save(dict(adaptation=policy.adaptation_state(), optimizer=optimizer.state_dict(), metadata=policy.metadata), out/'adaptation.pt')
    result = dict(passed=True, full_rollout_steps=1000, generated=4, real_rewards=events,
        max_unchanged_policy_logp_error=max(r['max_old_logp_error'] for r in replay), replay=replay,
        gradient_norm=float(norm), max_adaptation_change=changed, frozen_base_unchanged=True,
        trainable=policy.metadata, reference=reference.metadata, property=evaluator.properties.metadata,
        native_property_controls=evaluator.properties.predictor.native_controls,
        seconds=time.perf_counter()-started, rollout_sha256=sha(out/'rollout.pt'),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), losses=losses,
        interpretation='Implementation smoke with one genuine group update, not final GRPO experiment or benchmark result.')
    write(out/'complete.json', result)
    print(json.dumps({k: result[k] for k in ('passed', 'gradient_norm', 'max_adaptation_change', 'seconds', 'peak_cuda_allocated_bytes')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('GPU smoke must use Slurm.')
    main(a.run.resolve())
