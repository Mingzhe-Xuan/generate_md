"""Likelihoods and clipped GRPO on explicit augmented stochastic actions."""
import math
import torch
from torch import nn


def sym6(matrix):
    return torch.stack([matrix[..., i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], -1)


def matrix6(vector):
    return torch.stack([vector[..., k] for k in (0, 3, 4, 3, 1, 5, 4, 5, 2)], -1).reshape(*vector.shape[:-1], 3, 3)


def project_cell(raw):
    # This parameter-independent environment map acts on a recorded six-dimensional action.
    values, vectors = torch.linalg.eigh(matrix6(raw))
    return vectors @ torch.diag_embed(values.abs().clamp_min(.01)) @ vectors.transpose(-1, -2)


def normal_logp(action, mean, std):
    a, m, s = action.double(), mean.double(), std.double()
    if bool((s <= 0).any()) or not torch.isfinite(s).all():
        raise ValueError('Invalid Gaussian action standard deviation.')
    return (-.5*((a-m)/s)**2-s.log()-.5*math.log(2*math.pi)).sum(-1)


def normal_kl(mean, std, reference_mean, reference_std):
    m, s, rm, rs = (x.double() for x in (mean, std, reference_mean, reference_std))
    return ((rs/s).log()+.5*((s.square()+(m-rm).square())/rs.square()-1)).sum(-1)


def categorical_logp(logits, action):
    return torch.log_softmax(logits.double(), -1).gather(-1, action.long().unsqueeze(-1)).squeeze(-1)


def categorical_kl(logits, reference_logits):
    lp = torch.log_softmax(logits.double(), -1)
    rq = torch.log_softmax(reference_logits.double(), -1)
    return (lp.exp()*(lp-rq)).sum(-1)


def per_crystal(atom_values, batch_idx, n):
    return atom_values.new_zeros(n).index_add(0, batch_idx, atom_values)


def action_logp(distribution, action, batch_idx, n):
    result = per_crystal(normal_logp(action['pos'], *distribution['pos']), batch_idx, n)
    result = result + normal_logp(action['cell'], *distribution['cell'])
    if 'atomic_numbers' in distribution:
        result = result + per_crystal(categorical_logp(distribution['atomic_numbers'], action['atomic_numbers']), batch_idx, n)
    return result


def action_kl(distribution, reference, batch_idx, n):
    result = per_crystal(normal_kl(*distribution['pos'], *reference['pos']), batch_idx, n)
    result = result + normal_kl(*distribution['cell'], *reference['cell'])
    if 'atomic_numbers' in distribution:
        result = result + per_crystal(categorical_kl(distribution['atomic_numbers'], reference['atomic_numbers']), batch_idx, n)
    return result


def advantages(rewards, group_size):
    if rewards.ndim != 1 or group_size < 2 or len(rewards) % group_size:
        raise ValueError('Expected complete groups of at least two generated candidates.')
    if not torch.isfinite(rewards).all():
        raise ValueError('Nonfinite reward must be represented as an explicit finite failure penalty.')
    groups = rewards.reshape(-1, group_size)
    scale = groups.std(-1, unbiased=False, keepdim=True)
    a = (groups-groups.mean(-1, keepdim=True))/scale.clamp_min(1e-6)
    return torch.where(scale > 1e-6, a, torch.zeros_like(a)).reshape(-1)


def grpo_loss(logp, old_logp, advantage, reference_kl, clip=.2, beta=.01):
    log_ratio = logp - old_logp.detach()
    if not torch.isfinite(log_ratio).all() or bool((log_ratio.abs() > 60).any()):
        raise ValueError('Invalid/extreme policy ratio; preserve evidence and stop.')
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    loss = (policy+beta*reference_kl).mean()
    return loss, dict(policy=float(policy.mean().detach()), reference_kl=float(reference_kl.mean().detach()),
        ratio_max=float(ratio.max().detach()), ratio_min=float(ratio.min().detach()),
        clip_fraction=float(((ratio-1).abs()>clip).double().mean().detach()))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.base = base
        self.in_features = base.in_features; self.out_features = base.out_features
        for p in base.parameters(): p.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    @property
    def weight(self): return self.base.weight

    @property
    def bias(self): return self.base.bias

    def forward(self, x):
        return self.base(x)+(x@self.a.T)@self.b.T


def add_lora(module, rank=4):
    replaced = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank)); replaced.append(name)
        else:
            replaced.extend(name+'.'+key for key in add_lora(child, rank))
    return replaced


class Kernel:
    """Native score/ancestral coefficients with explicit Gaussian corrector actions.

    The corrector uses expected noise norm per crystal, rather than the native
    realized-noise/batch-dependent step. Preprojection actions are retained;
    wrapping and cell polar projection are fixed environment maps. Final-step
    Gaussian floors preserve nondegenerate policy likelihoods. This sampler is
    an explicitly documented experimental variant, not a native-PC replay.
    """
    def __init__(self, sampler, pos_std_floor=.001, cell_std_floor=.001):
        self.sampler = sampler; self.pos_floor = pos_std_floor; self.cell_floor = cell_std_floor
        if sampler.N != 1000:
            raise ValueError('Preserve trained 1000-step D3PM time semantics.')

    def distributions(self, x, t, dt, stage):
        score = self.sampler._score_fn(x, t)
        n = x.get_batch_size(); out = {}
        for name in ('pos', 'cell'):
            value = x[name]; g = score[name]
            if stage == 'predictor':
                p = self.sampler._predictors[name]
                xc, sc, std = p._get_coeffs(value, t, dt, x.get_batch_idx(name), x)
                mean = xc*value+sc*g
                if name == 'cell': mean = mean+(1-xc)*p.corruption.get_limit_mean(value, x)
            elif stage == 'corrector':
                p = self.sampler._correctors[name]
                if name == 'pos':
                    norms = per_crystal(g.square().sum(-1), x.batch, n)
                    dims = 3*x.num_atoms.to(g.dtype)
                else:
                    norms = g.square().sum((-1, -2)); dims = torch.full_like(norms, 9.)
                alpha = p.get_alpha(t, dt).reshape(n, -1)[:, 0]
                step = (2*p.snr**2*dims/norms.clamp_min(1e-12)*alpha).clamp(1e-8, float(p.max_step_size))
                if name == 'pos': step = step[x.batch, None]
                else: step = step[:, None, None]
                mean = value+step*g; std = torch.sqrt(2*step).expand_as(value)
            else:
                raise ValueError('Unknown stochastic stage.')
            std = torch.broadcast_to(std, mean.shape).clamp_min(self.pos_floor if name == 'pos' else self.cell_floor)
            if name == 'cell':
                if float((mean-mean.transpose(-1, -2)).abs().max().detach()) > 1e-4:
                    raise ValueError('Lattice mean outside symmetric action support.')
                mean = sym6((mean+mean.transpose(-1, -2))/2); std = sym6(std)
            out[name] = (mean, std)
        if stage == 'predictor':
            from mattergen.diffusion.discrete_time import to_discrete_time
            p = self.sampler._predictors['atomic_numbers']; c = p.corruption
            time = to_discrete_time(t, N=p.N, T=c.T)
            probs = torch.softmax(score.atomic_numbers, -1)
            logits, _ = c.d3pm.sample_and_compute_posterior_q(x_0=probs, t=time[x.batch].long(),
                make_one_hot=False, samples=c._to_zero_based(x.atomic_numbers), return_logits=True)
            out['atomic_numbers'] = logits
        if any(not torch.isfinite(v).all() for item in out.values() for v in (item if isinstance(item, tuple) else (item,))):
            raise ValueError('Nonfinite stochastic policy distribution.')
        return out

    def sample(self, x, distribution):
        action = {name: mean+std*torch.randn_like(mean) for name, (mean, std) in
                  ((name, distribution[name]) for name in ('pos', 'cell'))}
        replace = dict(pos=action['pos'].remainder(1.), cell=project_cell(action['cell']))
        if 'atomic_numbers' in distribution:
            action['atomic_numbers'] = torch.distributions.Categorical(logits=distribution['atomic_numbers']).sample()
            replace['atomic_numbers'] = self.sampler._predictors['atomic_numbers'].corruption._to_non_zero_based(action['atomic_numbers'])
        return x.replace(**replace), action
