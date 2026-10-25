"""Two-task symmetric PCGrad in the existing per-structure geometry variables."""
import hashlib
import json
from pathlib import Path

import torch


def project_pair(physical,weighted_property):
    """Project each task against the other ORIGINAL task; keep sum reduction."""
    if physical.shape!=weighted_property.shape or physical.ndim!=1:
        raise ValueError('Expected equal flat geometry task vectors.')
    p,g=physical.double(),weighted_property.double()
    if not torch.isfinite(p).all() or not torch.isfinite(g).all():
        raise ValueError('PCGrad input must be finite.')
    dot=(p*g).sum()
    conflict=bool(dot<0)
    if conflict:
        pp=(p*p).sum();gg=(g*g).sum()
        if not (pp>0 and gg>0):
            raise ValueError('Conflicting task has zero norm.')
        return p-dot/gg*g,g-dot/pp*p,True
    return p.clone(),g.clone(),False


class PCGrad:
    def __init__(self,context,weight):
        if weight not in (.1,.2,.3):
            raise ValueError('Use one of the matched force-blend weights.')
        self.context=context
        self.weight=weight
        self.calls=0
        self.conflicts=0
        self.output=context['output']
        (self.output/'pcgrad-snapshots').mkdir()

    def protocol(self):
        return dict(name='PCGrad',tasks=['MatterSim total energy','lambda*CF gap+formation-energy loss'],
            projection='Symmetric; each task projected against the other original task only if their dot product is negative.',
            grouping='Each structure separately; concatenate scaled lengths, scaled angles, fractional coordinates.',
            metric='Euclidean in the original scaled geometry parameterization; no force or RMS normalization.',
            reduction='Sum; exact original linear combination for nonconflicting task pairs.',
            arithmetic='F64 projection dot products and coefficients, cast final geometry gradients to original F32.',
            optimizer='Projected task sum before unchanged Adam; no energy-neutrality guarantee for actual Adam displacement.',
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weight=self.weight)

    def __call__(self,physical,property_gradient,active):
        c=self.context
        if len(physical)!=5 or len(property_gradient)!=5:
            raise ValueError('Expected the original five CF variable blocks.')
        if any(torch.count_nonzero(g) for g in (*physical[3:],*property_gradient[3:])):
            raise ValueError('PCGrad experiment fixes elements and oxidation placeholders.')
        mixed=[p+self.weight*g for p,g in zip(physical,property_gradient)]
        rows=[];snapshots=[]
        self.calls+=1
        for i,(key,sl) in enumerate(zip(c['ids'],c['slices'])):
            if not bool(active[i]):
                continue
            pieces=[physical[0][i],physical[1][i],physical[2][sl]]
            cfpieces=[property_gradient[0][i],property_gradient[1][i],property_gradient[2][sl]]
            p=torch.cat([v.reshape(-1) for v in pieces]).double()
            g=torch.cat([v.reshape(-1) for v in cfpieces]).double()*self.weight
            p_projected,g_projected,conflict=project_pair(p,g)
            total=p_projected+g_projected
            if conflict:
                offset=0
                for block,selection in enumerate((i,i,sl)):
                    size=pieces[block].numel()
                    mixed[block][selection]=total[offset:offset+size].reshape(pieces[block].shape).to(mixed[block].dtype)
                    offset+=size
            norm_p,norm_g=float(p.norm()),float(g.norm())
            norm_pp,norm_gp=float(p_projected.norm()),float(g_projected.norm())
            dot=float((p*g).sum());post_dot=float((p_projected*g_projected).sum())
            rows.append(dict(candidate_id=key,conflict=conflict,original_dot=dot,
                original_cosine=dot/(norm_p*norm_g) if norm_p*norm_g>1e-24 else None,
                projected_dot=post_dot,projected_cosine=post_dot/(norm_pp*norm_gp) if norm_pp*norm_gp>1e-24 else None,
                physical_norm=norm_p,weighted_property_norm=norm_g,projected_physical_norm=norm_pp,
                projected_property_norm=norm_gp,linear_sum_norm=float((p+g).norm()),pcgrad_sum_norm=float(total.norm()),
                physical_dot_sum=float((p*total).sum()),property_dot_sum=float((g*total).sum())))
            snapshots.append(dict(candidate_id=key,physical=p.cpu(),weighted_property=g.cpu(),
                projected_physical=p_projected.cpu(),projected_property=g_projected.cpu(),sum=total.cpu()))
            self.conflicts+=int(conflict)
        with (self.output/'pcgrad-trace.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(A_step=self.calls,weight=self.weight,cases=rows),allow_nan=False)+'\n')
        if self.calls==1 or self.calls%25==0:
            torch.save(dict(A_step=self.calls,weight=self.weight,cases=snapshots),self.output/'pcgrad-snapshots'/f'step-{self.calls:03d}.pt')
        return tuple(mixed)

    def finish(self):
        (self.output/'pcgrad-audit.json').write_text(json.dumps(dict(protocol=self.protocol(),gradient_calls=self.calls,
            conflicting_structure_steps=self.conflicts),indent=2)+'\n')
