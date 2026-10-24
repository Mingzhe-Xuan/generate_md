"""Memory-bounded CUDA gradient accumulation with one global Adam update."""
from types import SimpleNamespace
import torch


def accumulate(targets,groups,active,denominator,evaluate):
    gradients=[torch.zeros_like(p) for p in targets];losses=[]
    for start,stop in groups:
        loss=evaluate(start,stop)
        if not torch.isfinite(loss[active[start:stop]]).all():raise ValueError('Nonfinite active microbatch objective.')
        losses.append(loss.detach())
        if not active[start:stop].any():continue
        current=torch.autograd.grad(loss[active[start:stop]].sum()/denominator,targets)
        for total,value in zip(gradients,current):total.add_(value)
    return torch.cat(losses),gradients


class Microbatch:
    def __init__(self,context,size=16):
        self.context=context;self.size=size;n=len(context['ids'])
        self.groups=[(i,min(n,i+size)) for i in range(0,n,size)]

    def evaluate(self,start,stop,temperature):
        c=self.context;n=len(c['ids']);total=int(c['size'].sum())
        atoms=slice(c['slices'][start].start,c['slices'][stop-1].stop);crystals=slice(start,stop)
        data={}
        for key,value in c['data'].items():
            if torch.is_tensor(value) and value.ndim:
                data[key]=value[crystals] if value.shape[0]==n else value[atoms] if value.shape[0]==total else value
            else:data[key]=value
        data['batch']=c['batch'][atoms]-start
        selections=(crystals,crystals,atoms,atoms,crystals)
        targets=[p[s] for p,s in zip(c['targets'],selections)]
        scalers=dict(abc_scaler=SimpleNamespace(rescale=lambda x:x*c['base'][crystals]),
                     angle_scaler=SimpleNamespace(rescale=lambda x:x*60+60))
        prediction_targets=targets[:4]+[None] if c.get('unconstrained_elements') else targets
        gap,ef,out=c['author'].forward_propagation_Crystalformer(optimization_targets=prediction_targets,fixed_inputs=data,
            scalers=scalers,temperature=temperature,bandgap_model=c['gap_model'],e_form_model=c['ef_model'],
            adding_noise_scale=None,limit_coords_displacement=None,
            abc_range=(c['lower'][crystals],c['upper'][crystals]),angle_range=(c['angle_lower'][crystals],c['angle_upper'][crystals]))
        gap,ef=gap.reshape(-1),ef.reshape(-1)
        loss=torch.clamp(torch.abs(gap-2.5)-.04,min=0)+ef
        if c.get('unconstrained_elements'):
            loss=loss+targets[4].sum()*0
        return gap,ef,loss,out

    def gradients(self,temperature,active):
        return accumulate(self.context['targets'],self.groups,active,len(active),lambda a,b:self.evaluate(a,b,temperature)[2])

    def forward(self,temperature):
        if torch.is_grad_enabled():raise RuntimeError('Use streamed gradients(), not a retained full microbatch graph.')
        records=[self.evaluate(a,b,temperature) for a,b in self.groups]
        gap,ef,loss=[torch.cat([r[i] for r in records]) for i in range(3)]
        out={key:torch.cat([r[3][key] for r in records]) for key in ('lattice_vectors','normed_batch_dir_coords','normalized_dist','pos')}
        return gap,ef,loss,out
