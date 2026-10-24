"""Conservative proposal bounds for independent fixed-species Best-of-N branches."""
import itertools
import math
import torch


def periodic_separated(cell, frac, minimum=.5):
    shifts=torch.tensor(list(itertools.product((-1,0,1),repeat=3)),dtype=frac.dtype,device=frac.device)
    delta=frac[:,None,None,:]-frac[None,:,None,:]+shifts[None,None,:,:]
    distance=torch.linalg.vector_norm(delta@cell,dim=-1)
    indices=torch.arange(len(frac),device=frac.device)
    distance[indices,indices,13]=float('inf')
    return bool(torch.isfinite(cell).all() and torch.isfinite(frac).all() and distance.min()>=minimum)


def install(context,parent_frac,parent_cells,parent_abc,parent_angles,max_displacement=.4):
    if not math.isfinite(max_displacement) or max_displacement<=0:
        raise ValueError('Expected a finite positive proposal displacement bound.')
    dtype=context['targets'][0].dtype
    context['data']['parent_frac']=torch.as_tensor(parent_frac,dtype=dtype)
    context['data']['parent_cells']=torch.as_tensor(parent_cells,dtype=dtype)
    abc=torch.as_tensor(parent_abc,dtype=dtype)
    angles=torch.as_tensor(parent_angles,dtype=dtype)
    context['lower']=torch.maximum(context['lower'],abc*.98)
    context['upper']=torch.minimum(context['upper'],abc*1.02)
    context['angle_lower']=torch.maximum(context['angle_lower'],angles-2.)
    context['angle_upper']=torch.minimum(context['angle_upper'],angles+2.)

    def project():
        with torch.no_grad():
            ref=context['data']['parent_frac']
            delta=context['targets'][2]-ref
            delta-=torch.round(delta)
            cells=context['data']['parent_cells'][context['batch']]
            cart=torch.bmm(delta.unsqueeze(1),cells).squeeze(1)
            factor=torch.clamp(max_displacement/torch.linalg.vector_norm(cart,dim=1).clamp_min(1e-12),max=1.)
            context['targets'][2].copy_(ref+delta*factor[:,None])

    context['project_after_update']=project
    context['proposal_guard']=lambda i,out: periodic_separated(out['lattice_vectors'][i],out['normed_batch_dir_coords'][context['slices'][i]])
    context['branch_bounds']=dict(max_per_site_displacement_from_parent_A=max_displacement,max_length_fraction=.02,
        max_angle_change_deg=2.,minimum_periodic_distance_A=.5)
