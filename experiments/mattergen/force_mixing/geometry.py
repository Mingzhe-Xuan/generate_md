"""Pull raw ASE forces/stress back to fixed-species CF optimization variables."""
import torch


def geometry(context,targets=None):
    from src.utils.coord import compute_lattice_vectors
    p=context['targets'] if targets is None else targets
    abc=torch.clamp(p[0]*context['base'],min=context['lower'],max=context['upper'])
    angles=torch.clamp(p[1]*60+60,min=context['angle_lower'],max=context['angle_upper'])
    return compute_lattice_vectors(abc,angles),torch.remainder(p[2],1.)


def energy_surrogate(cells,frac,batch,energies,forces,stress):
    """Exact first-order E(s,H); stress includes affine atomic motion already."""
    volume=torch.linalg.det(cells).abs().detach()
    inv_transpose=torch.linalg.inv(cells.detach()).transpose(-1,-2)
    cell_gradient=volume[:,None,None]*(inv_transpose@stress.detach())
    fractional_gradient=-torch.bmm(forces.detach()[:,None,:],cells.detach()[batch].transpose(-1,-2)).squeeze(1)
    atomic=(fractional_gradient*(frac-frac.detach())).sum(1)
    terms=torch.zeros(len(cells),dtype=cells.dtype,device=cells.device).scatter_add(0,batch,atomic)
    return energies.detach()+terms+(cell_gradient*(cells-cells.detach())).sum(dim=(1,2))


def combine(physics,property_gradient,weight):
    if weight not in (0.,.1,.2,.3,1.):
        raise ValueError('Expected a declared mixing coefficient.')
    if len(physics)!=len(property_gradient):
        raise ValueError('Gradient blocks differ.')
    return tuple(p+weight*g for p,g in zip(physics,property_gradient))


def property_cartesian_force(property_fractional_gradient,cells,batch,denominator):
    return -torch.bmm((property_fractional_gradient.detach()*denominator)[:,None,:],
        torch.linalg.inv(cells.detach())[batch].transpose(-1,-2)).squeeze(1)
