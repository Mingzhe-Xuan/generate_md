"""Conservative bias fields and bounds in the already-proven q coordinates."""
import numpy as np
from anneal_policy import POLICY


def random_slope(gradient,rng,project):
    gradient=project(gradient)
    value=project(rng.normal(size=len(gradient)))
    for block in (slice(None,-6),slice(-6,None)):
        value[block]*=np.linalg.norm(gradient[block])/max(np.linalg.norm(value[block]),1e-30)
    return value


def bias(q,arm,cf,anchor_q,anchor_loss,slope):
    if arm=='zero':
        return 0.,np.zeros_like(q)
    if arm=='random':
        return float(anchor_loss+np.dot(slope,q-anchor_q)),np.asarray(slope).copy()
    if arm not in ('full','reverse','fixed') or cf is None:
        raise ValueError('True CF bias requires a CF value and derivative.')
    sign=-1. if arm=='reverse' else 1.
    return sign*cf['loss'],sign*np.asarray(cf['gradient'])


def combine(q,arm,weight,energy_total,energy_gradient_total,natoms,reference_energy,cf,anchor_q,anchor_loss,slope,project):
    if natoms<=0 or weight<0:
        raise ValueError('Positive atom count and nonnegative weight required.')
    value,g=bias(q,arm,cf,anchor_q,anchor_loss,slope) if weight else (0.,np.zeros_like(q))
    energy=(energy_total-reference_energy)/natoms+weight*value
    gradient=project(np.asarray(energy_gradient_total)/natoms+weight*g)
    if not np.isfinite(energy) or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite biased objective.')
    return dict(energy=float(energy),gradient=gradient,bias_value=float(value),bias_gradient=g,
        physical_energy_per_atom=float(energy_total/natoms),weight=float(weight))


def bounded_project(geometry,q):
    q=geometry.project(q)
    maximum=np.linalg.norm(np.sqrt(geometry.n)*q[:-6].reshape(geometry.n,3),axis=1).max()
    q[:-6]*=min(1.,POLICY['atomic_offset_max_A']/max(maximum,1e-30))
    strain=np.einsum('a,aij->ij',q[-6:]/geometry.scale,geometry.basis)
    q[-6:]*=min(1.,POLICY['strain_norm_max']/max(np.linalg.norm(strain,2),1e-30))
    return geometry.project(q)
