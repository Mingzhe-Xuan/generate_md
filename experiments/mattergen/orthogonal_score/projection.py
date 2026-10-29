"""Remove deterministic Cartesian score displacement along available physical force."""
import numpy as np


def force_supported(numbers,cell,frac,max_z):
    z=np.asarray(numbers);cell=np.asarray(cell);frac=np.asarray(frac)
    if np.any(z==101):return 'masked_atom_types'
    if np.any((z<1)|(z>max_z)):return 'unsupported_atom_types'
    if not np.isfinite(cell).all() or not np.isfinite(frac).all():return 'nonfinite_geometry'
    if np.linalg.det(cell)<=1e-6:return 'nonpositive_or_singular_cell'
    return None


def remove_parallel(score_fractional,cell,force):
    """No cosine threshold, no norm restoration; global per-structure projection.

    All vectors are row vectors. Projection operates on s_cart=s_frac L,
    the positional score-driven displacement, not the density covector.
    Centering force removes roundoff net translation; original score COM stays.
    """
    sf=np.asarray(score_fractional,dtype=float)
    force=np.asarray(force,dtype=float)
    cell=np.asarray(cell,dtype=float)
    if not all(np.isfinite(v).all() for v in (sf,cell,force)) or sf.shape!=force.shape:
        raise ValueError('Invalid projection vectors.')
    f=force-force.mean(0)
    raw=sf@cell;s=raw-raw.mean(0)
    nf,ns=np.linalg.norm(f),np.linalg.norm(s)
    meta=dict(force_norm=float(nf),before_norm=float(ns),triggered=False,
              cosine_before=None,cosine_after=None,coefficient=0.,reason=None)
    if nf<=1e-8 or ns<=1e-10:
        meta['reason']='stationary_force_or_score'
        return sf.copy(),meta
    coefficient=float(np.sum(s*f)/(nf*nf))
    residual=raw-coefficient*f
    after=residual-residual.mean(0);na=np.linalg.norm(after)
    meta.update(triggered=True,coefficient=coefficient,cosine_before=float(np.sum(s*f)/(ns*nf)),
                after_norm=float(na),norm_ratio=float(na/ns),removed_norm=float(abs(coefficient)*nf),
                cosine_after=float(np.sum(after*f)/(na*nf)) if na>1e-10 else None)
    if abs(np.sum(after*f))>1e-9*max(ns*nf,1.):
        raise ValueError('Projection failed orthogonality.')
    if na>ns+1e-9*max(ns,1.):raise ValueError('Projection increased centered norm.')
    return residual@np.linalg.inv(cell),meta
