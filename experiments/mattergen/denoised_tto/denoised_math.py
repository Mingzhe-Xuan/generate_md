"""Direction controls, common physical proposal guards and complete query coverage."""
import numpy as np
from denoised_policy import PARENTS, TIMES, SCALES, ARMS, POLICY


def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v/norm if norm > 1e-12 else np.zeros_like(v)


def controlled_directions(full, stop, direct, project, rng):
    full, stop, direct = [unit(project(v)) for v in (full, stop, direct)]
    random = project(rng.normal(size=len(full)))
    for block in (slice(None, -6), slice(-6, None)):
        random[block] = unit(random[block])*np.linalg.norm(full[block])
    return dict(full=-full, reverse=full, random=random, stop=-stop, direct=-direct)


def feasible_pose(numbers, cell, frac, reference_cell, reference_frac):
    from ase import Atoms
    from ase.geometry import find_mic
    from ase.neighborlist import neighbor_list
    cell, frac = np.asarray(cell), np.asarray(frac)
    if not np.isfinite(cell).all() or not np.isfinite(frac).all() or np.linalg.det(cell) <= 1e-6:
        return False
    ratio = np.linalg.det(cell)/np.linalg.det(reference_cell)
    stretches=np.linalg.svd(np.linalg.solve(reference_cell,cell),compute_uv=False)
    lower,upper=POLICY['original_volume_bounds']
    if not lower <= ratio <= upper or np.abs(stretches-1).max() > POLICY['proposal_max_strain']:
        return False
    delta=frac-reference_frac
    translation=np.angle(np.exp(2j*np.pi*delta).mean(0))/(2*np.pi)
    vectors,_=find_mic((delta-translation)@cell,cell,pbc=True)
    vectors-=vectors.mean(0)
    if np.linalg.norm(vectors,axis=1).max() > POLICY['proposal_max_atomic_A']:
        return False
    return not len(neighbor_list('d', Atoms(numbers=numbers, cell=cell, scaled_positions=frac, pbc=True), POLICY['proposal_min_distance_A']-1e-9))


def validate_coverage(rows, parents=PARENTS):
    expected = {(p,t,'seed',None,None) for p in parents for t in TIMES}
    expected |= {(p,t,'ray',a,s) for p in parents for t in TIMES for a in ARMS for s in SCALES}
    actual = [(r['parent_id'],r['time'],r['kind'],r.get('arm'),r.get('scale')) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or len({q['id'] for q in rows}) != len(rows):
        raise ValueError('Denoised query inventory is missing, duplicated or changed.')
