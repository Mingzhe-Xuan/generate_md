"""Paired perturbations and CF-only candidate selection; no held-out predictor leakage."""
import math
import numpy as np

AMPLITUDES=(.10,.10,.05,.10,.20,.10,.10,.05,.10,.20)


def displacement(n, amplitude, seed):
    if n<2 or amplitude<0:
        raise ValueError('Expected at least two atoms and a nonnegative amplitude.')
    if amplitude==0:
        return np.zeros((n,3))
    rng=np.random.default_rng(seed)
    for _ in range(1000):
        delta=rng.normal(size=(n,3))
        delta-=delta.mean(0)
        rms=np.sqrt(np.mean(np.sum(delta*delta,axis=1)))
        if rms<=1e-12:
            continue
        delta*=amplitude/rms
        if np.linalg.norm(delta,axis=1).max()<=2*amplitude:
            return delta
    raise ValueError('Could not draw a bounded zero-translation displacement.')


def eligible(row):
    b=row['relaxation']
    numbers=[b.get('max_filter_force'),b.get('max_raw_atomic_force'),row.get('crystalformer_gap'),row.get('crystalformer_eform')]
    if b.get('status')!='converged' or not b.get('converged') or any(v is None or not math.isfinite(v) for v in numbers):
        return False
    return numbers[0]<=.01 and numbers[1]<=.01 and numbers[3]<-.5 and row.get('structure_valid') is True


def best(rows, prefix=10):
    selected=[r for r in rows if 1<=r['branch_index']<=prefix and eligible(r)]
    if not selected:
        return None
    return min(selected,key=lambda r:(abs(r['crystalformer_gap']-2.5),
        r['relaxation']['final_energy_eV']/r['relaxation']['num_atoms'],r['branch_index']))
