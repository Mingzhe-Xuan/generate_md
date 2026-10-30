"""Energy-based descriptive pair classification, independent of GPU imports."""
import numpy as np


def eligible(row,fmax=.01):
    return bool(row.get('status')=='converged' and row.get('converged') is True and
        all(row.get(k) is not None and np.isfinite(row[k]) and row[k]<=fmax+1e-10 for k in ('max_raw_force','max_filter_force')))


def classify(delta,strict,conventional):
    if not np.isfinite(delta):raise ValueError('Nonfinite energy difference.')
    return dict(delta_energy_eV_atom=float(delta),strict_match=bool(strict),conventional_match=bool(conventional),
        consistent_endpoint=bool(strict and abs(delta)<=.001),
        energy_winner='early' if delta<-.001 else 'full' if delta>.001 else 'within_1meV_atom')


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),min=float(a.min()),max=float(a.max()),
        median=float(np.median(a)),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75))) if len(a) else dict(n=0)
