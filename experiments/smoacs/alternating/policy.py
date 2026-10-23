"""Explicit stopping rules for alternating property and energy optimization."""
import math
import numpy as np


def schedule(index, anneal=200):
    if index < 0 or anneal < 1:
        raise ValueError('Invalid schedule index.')
    if index < anneal:
        return .0001+(.01-.0001)*(1-index/anneal), .5*(1+math.cos(math.pi*index/anneal))
    return .0001, .1


def geometry_change(old_cell, old_frac, cell, frac):
    old_cell,cell,old_frac,frac=map(np.asarray,(old_cell,cell,old_frac,frac))
    delta=frac-old_frac
    delta-=np.rint(delta)
    cart=delta@((old_cell+cell)/2)
    cart-=cart.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(cart**2,axis=1)))), float(np.linalg.norm(cell-old_cell)/max(np.linalg.norm(old_cell),1e-12))


def stop_status(gradient, atom_gradient, a_step, stable_rounds, fixed_temperature, b_converged,
                patience=5, cycle_patience=10):
    if not b_converged:return None
    if not fixed_temperature:return None
    if stable_rounds>=patience and gradient<=.001 and a_step<=.001:
        return 'jointly_converged'
    if stable_rounds>=cycle_patience and atom_gradient<=.001 and gradient>.001:
        return 'nonjoint_physical_cycle'
    return None
