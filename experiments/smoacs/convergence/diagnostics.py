"""Finite-history diagnostics, not an automatic proof of convergence."""
import numpy as np


def objective(gap, energy, target=2.5, margin=.04):
    gap, energy = np.asarray(gap), np.asarray(energy)
    if gap.shape != energy.shape or not np.isfinite(gap).all() or not np.isfinite(energy).all():
        raise ValueError('Property arrays must match and be finite.')
    return np.maximum(np.abs(gap-target)-margin, 0)+energy


def window_changes(values, start, end, tolerance=.01):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not 0 <= start < end < len(values) or tolerance <= 0:
        raise ValueError('Invalid history window.')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite history.')
    delta = values[end]-values[start]
    span = np.ptp(values[start:end+1], axis=0)
    return dict(start_record_index=start, end_record_index=end, updates_between=end-start,
                mean_delta=float(np.mean(delta)), median_delta=float(np.median(delta)),
                decreased_gt_tolerance=int(np.sum(delta < -tolerance)),
                increased_gt_tolerance=int(np.sum(delta > tolerance)),
                endpoint_change_within_tolerance=int(np.sum(np.abs(delta) <= tolerance)),
                full_window_span_within_tolerance=int(np.sum(span <= tolerance)),
                maximum_span=float(np.max(span)), tolerance=tolerance)


def normalized_direction(gradient, learning_rate):
    gradient = np.asarray(gradient)
    if not np.isfinite(gradient).all() or learning_rate <= 0:
        raise ValueError('Invalid gradient or step scale.')
    scale = float(np.max(np.abs(gradient)))
    return -learning_rate*gradient/scale if scale else np.zeros_like(gradient)
