"""Finite-amplitude raw/relaxed property response and translation-free controls."""
import numpy as np
from relations import centered


def rms_direction(vector, n):
    vector = centered(vector)
    norm = np.linalg.norm(vector)
    return None if norm < 1e-10 else (vector*np.sqrt(n)/norm).reshape(n, 3)


def response(raw_plus, raw_minus, relaxed_plus, relaxed_minus, raw_gradients,
             relaxed_gradients, direction, relaxed_difference, amplitude):
    n = len(direction)
    raw_fd = (np.asarray(raw_plus)-raw_minus)/(2*amplitude)
    post_fd = (np.asarray(relaxed_plus)-relaxed_minus)/(2*amplitude)
    j_direction = centered(relaxed_difference)/(2*amplitude)
    raw_ad = np.asarray(raw_gradients).reshape(2, -1) @ np.asarray(direction).ravel()
    pullback = np.asarray(relaxed_gradients).reshape(2, -1) @ j_direction
    return dict(raw_FD=raw_fd.tolist(), raw_AD=raw_ad.tolist(), relaxed_FD=post_fd.tolist(),
                finite_map_pullback=pullback.tolist(),
                response_ratio=[float(a/b) if abs(b) >= 1e-5 else None for a, b in zip(post_fd, raw_fd)],
                absolute_response_retention=[float(abs(a/b)) if abs(b) >= 1e-5 else None for a, b in zip(post_fd, raw_fd)],
                geometry_response_gain=float(np.linalg.norm(j_direction)/np.sqrt(n)))
