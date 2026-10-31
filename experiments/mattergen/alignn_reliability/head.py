"""Small, pickle-free reliability head; inference needs NumPy only."""
from pathlib import Path
import numpy as np
from core import GAP_SHA


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0, -x))


class Head:
    def __init__(self, arrays): self.arrays = arrays

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d: arrays = {k: d[k].copy() for k in d.files}
        if str(arrays['predictor_sha256']) != GAP_SHA: raise ValueError('Predictor identity mismatch.')
        return cls(arrays)

    def logits(self, features, raw_prediction=None):
        d = self.arrays; x = np.asarray(features)
        if str(d['input']) == 'raw_prediction':
            if raw_prediction is None: raise ValueError('Raw bandgap predictions required.')
            x = np.asarray(raw_prediction).reshape(-1, 1)
        if x.ndim != 2 or x.shape[1] != len(d['mean']) or not np.isfinite(x).all():
            raise ValueError('Feature shape/finite failure.')
        z = (x-d['mean'])/d['scale']; kind = str(d['kind'])
        if kind == 'mlp':
            z = z.astype(np.float32)
            h = np.maximum(z@d['w1'].T+d['b1'], 0)
            return (h@d['w2'].T+d['b2']).reshape(-1).astype(float)
        if kind == 'linear': return (z@d['weight']+d['bias']).reshape(-1)
        if kind == 'constant': return np.full(len(z), float(d['bias']))
        raise ValueError('Unknown classifier type.')

    def predict_proba(self, features, raw_prediction=None):
        z = self.logits(features, raw_prediction)
        return sigmoid(float(self.arrays.get('cal_a', 1.))*z+float(self.arrays.get('cal_b', 0.)))

    def save(self, path): np.savez_compressed(path, **self.arrays)
