"""Read task checkpoints containing NumPy64 optimizer scalars with weights-only mode."""
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys
import numpy as np
import torch


@contextmanager
def optimizer_scalar_context():
    # The cosine schedule stores numpy.float64 LR values; permit only their reconstruction types.
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]):
        yield


def load_checkpoint(path):
    with optimizer_scalar_context():
        return torch.load(path, map_location='cpu', weights_only=True)


if __name__ == '__main__':
    # Keep train.py byte-identical to its validated runtime identity when resuming.
    trainer = Path(__file__).with_name('train.py')
    sys.argv[0] = str(trainer)
    with optimizer_scalar_context():
        runpy.run_path(str(trainer), run_name='__main__')
