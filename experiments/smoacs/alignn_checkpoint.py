"""Scope legacy deserialization to checkpoint bytes from pinned official ZIPs."""
import hashlib
from pathlib import Path


class TrustedCheckpointTorch:
    def __init__(self, torch_module, allowed_digests):
        self.module = torch_module
        self.allowed_digests = frozenset(allowed_digests)

    def __getattr__(self, name):
        return getattr(self.module, name)

    def load(self, filename, *args, **kwargs):
        digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if digest not in self.allowed_digests:
            raise ValueError('Legacy loading requires an explicitly verified checkpoint.')
        kwargs['weights_only'] = False
        return self.module.load(filename, *args, **kwargs)
