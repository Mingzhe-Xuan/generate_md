import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from alignn_checkpoint import TrustedCheckpointTorch


class CheckpointTest(unittest.TestCase):
    def test_only_verified_bytes_use_legacy_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tmp.pt'
            path.write_bytes(b'pinned checkpoint')
            backend = Mock()
            proxy = TrustedCheckpointTorch(backend, [hashlib.sha256(path.read_bytes()).hexdigest()])
            proxy.load(path, map_location='cuda')
            backend.load.assert_called_once_with(path, map_location='cuda', weights_only=False)
            self.assertIs(proxy.device, backend.device)
            path.write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):
                proxy.load(path)
            self.assertEqual(backend.load.call_count, 1)


if __name__ == '__main__':
    unittest.main()
