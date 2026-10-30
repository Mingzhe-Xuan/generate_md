"""Regression for stale manifests when one renderer follows another."""
from pathlib import Path
import tempfile
import unittest
from run import owned_artifacts


class ArtifactTests(unittest.TestCase):
    def test_later_renderer_does_not_invalidate_primary_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for prefix in ('bandgap-trajectory', 'stage-probes'):
                for ext in ('png', 'svg', 'pdf'): (root/f'{prefix}.{ext}').write_bytes(b'original')
            (root/'states.csv').write_text('a,b\n1,2\n')
            primary = owned_artifacts(root, 'bandgap-trajectory', ('states.csv',))
            phase = owned_artifacts(root, 'stage-probes')
            (root/'stage-probes.svg').write_bytes(b'rerender with new metadata')
            self.assertEqual(primary, owned_artifacts(root, 'bandgap-trajectory', ('states.csv',)))
            self.assertNotEqual(phase, owned_artifacts(root, 'stage-probes'))
            self.assertEqual(len(primary), 4)

    def test_missing_owned_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError): owned_artifacts(Path(tmp), 'bandgap-trajectory')


if __name__ == '__main__': unittest.main()
