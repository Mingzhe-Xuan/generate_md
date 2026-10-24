"""Requeued jobs must initialize without overwriting the first capture."""
from pathlib import Path
import tempfile
import unittest

from initialization import capture_name


class CaptureTests(unittest.TestCase):
    def test_fresh_capture_preserves_author_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(capture_name(folder, '312', False), 'capture-312')

    def test_requeue_preserves_all_previous_captures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('capture-312', 'capture-312-resume-1'):
                (root / name).mkdir()
                (root / name / 'evidence').write_text(name)
            chosen = root / capture_name(root, '312', True)
            chosen.mkdir(exist_ok=False)
            self.assertEqual(chosen.name, 'capture-312-resume-2')
            for name in ('capture-312', 'capture-312-resume-1'):
                self.assertEqual((root / name / 'evidence').read_text(), name)


if __name__ == '__main__':
    unittest.main()
