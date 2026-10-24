import json
from pathlib import Path
import tempfile
import unittest
from shared_allocation import wait_for_children


class SharedAllocationTests(unittest.TestCase):
    def test_no_children_or_completed_failure_do_not_hold_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(wait_for_children(root), [])
            folder = root/'shared-children'
            folder.mkdir()
            record = dict(coordinator='12', completed=True, exit_code=1)
            (folder/'12.json').write_text(json.dumps(record))
            self.assertEqual(wait_for_children(root), [record])

    def test_unfinished_child_cannot_silently_release_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'shared-children'
            folder.mkdir()
            (folder/'12.json').write_text(json.dumps(dict(coordinator='12', completed=False)))
            with self.assertRaises(TimeoutError):
                wait_for_children(root, timeout=0)
