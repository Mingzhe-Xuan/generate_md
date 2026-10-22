import tempfile
from pathlib import Path
import unittest
from inventory import candidate_files


class IdentityTests(unittest.TestCase):
    def test_denominator_duplicates_and_missing_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'Optimized_based_on_a.vasp').touch()
            (root/'Optimized_based_on_b.vasp').touch()
            rows = [dict(original_fnames='a.vasp'), dict(original_fnames='b.vasp')]
            self.assertEqual(len(candidate_files(rows, root, 2)), 2)
            for invalid, expected in [(rows[:1], 2), (rows*2, 4),
                                      ([dict(original_fnames='../a.vasp')], 1),
                                      ([dict(original_fnames='missing.vasp')], 1)]:
                with self.assertRaises(ValueError):
                    candidate_files(invalid, root, expected)


if __name__ == '__main__':
    unittest.main()
