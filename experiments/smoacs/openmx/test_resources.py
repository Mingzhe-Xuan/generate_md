from pathlib import Path
import tempfile
import unittest
from resources import select_species


class ResourceTests(unittest.TestCase):
    def test_semicore_must_match_orbital_valence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'VPS').mkdir()
            (root/'PAO').mkdir()
            (root/'VPS/Fe_PBE19H.vps').write_text('xc.type GGA\nvalence.electron 16\n')
            (root/'PAO/Fe6.0H.pao').write_text('valence.electron 8\nPAO.Lmax 3\nPAO.Mul 15\n')
            with self.assertRaises(FileNotFoundError):
                select_species('Fe', root)
            (root/'PAO/Fe6.0H.pao').write_text('valence.electron 16\nPAO.Lmax 3\nPAO.Mul 15\n')
            selected = select_species('Fe', root)
            self.assertEqual(selected['valence'], 16)
            self.assertEqual(selected['potential'], 'Fe_PBE19H')
            self.assertEqual(len(selected['vps_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
