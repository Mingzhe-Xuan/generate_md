from pathlib import Path
import tempfile
import unittest
from pymatgen.core import Structure, Lattice
from convert import convert_record, digest


class ConvertTests(unittest.TestCase):
    def test_identity_and_composition_not_silently_changed(self):
        structure = Structure(Lattice.cubic(3), ['Si'], [[0,0,0]])
        row = dict(material_id='mp-149', structure=structure.as_dict(), nsites=1,
                   chemsys='Si', energy_above_hull=0)
        with tempfile.TemporaryDirectory() as folder:
            result = convert_record(row, folder)
            self.assertEqual(result['sha256'], digest(Path(folder)/'mp-149.vasp'))
            self.assertEqual(result['elements'], ['Si'])
            with self.assertRaises(ValueError):
                convert_record(dict(row, chemsys='C'), folder)
            with self.assertRaises(ValueError):
                convert_record(dict(row, nsites=2), folder)


if __name__ == '__main__':
    unittest.main()
