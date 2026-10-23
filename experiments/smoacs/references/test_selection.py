import unittest
from prepare import clean, is_reference


class SelectionTests(unittest.TestCase):
    def test_subsystem_hull_and_missing_elementals(self):
        systems = [frozenset(['Ca', 'S'])]
        row = dict(material_id='mp-1', chemsys='Ca', nsites=2, deprecated=False, energy_above_hull=0)
        self.assertTrue(is_reference(row, systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, energy_above_hull=.31), systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, chemsys='Ca-O'), systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, deprecated=True), systems, {'mp-1'}))
        self.assertTrue(is_reference(dict(row, energy_above_hull=None), systems, set()))
        self.assertFalse(is_reference(dict(row, energy_above_hull=None, nsites=11), systems, set()))
        self.assertFalse(is_reference(dict(row, chemsys='Ca-S', energy_above_hull=None), systems, set()))

    def test_null_species_metadata_removed(self):
        self.assertEqual(clean({'species':[{'element':'Ca', 'oxidation_state':None}]}),
                         {'species':[{'element':'Ca'}]})


if __name__ == '__main__':
    unittest.main()
