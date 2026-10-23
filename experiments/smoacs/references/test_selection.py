import unittest
from prepare import clean, is_reference, entry_material_ids


class SelectionTests(unittest.TestCase):
    def test_subsystem_hull_and_missing_elementals(self):
        systems = [frozenset(['Ca', 'S'])]
        row = dict(material_id='mp-1', chemsys='Ca', nsites=2, deprecated=False, energy_above_hull=0)
        self.assertTrue(is_reference(row, systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, energy_above_hull=.31), systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, chemsys='Ca-O'), systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, deprecated=True), systems, {'mp-1'}))
        self.assertFalse(is_reference(dict(row, energy_above_hull=None), systems, set()))
        self.assertFalse(is_reference(dict(row, energy_above_hull=None, nsites=11), systems, set()))
        self.assertFalse(is_reference(dict(row, chemsys='Ca-S', energy_above_hull=None), systems, set()))

    def test_null_species_metadata_removed(self):
        self.assertEqual(clean({'species':[{'element':'Ca', 'oxidation_state':None}]}),
                         {'species':[{'element':'Ca'}]})

    def test_nested_entry_id_not_parent(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import io
        kind = pa.map_(pa.string(), pa.struct([('data', pa.struct([('material_id', pa.string())])),
                                              ('energy', pa.float64())]))
        table = pa.table({'material_id':['mp-1'], 'entries':pa.array([
            [('GGA', {'data':{'material_id':'mp-2'}, 'energy':1.0})]], type=kind)})
        data = io.BytesIO()
        pq.write_table(table, data)
        projected = pq.ParquetFile(io.BytesIO(data.getvalue())).read(columns=[
            'material_id', 'entries.key_value.key', 'entries.key_value.value.data.material_id'])
        ids, differences = entry_material_ids(projected)
        self.assertEqual(ids, {'mp-2'})
        self.assertEqual(differences, [{'parent':'mp-1', 'entry_material_id':'mp-2'}])


if __name__ == '__main__':
    unittest.main()
