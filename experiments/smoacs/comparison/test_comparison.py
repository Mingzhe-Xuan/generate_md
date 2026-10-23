import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paired_statistics import conjunction, index_rows, property_flags, summarize


class PairedStatisticsTests(unittest.TestCase):
    def test_fixed_denominator_transitions_and_unknown_bounds(self):
        rows = [dict(before={'stable':b}, after={'stable':a}) for b,a in
                [(True,True), (True,False), (False,True), (False,False), (True,None)]]
        metric = summarize(rows)['metrics']['stable']
        self.assertEqual(metric['transitions'], dict(retained=1,lost=1,gained=1,neither=1,unknown=1))
        self.assertEqual(metric['after']['rate_lower_bound'], .4)
        self.assertEqual(metric['after']['rate_upper_bound'], .6)
        self.assertAlmostEqual(metric['delta_lower_bound_percentage_points'], -20)
        self.assertIsNone(conjunction(True, None))
        self.assertFalse(conjunction(False, None))

    def test_identity_and_thresholds(self):
        with self.assertRaises(ValueError):
            index_rows([{'candidate_id':'a'}, {'candidate_id':'a'}], ['a','b'])
        with self.assertRaises(ValueError):
            index_rows([{'candidate_id':'other'}], ['a'])
        self.assertTrue(property_flags(2.3, -.5001, True)['property_success'])
        self.assertTrue(property_flags(2.7, -1., True)['gap_success'])
        self.assertFalse(property_flags(2.5, -.5, True)['eform_success'])
        self.assertFalse(property_flags(2.5, -1., False)['property_success'])
        with self.assertRaises(ValueError):
            property_flags(math.nan, -1., True)
        self.assertIsNone(summarize([])['metrics']['stable']['after']['rate_lower_bound'])

    def test_continuous_changes_use_identical_pairs(self):
        result = summarize([dict(before={'hull_eV_atom':.4},after={'hull_eV_atom':.1}),
                            dict(before={'hull_eV_atom':.2},after={})])
        self.assertEqual(result['continuous']['hull_eV_atom']['paired_count'], 1)
        self.assertAlmostEqual(result['continuous']['hull_eV_atom']['delta_mean'], -.3)

    def test_geometry_uses_strict_distance_boundary(self):
        from evaluate import geometry_valid
        from pymatgen.core import Structure, Lattice
        for distance, expected in ((.49,False), (.5,True), (.51,True)):
            structure = Structure(Lattice.cubic(4), ['Li','Li'], [[0,0,0],[distance,0,0]],coords_are_cartesian=True)
            self.assertEqual(geometry_valid(structure),expected)

    def test_original_failed_proposal_reference_scope_is_not_silently_complete(self):
        from evaluate import excluded_references
        from inventory import sha256
        from pymatgen.core import Structure, Lattice
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'reference-structures.jsonl.gz'
            structure = Structure(Lattice.cubic(3),['Li'],[[0,0,0]])
            with gzip.open(path,'wt') as stream:
                stream.write(json.dumps(dict(material_id='mp-extra',chemsys='Li',structure=structure.as_dict()))+'\n')
            inv = dict(scope_excluded_reference_ids=['mp-extra'],reference_records_sha256=sha256(path))
            self.assertEqual(excluded_references(inv,root),[dict(candidate_id='mp-extra.vasp',elements=['Li'],formula='Li')])
            inv['scope_excluded_reference_ids'].append('mp-missing')
            with self.assertRaises(ValueError):
                excluded_references(inv,root)

    def test_complete_small_pipeline_and_paired_uniqueness(self):
        import evaluate
        from assess import hull_query
        from inventory import sha256
        from pymatgen.core import Structure, Lattice
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        checkpoint = json.loads((evaluate.ROOT/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, relax, sp, pred = [root/n for n in ('selection','relax','sp','predictor')]
            for folder in (selection,relax,sp,pred):
                folder.mkdir()
            structure = Structure(Lattice.cubic(3), ['Li'], [[0,0,0]])
            originals, energy_rows, prediction_rows = [], [], []
            for i in range(3):
                path = root/f'candidate{i}.vasp'
                structure.to(filename=str(path), fmt='poscar')
                candidate = dict(candidate_id=path.name, source=str(path), sha256=sha256(path),
                                 num_atoms=1, elements=['Li'], formula='Li1', original_gap=2.5, original_eform=-1.)
                originals.append(candidate)
                energy_rows.append(dict(candidate_id=path.name, input_sha256=sha256(path),
                                        status='completed', final_energy_eV=1.))
                record = dict(candidate_id=path.name, input_sha256=sha256(path), status='failed')
                folder = relax/path.stem
                folder.mkdir()
                if i:
                    structure.to(filename=str(folder/'relaxed.vasp'),fmt='poscar')
                    record.update(status='converged', final_energy_eV=-1., checkpoint_sha256=checkpoint,
                                  relaxed_sha256=sha256(folder/'relaxed.vasp'))
                (folder/'result.json').write_text(json.dumps(record))
                prediction_rows.append(dict(candidate_id=path.name,gap_before=2.5,eform_before=-1.,
                    gap_after=1. if i==1 else 2.5,eform_after=-1.,relaxation_status=record['status']))
            inventory = root/'inventory.json'
            author_csv = root/'author.csv'
            with author_csv.open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=['original_fnames','is_neutral_cmn'])
                writer.writeheader()
                writer.writerows(dict(original_fnames=c['candidate_id'],is_neutral_cmn='True') for c in originals)
            inventory.write_text(json.dumps(dict(denominator=3,candidates=originals,source_csv_sha256=sha256(author_csv))))
            (sp/'candidates.json').write_text(json.dumps(energy_rows))
            (sp/'environment.json').write_text(json.dumps(dict(inventory_sha256=sha256(inventory),checkpoint_sha256=checkpoint)))
            with (pred/'predictions.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(prediction_rows[0]))
                writer.writeheader()
                writer.writerows(prediction_rows)
            (pred/'summary.json').write_text(json.dumps(dict(control_passed=True,candidate_count=3,control_tolerance=.001)))
            reference = dict(originals[0],candidate_id='mp-reference.vasp')
            ref_folder = selection/'mattersim-relaxation/mp-reference'
            ref_folder.mkdir(parents=True)
            structure.to(filename=str(ref_folder/'relaxed.vasp'),fmt='poscar')
            (ref_folder/'result.json').write_text(json.dumps(dict(status='converged', input_sha256=reference['sha256'],
                checkpoint_sha256=checkpoint,relaxed_sha256=sha256(ref_folder/'relaxed.vasp'),final_energy_eV=0.)))
            (selection/'mattersim-inventory.json').write_text(json.dumps(dict(candidates=[reference],conversion_failures=[])))
            argv = ['evaluate.py','--inventory',str(inventory),'--selection',str(selection),
                    '--singlepoint',str(sp),'--predictor',str(pred),'--relaxation',str(relax),
                    '--author-csv',str(author_csv),
                    '--output',str(root/'out'),'--benchmark','fixture']
            def entry(path, record, identifier):
                return ComputedStructureEntry(Structure.from_file(path), record['final_energy_eV'],entry_id=identifier)
            with patch('sys.argv',argv), patch.dict(os.environ,{'SLURM_JOB_ID':'fixture'}), \
                    patch.object(evaluate,'corrected_entry',entry), \
                    patch.object(evaluate,'hull_with_timeout',hull_query), patch('builtins.print'):
                evaluate.main()
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['full_inventory']['denominator'],3)
            self.assertEqual(summary['converged_pairs']['denominator'],2)
            self.assertEqual(summary['converged_pairs']['metrics']['stable']['transitions']['gained'],2)
            self.assertEqual(summary['converged_pairs']['metrics']['property_success']['transitions']['lost'],1)
            self.assertEqual(summary['converged_pairs']['metrics']['stable_and_property']['after']['passed'],1)
            self.assertEqual(summary['full_inventory']['metrics']['unique']['after']['passed'],1)
            self.assertEqual(summary['converged_pairs']['metrics']['unique']['before']['passed'],1)


if __name__ == '__main__':
    unittest.main()
