import json
from pathlib import Path
import tempfile
import unittest

from fetch import digest
from reference_resume import validate


class ReferenceResumeTests(unittest.TestCase):
    def test_reuses_valid_cache_without_changing_full_reference_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);selection=run/'references/selected';selection.mkdir(parents=True)
            scope=run/'reference-scope.json';scope.write_text('{}')
            (selection/'summary.json').write_text(json.dumps(dict(inventory_sha256=digest(scope),snapshot='2026-04-13',reference_filter='published Zenodo20149017')))
            candidates=[]
            for name in ('a','b'):
                source=selection/(name+'.vasp');source.write_text(name)
                candidates.append(dict(candidate_id=name+'.vasp',source=str(source),sha256=digest(source)))
            (selection/'mattersim-inventory.json').write_text(json.dumps(dict(candidates=candidates)))
            folder=selection/'mattersim-relaxation/a';folder.mkdir(parents=True);pose=folder/'relaxed.vasp';pose.write_text('pose')
            record=dict(input_sha256=candidates[0]['sha256'],checkpoint_sha256='pinned',status='converged',converged=True,max_filter_force=.009,relaxed_sha256=digest(pose))
            (folder/'result.json').write_text(json.dumps(record))
            result=validate(run,selection,'pinned')
            self.assertEqual((result['references'],result['cached'],result['missing']),(2,1,1))
            self.assertEqual(result['mode'],'canonical_serial')
            pose.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Invalid cached'):validate(run,selection,'pinned')
            pose.write_text('pose');scope.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'scope hash'):validate(run,selection,'pinned')


if __name__=='__main__':unittest.main()
