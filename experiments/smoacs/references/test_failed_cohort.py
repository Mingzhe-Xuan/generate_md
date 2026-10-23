import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from assess import main


class FailedCohortTests(unittest.TestCase):
    def test_all_failed_proposals_still_write_complete_candidate_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proposal=root/'proposal';selection=root/'selection';selection.mkdir()
            (proposal/'inventory-249').mkdir(parents=True);candidates=[]
            for identifier in ('a','b','c','d'):
                candidates.append(dict(candidate_id=identifier,sha256='input',elements=['Na','Cl']))
                destination=proposal/'relaxation'/identifier;destination.mkdir(parents=True)
                (destination/'result.json').write_text(json.dumps(dict(input_sha256='input',status='failed')))
            (proposal/'inventory-249/inventory.json').write_text(json.dumps(dict(denominator=4,candidates=candidates)))
            (selection/'mattersim-inventory.json').write_text(json.dumps(dict(candidates=[],conversion_failures=[])))
            with gzip.open(selection/'novelty-structures.jsonl.gz','wt') as stream:stream.write('')
            output=root/'assessment'
            with patch.dict(os.environ,SLURM_JOB_ID='unit'),patch('sys.argv',['assess','--selection',str(selection),
                    '--proposal-root',str(proposal),'--output',str(output)]):main()
            rows=json.loads((output/'candidates.json').read_text());self.assertEqual(len(rows),4)
            self.assertTrue(all(r['sun'] is False and r['stable'] is None for r in rows))
            self.assertEqual(json.loads((output/'summary.json').read_text())['denominator'],4)


if __name__=='__main__':unittest.main()
