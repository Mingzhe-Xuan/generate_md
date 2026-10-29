import json
from pathlib import Path
import tempfile
import unittest
from resume_property import continuation
from csp_model import sha


class ResumePropertyTests(unittest.TestCase):
    def test_continuation_preserves_order_status_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            selection=dict(selected=[dict(id='a'),dict(id='b')])
            raw=json.dumps(selection)
            (source/'selection.json').write_text(raw);(root/'selection.json').write_text(raw)
            policy=dict(fmax=.001)
            (source/'plan.json').write_text(json.dumps(dict(selection=selection,selection_sha256=sha(root/'selection.json'),policy=policy)))
            row=dict(id='a',status='anchor_unconverged')
            (source/'a').mkdir();(source/'a/result.json').write_text(json.dumps(row))
            (source/'progress.json').write_text(json.dumps([row]))
            carried,_=continuation(source,root/'selection.json',policy)
            self.assertEqual(carried['a']['status'],'anchor_unconverged')
            with self.assertRaisesRegex(ValueError,'policy'):continuation(source,root/'selection.json',dict(fmax=.01))
            (root/'selection.json').write_text(raw+' ')
            with self.assertRaisesRegex(ValueError,'selection'):continuation(source,root/'selection.json',policy)


if __name__=='__main__':unittest.main()
