import importlib.util
from pathlib import Path
import unittest
import numpy as np
import json
import tempfile
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('score_probe_math_test', Path(__file__).with_name('probe_math.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ScoreProbeTests(unittest.TestCase):
    def test_covector_skew_rotation_and_translation(self):
        cell = np.array([[3., .2, .1], [.6, 4., .3], [.2, .5, 5.]])
        force = m.project(np.arange(12).reshape(4, 3)**2).reshape(-1, 3)
        fractional = force @ cell.T
        np.testing.assert_allclose(m.cartesian_score(fractional, cell), force.ravel(), atol=1e-12)
        theta = .43
        rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        np.testing.assert_allclose(m.cartesian_score(fractional, cell@rot), (force@rot).ravel(), atol=1e-12)
        np.testing.assert_allclose(m.project(force+23), force.ravel())

    def test_zero_nonfinite_and_collinear(self):
        self.assertIsNone(m.cosine(np.zeros(6), np.ones(6)))
        with self.assertRaises(ValueError):
            m.project([[np.nan, 0, 1]])
        result = m.local_projection([1, 0], [1, 0], [-1, 0])
        self.assertEqual(result['rank'], 1)
        self.assertAlmostEqual(result['explained_fraction'], 1.)

    def fixture(self):
        rows = []
        rng = np.random.default_rng(34)
        for p in range(8):
            for _ in range(4):
                a = m.unit(rng.normal(size=12))
                b = m.unit(rng.normal(size=12))
                # Known fixed coefficient target, already unit length.
                b = m.unit(b-(b@a)*a)
                rows.append(dict(parent_id=str(p), F=a, G=b, target=.6*a+.8*b))
        return rows

    def test_out_of_parent_joint_beats_force(self):
        rows = self.fixture()
        joint = m.heldout(rows, ['F', 'G'])
        force = m.heldout(rows, ['F'])
        self.assertAlmostEqual(joint['heldout_R2_zero_baseline'], 1.)
        for fold in joint['folds']:
            np.testing.assert_allclose(fold['coefficient'], [.6, .8])
        gain = m.paired_parent_interval(joint, force)
        self.assertAlmostEqual(gain['R2_gain'], .64)
        self.assertGreater(gain['parent_bootstrap_CI95'][0], .63)
        self.assertEqual(len(joint['folds']), 8)

    def test_negative_r2_retained_and_parent_minimum(self):
        rows = self.fixture()[:8]
        with self.assertRaises(ValueError):
            m.heldout(rows, ['F'])
        rows = self.fixture()
        rows[-4:] = [dict(r, target=-r['target']*100) for r in rows[-4:]]
        fitted = m.heldout(rows, ['F', 'G'])
        self.assertLess(fitted['folds'][-1]['MSE'], 4.01)
        self.assertGreater(fitted['folds'][-1]['MSE'], 3.9)

    def test_complete_report_hash_coverage_and_json(self):
        import hashlib
        old = sys.modules.get('probe_math')
        sys.modules['probe_math'] = m
        try:
            spec = importlib.util.spec_from_file_location('score_probe_report_test', Path(__file__).with_name('report_probe.py'))
            report = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(report)
        finally:
            if old is None:
                sys.modules.pop('probe_math', None)
            else:
                sys.modules['probe_math'] = old
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policy = dict(anchors=['raw','relaxed'],times=[.05,.15,.3],replicas=2,seed=34)
            rows = []
            for parent in range(16):
                for anchor in policy['anchors']:
                    for t in policy['times']:
                        for replica in range(2):
                            a = np.array([1.,0.,0.,-1.,0.,0.])
                            b = np.array([0.,1.,0.,0.,-1.,0.])
                            filename = f'{len(rows)}.npz'
                            names=('unconditional','conditional','delta','guided','drift_unconditional','drift_conditional','drift_delta','drift_guided')
                            np.savez(root/filename,F=a,G=b,restoration=a,**dict(zip(names,[a,a+b,b,a+2*b]*2)))
                            rows.append(dict(parent_id=str(parent),anchor=anchor,t=t,replica=replica,
                                vector_path=filename,vector_sha256=hashlib.sha256((root/filename).read_bytes()).hexdigest(),
                                noise_RMS_A=.1,cos_F_G=0.,cosine={k:dict(force=0.,property=1.) for k in names},
                                local_projection={k:dict(explained_fraction=1.) for k in names}))
            (root/'plan.json').write_text(json.dumps(dict(policy=policy,inputs=[dict(parent_id=str(i)) for i in range(16)])))
            complete=dict(passed=True,policy=policy,states=rows,elapsed_seconds=1.,NFE=192,native_checks=[])
            (root/'complete.json').write_text(json.dumps(complete))
            with patch.object(report,'plot'):
                output=report.report(root)
            self.assertEqual(output['states'],192)
            self.assertAlmostEqual(output['groups'][0]['targets']['delta']['fits']['joint']['heldout_R2_zero_baseline'],1.)
            (root/'0.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                report.report(root)


if __name__ == '__main__':
    unittest.main()
