import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
from unittest.mock import patch
import numpy as np


def load(name):
    spec=importlib.util.spec_from_file_location('orthogonal_test_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

m=load('projection');Offline=load('offline_references').OfflineRanges


class ProjectionTests(unittest.TestCase):
    def test_offline_module_import_without_numpy(self):
        import builtins
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name=='numpy' or name.startswith('numpy.'):
                raise ImportError('Minimal reference environment has no numpy')
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            module=load('offline_references')
        self.assertEqual(module.OfflineRanges.block_size,1024*1024)

    def test_positive_negative_zero_dot_without_threshold(self):
        force=np.array([[1.,0.,0.],[-1.,0.,0.]])
        for direction in [np.array([[.2,2.,0.],[-.2,-2.,0.]]),np.array([[-.2,2.,0.],[.2,-2.,0.]]),np.array([[0.,2.,0.],[0.,-2.,0.]])]:
            changed,meta=m.remove_parallel(direction,np.eye(3),force)
            self.assertTrue(meta['triggered'])
            self.assertLess(abs(np.sum(changed*force)),1e-12)
            self.assertLess(abs(meta['cosine_before']),.7)
            self.assertLessEqual(np.linalg.norm(changed),np.linalg.norm(direction)+1e-12)

    def test_skew_roundtrip_force_scale_rotation_com(self):
        rng=np.random.default_rng(3);cell=np.array([[3.,.2,.4],[.2,5.,.3],[.6,.3,7.]])
        sf=rng.normal(size=(7,3));force=rng.normal(size=(7,3))
        changed,meta=m.remove_parallel(sf,cell,force)
        scaled,_=m.remove_parallel(sf,cell,force*173)
        np.testing.assert_allclose(changed,scaled,atol=1e-12)
        np.testing.assert_allclose((changed@cell).mean(0),(sf@cell).mean(0),atol=1e-12)
        rot=np.array([[0.,1,0],[-1,0,0],[0,0,1]])
        rotated,_=m.remove_parallel(sf,cell@rot,force@rot)
        np.testing.assert_allclose(rotated,changed,atol=1e-12)
        self.assertAlmostEqual(meta['before_norm']**2,meta['after_norm']**2+meta['removed_norm']**2)

    def test_parallel_zero_force_support(self):
        f=np.array([[1.,2.,3.],[-1.,-2.,-3.]])
        out,meta=m.remove_parallel(f,np.eye(3),f)
        np.testing.assert_allclose(out,0,atol=1e-12)
        self.assertIsNone(meta['cosine_after'])
        out,meta=m.remove_parallel(f,np.eye(3),f*0)
        np.testing.assert_array_equal(out,f);self.assertFalse(meta['triggered'])
        self.assertEqual(m.force_supported([101,8],np.eye(3),f,94),'masked_atom_types')
        self.assertEqual(m.force_supported([95,8],np.eye(3),f,94),'unsupported_atom_types')
        self.assertIsNone(m.force_supported([8,8],np.eye(3),f,94))

    def test_injected_float32_still_orthogonal(self):
        rng=np.random.default_rng(54)
        cell=np.array([[2.,.7,.3],[.3,4.,.8],[.6,.4,6.]])
        force=rng.normal(size=(17,3));score=rng.normal(size=(17,3)).astype(np.float32)
        out,meta=m.remove_parallel(score,cell,force)
        actual=out.astype(np.float32).astype(float)@cell
        f=force-force.mean(0);actual-=actual.mean(0)
        self.assertLess(abs(np.sum(actual*f))/(np.linalg.norm(actual)*np.linalg.norm(f)),1e-6)
        np.testing.assert_array_equal(score,score.copy())

    def test_offline_reader_full_edges_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            key='example';folder=Path(tmp)/hashlib.sha256(key.encode()).hexdigest();folder.mkdir()
            size=Offline.block_size+7;data=np.random.default_rng(2).bytes(size)
            (folder/'object.json').write_text(json.dumps(dict(key=key,size=size,etag='immutable')))
            for i,start in enumerate(range(0,size,Offline.block_size)):
                p=folder/f'{i:06d}.part';piece=data[start:start+Offline.block_size];p.write_bytes(piece)
                p.with_suffix('.sha256').write_text(hashlib.sha256(piece).hexdigest())
            with Offline(key,tmp) as f:
                f.seek(Offline.block_size-3);self.assertEqual(f.read(10),data[-10:]);f.seek(0);self.assertEqual(f.read(),data)
                f.seek(-4,2);self.assertEqual(f.read(),data[-4:]);self.assertEqual(f.read(),b'')
            (folder/'000001.part').write_bytes(b'changed')
            with Offline(key,tmp) as f:
                f.seek(Offline.block_size)
                with self.assertRaises(ValueError):f.read()


if __name__=='__main__':unittest.main()
