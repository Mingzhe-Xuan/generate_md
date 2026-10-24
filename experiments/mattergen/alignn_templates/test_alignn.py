import unittest
import importlib.util
from pathlib import Path

import torch

from geometry import edge_features,recover_images
from report import paired,summarize
from device_control import topology_signature
from continuation import inference_signature,pre_A_only


class AlignnTemplateTests(unittest.TestCase):
    def test_continuation_allows_only_control_changes_and_no_partial_A(self):
        import tempfile
        original='def predict(x, native_control=False):\n y=x*2\n if native_control:\n  print(y)\n return y\n'
        control=original.replace('print(y)','print(y, x)')
        changed=original.replace('x*2','x*3')
        self.assertEqual(inference_signature(original),inference_signature(control))
        self.assertNotEqual(inference_signature(original),inference_signature(changed))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'protocol.json').write_text('{}')
            self.assertTrue(pre_A_only(path))
            (path/'A-trace.jsonl').write_text('actual update')
            self.assertFalse(pre_A_only(path))

    def test_graph_control_ignores_edge_order_but_detects_periodic_topology(self):
        from types import SimpleNamespace
        first=SimpleNamespace(edges=lambda **kw:(torch.tensor([0,1]),torch.tensor([1,0])))
        second=SimpleNamespace(edges=lambda **kw:(torch.tensor([1,0]),torch.tensor([0,1])))
        a=(first,None,torch.tensor([[-1.,0,0],[1.,0,0]]))
        b=(second,None,torch.tensor([[1.,0,0],[-1.,0,0]]))
        self.assertEqual(topology_signature(a),topology_signature(b))
        c=(first,None,torch.zeros(2,3))
        self.assertNotEqual(topology_signature(a),topology_signature(c))

    def test_entry_resolves_shared_initialization_geometry_checker(self):
        spec=importlib.util.spec_from_file_location('alignn_entry_test',Path(__file__).with_name('run.py'))
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        from solver import gram_check
        self.assertTrue(gram_check(torch.eye(3).numpy(),torch.eye(3).numpy())['passed'])

    def test_signed_reverse_images_and_reconstruction_guard(self):
        cell=torch.eye(3,dtype=torch.float64)*4
        frac=torch.tensor([[.1,.2,.3],[.8,.4,.2]],dtype=torch.float64)
        src=torch.tensor([0,1])
        dst=torch.tensor([1,0])
        images=torch.tensor([[-1.,0,0],[1.,0,0]],dtype=torch.float64)
        vectors=(frac[dst]-frac[src]+images)@cell
        actual=recover_images(vectors,cell,frac,src,dst)
        torch.testing.assert_close(actual,images)
        self.assertFalse(torch.equal(actual[0],actual[1]))
        with self.assertRaises(ValueError):
            recover_images(vectors+.1,cell,frac,src,dst)

    def test_lattice_and_fractional_geometry_gradients(self):
        cell=torch.tensor([[4.,0,0],[.2,3.7,0],[.1,.3,4.1]],dtype=torch.float64,requires_grad=True)
        frac=torch.tensor([[.1,.2,.3],[.8,.4,.2],[.2,.7,.4]],dtype=torch.float64,requires_grad=True)
        src=torch.tensor([0,1,1,2])
        dst=torch.tensor([1,0,2,1])
        images=torch.tensor([[-1.,0,0],[1.,0,0],[0,0,0],[0,0,0]],dtype=torch.float64)
        line_src=torch.tensor([0,1,2])
        line_dst=torch.tensor([2,0,3])
        def objective(c,f):
            r,h=edge_features(c,f,src,dst,images,line_src,line_dst)
            return r.square().sum()+.3*h.square().sum()
        grads=torch.autograd.grad(objective(cell,frac),(cell,frac))
        for index,parameter in enumerate((cell,frac)):
            direction=torch.arange(1,parameter.numel()+1,dtype=torch.float64).reshape(parameter.shape)
            direction/=direction.norm()
            eps=1e-6
            with torch.no_grad():
                args=[cell,frac]
                args[index]=parameter+eps*direction
                upper=objective(*args)
                args[index]=parameter-eps*direction
                lower=objective(*args)
            self.assertAlmostEqual(float((upper-lower)/(2*eps)),float((grads[index]*direction).sum()),places=6)
        torch.testing.assert_close(grads[1].sum(0),torch.zeros(3,dtype=torch.float64),atol=1e-12,rtol=0)
        torch.testing.assert_close(objective(cell,frac+.4),objective(cell,frac))

    def test_degenerate_bond_is_rejected(self):
        with self.assertRaises(ValueError):
            edge_features(torch.eye(3),torch.zeros(2,3),torch.tensor([0]),torch.tensor([1]),
                torch.zeros(1,3),torch.tensor([0]),torch.tensor([0]))

    def row(self,key,gap):
        flags=dict(gap=None,eform=None,joint=None) if gap is None else dict(gap=True,eform=True,joint=False)
        return dict(candidate_id=key,crystalformer_gap=gap,alignn_gap=gap,crystalformer_flags=flags,alignn_flags=flags)

    def test_unknowns_remain_in_denominator(self):
        rows=[self.row('a',2.6),self.row('b',None)]
        result=summarize(rows)
        self.assertEqual(result['denominator'],2)
        self.assertEqual(result['alignn']['unknown'],1)
        self.assertEqual(result['alignn']['gap_pass'],1)
        self.assertAlmostEqual(result['alignn']['MAE'],.1)

    def test_same_parent_pairing_and_different_coverage(self):
        first=[self.row('a',2.6),self.row('b',None)]
        second=[self.row('b',2.5),self.row('a',2.7)]
        result=paired(first,second,['a','b'])
        self.assertEqual(result['alignn']['ids'],['a'])
        self.assertAlmostEqual(result['alignn']['first_MAE'],.1)
        self.assertAlmostEqual(result['alignn']['second_MAE'],.2)
        with self.assertRaises(ValueError):
            paired(first,second[:1],['a','b'])


if __name__=='__main__':
    unittest.main()
