import json
from pathlib import Path
import tempfile
import unittest

from resume import INVARIANTS,restore


class ResumeTests(unittest.TestCase):
    def test_pending_B_restores_exact_moments_without_repeating_A(self):
        import torch
        p=[torch.nn.Parameter(torch.ones(2,3)) for _ in range(5)]
        opt=[torch.optim.Adam(p[:2],lr=.002),torch.optim.Adam(p[3:],lr=.007),torch.optim.Adam([p[2]],lr=.005)]
        for _ in range(3):
            for x in p:x.grad=2*x.detach()
            for o in opt:o.step()
        bounds=[torch.ones(2,3)*i for i in range(4)]
        protocol={k:'same' for k in INVARIANTS};protocol['ids']=['a','b']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=dict(protocol=protocol,current_cycle=1,phase='B_pending',cycles=[],states=[])
            (root/'results.json').write_text(json.dumps(data))
            torch.save(dict(targets=[x.detach() for x in p],optimizers=[o.state_dict() for o in opt],bounds=bounds,
                active=torch.tensor([True,False]),cycle=1,phase='B_pending'),root/'checkpoint.pt')
            q=[torch.nn.Parameter(torch.zeros_like(x)) for x in p]
            restored_opt=[torch.optim.Adam(q[:2]),torch.optim.Adam(q[3:]),torch.optim.Adam([q[2]])]
            restored_bounds=[torch.zeros_like(x) for x in bounds]
            _,mask,cycle,pending=restore(root,protocol,q,restored_opt,restored_bounds)
            self.assertEqual(cycle,1);self.assertTrue(pending);self.assertEqual(mask.tolist(),[True,False])
            for o in restored_opt:
                for value in o.state.values():self.assertEqual(value['step'].item(),3)
            for values,optimizers in ((p,opt),(q,restored_opt)):
                for x in values:x.grad=2*x.detach()
                for o in optimizers:o.step()
            for a,b in zip(p+bounds,q+restored_bounds):torch.testing.assert_close(a,b,rtol=0,atol=0)
            data['phase']='cycle_completed';(root/'results.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'phase mismatch'):restore(root,protocol,q,restored_opt,restored_bounds)

    def test_numerical_protocol_change_rejected_before_state_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol={k:'old' for k in INVARIANTS}
            (root/'results.json').write_text(json.dumps(dict(protocol=protocol)))
            changed=dict(protocol,steps_per_A=600)
            with self.assertRaisesRegex(ValueError,'steps_per_A'):restore(root,changed,[],[],[])


if __name__=='__main__':unittest.main()
