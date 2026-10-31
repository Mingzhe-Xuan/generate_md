from pathlib import Path
import pickle
import tempfile
import unittest
import numpy as np
import torch
from checkpoint_io import load_checkpoint


class CheckpointIO(unittest.TestCase):
    def test_real_optimizer_numpy_lr_roundtrip_preserves_values_and_tensors(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-6)
        opt.param_groups[0]['lr'] = np.float64(2e-6)
        model(torch.ones(3, 2)).square().mean().backward(); opt.step()
        state = dict(adaptation={k: v.detach().clone() for k, v in model.state_dict().items()},
                     optimizer=opt.state_dict(), numpy_rng=np.random.default_rng(42).bit_generator.state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'; torch.save(state, path)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(path, map_location='cpu', weights_only=True)
            restored = load_checkpoint(path)
            for key in state['adaptation']:
                torch.testing.assert_close(state['adaptation'][key], restored['adaptation'][key], rtol=0, atol=0)
            self.assertEqual(restored['optimizer']['param_groups'][0]['lr'].hex(), np.float64(2e-6).hex())
            self.assertEqual(state['numpy_rng'], restored['numpy_rng'])
            opt2 = torch.optim.AdamW(model.parameters(), lr=1.)
            opt2.load_state_dict(restored['optimizer'])
            self.assertEqual(opt2.param_groups[0]['lr'], opt.param_groups[0]['lr'])
            for key in opt.state_dict()['state']:
                for name, value in opt.state_dict()['state'][key].items():
                    torch.testing.assert_close(value, opt2.state_dict()['state'][key][name], rtol=0, atol=0)

    def test_numpy_allowlist_does_not_leak_after_read(self):
        original = list(torch.serialization.get_safe_globals())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'; torch.save({'lr': np.float64(.001)}, path)
            self.assertEqual(load_checkpoint(path)['lr'], .001)
        self.assertEqual(torch.serialization.get_safe_globals(), original)


if __name__ == '__main__': unittest.main()
