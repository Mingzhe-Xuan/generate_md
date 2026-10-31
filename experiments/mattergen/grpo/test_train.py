import unittest
import torch
from train import subset_record, microbatches, property_metrics, validation_key
from models import tensor_state_hash


class TrainingContracts(unittest.TestCase):
    def test_frozen_hash_covers_buffers_and_excludes_trainable_parameters(self):
        model = torch.nn.Linear(2, 1)
        model.register_buffer('fixed_buffer', torch.tensor([1.]))
        digest = tensor_state_hash(model, frozen_only=True)
        with torch.no_grad(): model.weight.add_(1)
        self.assertEqual(tensor_state_hash(model, frozen_only=True), digest)
        model.fixed_buffer.add_(1)
        self.assertNotEqual(tensor_state_hash(model, frozen_only=True), digest)

    def test_microbatch_action_state_and_old_probability_alignment(self):
        counts = torch.tensor([2, 3, 1]); atoms = torch.arange(6)
        record = dict(state=dict(num_atoms=counts, pos=atoms[:, None].expand(-1, 3),
            atomic_numbers=atoms+1, cell=torch.arange(27).reshape(3, 3, 3)),
            action=dict(pos=atoms[:, None].expand(-1, 3)+100, cell=torch.arange(18).reshape(3, 6), atomic_numbers=atoms+20),
            old_logp=torch.tensor([.1, .2, .3]), t=.5, dt=-.001, stage='predictor', index=500)
        piece = subset_record(record, [2, 0])
        self.assertEqual(piece['state']['num_atoms'].tolist(), [1, 2])
        self.assertEqual(piece['state']['atomic_numbers'].tolist(), [6, 1, 2])
        self.assertEqual(piece['action']['atomic_numbers'].tolist(), [25, 20, 21])
        torch.testing.assert_close(piece['old_logp'], torch.tensor([.3, .1]))
        self.assertEqual(piece['action']['cell'].tolist(), [[12, 13, 14, 15, 16, 17], [0, 1, 2, 3, 4, 5]])

    def test_microbatch_limits_preserve_all_samples_once(self):
        parts = list(microbatches([20, 20, 20, 20, 4, 4], max_graphs=4, max_atoms=64))
        self.assertEqual(parts, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual([i for part in parts for i in part], list(range(6)))

    def test_failures_stay_in_success_denominator(self):
        events = [dict(valid=True, pass_gap=True, error_eV=.05, reward=.9), dict(valid=False, reward=-1.)]
        metrics = property_metrics(events)
        self.assertEqual(metrics['gap_pass_fraction'], .5)
        self.assertEqual(metrics['valid_fraction'], .5)
        self.assertEqual(metrics['gap_MAE_valid_eV'], .05)
        self.assertLess(validation_key(property_metrics([dict(valid=False, reward=-1.)])), validation_key(metrics))


if __name__ == '__main__': unittest.main()
