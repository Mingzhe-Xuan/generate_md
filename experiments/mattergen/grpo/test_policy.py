import unittest
import torch
import hashlib
from recover_base import matching_tensors
from policy import (normal_logp, normal_kl, categorical_logp, categorical_kl,
                    matrix6, sym6, project_cell, advantages, grpo_loss, add_lora)


class PolicyContracts(unittest.TestCase):
    def test_backbone_recovery_requires_every_exact_official_tensor(self):
        v = torch.tensor([1., 2.]); expected = dict(shape=[2], dtype=str(v.dtype), sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest())
        manifest = dict(tensors={'base': expected})
        found, bad = matching_tensors({'base': v, 'adapter': torch.zeros(2)}, manifest)
        self.assertEqual(list(found), ['base']); self.assertEqual(bad, [])
        for state in ({}, {'base': v+1}, {'base': v.double()}, {'base': v.reshape(1, 2)}):
            self.assertEqual(matching_tensors(state, manifest)[1], ['base'])
    def test_gaussian_density_and_kl_match_distributions(self):
        m = torch.tensor([[.2, -.4]], dtype=torch.float64, requires_grad=True)
        s = torch.tensor([[.1, .3]], dtype=torch.float64)
        a = torch.tensor([[.3, -.5]], dtype=torch.float64)
        p = torch.distributions.Normal(m, s); q = torch.distributions.Normal(m.detach()+.1, 2*s)
        torch.testing.assert_close(normal_logp(a, m, s), p.log_prob(a).sum(-1))
        torch.testing.assert_close(normal_kl(m, s, q.loc, q.scale), torch.distributions.kl_divergence(p, q).sum(-1))
        grad, = torch.autograd.grad(normal_logp(a, m, s).sum(), m)
        torch.testing.assert_close(grad, (a-m)/s.square())

    def test_categorical_density_and_kl(self):
        p = torch.tensor([[.2, .4, .8]], dtype=torch.float64)
        q = torch.tensor([[.3, .1, .2]], dtype=torch.float64)
        a = torch.tensor([2])
        torch.testing.assert_close(categorical_logp(p, a), torch.distributions.Categorical(logits=p).log_prob(a))
        torch.testing.assert_close(categorical_kl(p, q), torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=p), torch.distributions.Categorical(logits=q)))

    def test_six_dimensional_symmetric_action_and_projection(self):
        v = torch.tensor([[1., -2., 3., .1, .2, -.1]], dtype=torch.float64)
        torch.testing.assert_close(sym6(matrix6(v)), v)
        c = project_cell(v)
        self.assertTrue(bool((torch.linalg.eigvalsh(c) > 0).all()))
        torch.testing.assert_close(c, c.transpose(-1, -2))
        torch.testing.assert_close(c@c, matrix6(v)@matrix6(v))

    def test_latent_wrapping_does_not_change_recorded_action(self):
        action = torch.tensor([[1.2, -.2, .5]], dtype=torch.float64)
        wrapped = action.remainder(1.)
        self.assertFalse(torch.equal(action, wrapped))
        self.assertNotEqual(float(normal_logp(action, torch.zeros_like(action), torch.ones_like(action))),
                            float(normal_logp(wrapped, torch.zeros_like(action), torch.ones_like(action))))

    def test_group_advantage_includes_failures_and_zero_variance(self):
        r = torch.tensor([-1., 1., .2, .2])
        torch.testing.assert_close(advantages(r, 2), torch.tensor([-1., 1., 0., 0.]))
        with self.assertRaises(ValueError): advantages(torch.ones(3), 2)

    def test_clipped_objective_handles_both_advantage_signs(self):
        ratio = torch.tensor([2., .5, 2., .5], dtype=torch.float64)
        adv = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
        loss, _ = grpo_loss(ratio.log(), torch.zeros_like(ratio), adv, torch.zeros_like(ratio), beta=0)
        self.assertAlmostEqual(float(loss), (-1.2-.5+2.+.8)/4)
        lp = torch.tensor([-.2, -.4], dtype=torch.float64)
        _, stats = grpo_loss(lp, lp, torch.tensor([1., -1.]), torch.zeros_like(lp))
        self.assertEqual(stats['ratio_min'], 1.); self.assertEqual(stats['ratio_max'], 1.)

    def test_lora_zero_initialization_and_base_frozen(self):
        torch.manual_seed(1); model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU(), torch.nn.Linear(4, 2)).double()
        x = torch.randn(8, 3, dtype=torch.float64); before = model(x).detach().clone()
        base = [p.detach().clone() for p in model.parameters()]
        names = add_lora(model, 2); self.assertEqual(names, ['0', '2'])
        torch.testing.assert_close(model(x), before, rtol=0, atol=0)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
        opt.zero_grad(); model(x).square().mean().backward(); opt.step()
        self.assertGreater(float((model(x)-before).abs().max().detach()), 0)
        frozen = [p for name, p in model.named_parameters() if '.base.' in name]
        for expected, actual in zip(base, frozen): torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
