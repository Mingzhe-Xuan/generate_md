from pathlib import Path
import importlib.util
from types import SimpleNamespace
import unittest

import torch
from fixed_species import forward

spec = importlib.util.spec_from_file_location('fixed_test_microbatch', Path(__file__).resolve().parents[2]/'smoacs/adam_cycles/microbatch.py')
microbatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(microbatch)
Microbatch = microbatch.Microbatch


class FixedSpeciesTests(unittest.TestCase):
    def context(self):
        size = torch.tensor([2, 2, 2, 2])
        batch = torch.repeat_interleave(torch.arange(4), size)
        onehot = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]), 98).double()
        targets = [torch.nn.Parameter(x) for x in (torch.ones(4, 3, dtype=torch.float64),
            torch.ones(4, 3, dtype=torch.float64), torch.ones(8, 3, dtype=torch.float64),
            onehot.clone(), torch.zeros(4, 10, dtype=torch.float64))]
        seen = []

        def preprocess(**kw):
            lengths = kw['abc_scaler'].rescale(kw['scaled_batch_abc'])
            angles = kw['angle_scaler'].rescale(kw['scaled_batch_angle'])
            local_batch = torch.repeat_interleave(torch.arange(len(kw['size'])), kw['size'])
            coords = kw['batch_dir_coords']
            pos = coords+lengths[local_batch]+angles[local_batch]*.01
            return lengths, angles, coords, torch.full_like(kw['atomic_distribution'], 1/98), pos, torch.diag_embed(lengths), kw['ox_states_used_mask']

        def prediction(**kw):
            seen.append(kw['normalized_dist'].detach().clone())
            self.assertTrue(torch.all(kw['normalized_dist'].sum(1)==1))
            self.assertTrue(torch.all((kw['normalized_dist']==0)|(kw['normalized_dist']==1)))
            signal = torch.zeros(len(kw['size']), dtype=torch.float64).index_add(0, kw['batch'], kw['pos'].sum(1))
            return signal, signal*.1

        author = SimpleNamespace(Crystalformer_preprocess=preprocess, Crystalformer_prediction=prediction)
        author.forward_propagation_Crystalformer = lambda **kw: forward(author, **kw)
        context = dict(ids=list('abcd'), targets=targets, size=size, batch=batch,
            slices=[slice(i*2,i*2+2) for i in range(4)],
            data=dict(size=size, batch=batch, fixed_atomic_distribution=onehot, init_coords=targets[2].detach().clone(),
                      atomic_mask=torch.ones(8,98,10), ox_states_used_mask=torch.ones(8,10)),
            base=torch.ones(4,3), lower=torch.ones(4,3), upper=torch.ones(4,3)*10,
            angle_lower=torch.ones(4,3)*60, angle_upper=torch.ones(4,3)*120,
            gap_model=None, ef_model=None, author=author, fixed_species=True, unconstrained_elements=True)
        return context, seen

    def test_exact_species_zero_placeholder_gradients_and_partition_equivalence(self):
        c, seen = self.context()
        active = torch.ones(4,dtype=torch.bool)
        a, ga = Microbatch(c,2).gradients(.01,active)
        b, gb = Microbatch(c,4).gradients(.01,active)
        self.assertTrue(torch.allclose(a,b))
        for x,y in zip(ga,gb):
            self.assertTrue(torch.allclose(x,y))
        self.assertTrue(all(torch.count_nonzero(g)>0 for g in ga[:3]))
        self.assertTrue(all(torch.count_nonzero(g)==0 for g in ga[3:]))
        self.assertTrue(torch.equal(torch.cat(seen[:2]), c['data']['fixed_atomic_distribution']))
        saved = [p.detach().clone() for p in c['targets']]
        optimizer = torch.optim.Adam(c['targets'],lr=.002)
        for _ in range(25):
            _, gradients = Microbatch(c,2).gradients(.01,active)
            optimizer.zero_grad()
            for p,g in zip(c['targets'],gradients):
                p.grad = g
            optimizer.step()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(saved[3:],c['targets'][3:])))
        self.assertTrue(all(not torch.equal(x,y) for x,y in zip(saved[:3],c['targets'][:3])))

    def test_mutated_fixed_species_rejected_before_prediction(self):
        c, seen = self.context()
        with torch.no_grad():
            c['targets'][3][0,0] = .5
        with self.assertRaisesRegex(ValueError,'per-site elements changed'):
            Microbatch(c,2).gradients(.01,torch.ones(4,dtype=torch.bool))
        self.assertEqual(seen,[])


if __name__ == '__main__':
    unittest.main()
