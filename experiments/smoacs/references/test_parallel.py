import unittest
from parallel import partition


class PartitionTests(unittest.TestCase):
    def test_disjoint_exhaustive_and_large_cells_isolated(self):
        candidates = [dict(candidate_id=str(i), num_atoms=n) for i,n in enumerate([200,160,64,40,30,20,10,5])]
        groups = partition(candidates, 4)
        self.assertEqual({c['candidate_id'] for g in groups for c in g}, {str(i) for i in range(8)})
        self.assertEqual({c['num_atoms'] for c in groups[0]}, {200,160})
        self.assertTrue(all(c['num_atoms'] <= 128 for g in groups[1:] for c in g))
        with self.assertRaises(ValueError):
            partition(candidates+[candidates[0]], 4)


if __name__ == '__main__':
    unittest.main()
