import itertools
import random
import unittest

from template_policy import neutral_assignments, select_pilot


class TemplatePolicyTests(unittest.TestCase):
    def test_matches_exhaustive_neutral_paths(self):
        rng = random.Random(42)
        for _ in range(100):
            options = [rng.sample(range(-2, 4), rng.randint(1, 4)) for _ in range(rng.randint(1, 5))]
            expected = sorted(p for p in itertools.product(*map(sorted, options)) if sum(p) == 0)[:10]
            self.assertEqual(neutral_assignments(options), expected)

    def test_mixed_valence_empty_and_large(self):
        self.assertEqual(neutral_assignments([[2, 3], [2, 3], [-2], [-2], [-2]]), [(3, 3, -2, -2, -2)])
        self.assertEqual(neutral_assignments([[1], [1]]), [])
        self.assertEqual(neutral_assignments([[0], []]), [])
        paths = neutral_assignments([[-1, 0, 1]]*20)
        self.assertEqual(len(paths), 10)
        self.assertTrue(all(sum(p) == 0 and len(p) == 20 for p in paths))

    def test_selection_ignores_properties_and_ineligible(self):
        rows = [dict(candidate_id=str(n), num_atoms=n, eligible=True, gap=999) for n in (4, 8, 12, 20)]
        rows += [dict(candidate_id='bad', num_atoms=4, eligible=False, gap=2.5)]
        self.assertEqual([r['num_atoms'] for r in select_pilot(rows)], [4, 8, 12, 20])
        with self.assertRaises(ValueError):
            select_pilot(rows[:3])


if __name__ == '__main__':
    unittest.main()
