from itertools import product
import random
from types import SimpleNamespace
import unittest

from neutrality import reachable_zero,super_common


class NeutralityTests(unittest.TestCase):
    def test_equivalence_to_exhaustive_assignments(self):
        randomizer=random.Random(42)
        for _ in range(500):
            options=[randomizer.sample(range(-4,5),randomizer.randrange(4)) for _ in range(randomizer.randrange(8))]
            expected=any(sum(values)==0 for values in product(*options))
            self.assertEqual(reachable_zero(options),expected)

    def test_mixed_valence_and_large_space(self):
        self.assertTrue(reachable_zero([[2,3]]*3+[[-2]]*4))
        self.assertFalse(reachable_zero([[2,3]]*3+[[-2]]*2))
        self.assertTrue(reachable_zero([[-1,0,1]]*20))
        self.assertFalse(reachable_zero([[1,2,3]]*20))
        self.assertFalse(reachable_zero([[1,-1],[]]))

    def test_identical_intersection_and_native_disagreement_rejected(self):
        def element(symbol):return SimpleNamespace(icsd_oxidation_states=[1,2,3],oxidation_states=[1,2,4],common_oxidation_states=[2,3])
        author=SimpleNamespace(Element_pmg=element,Element=lambda e:SimpleNamespace(oxidation_states=[2,5]),
            elec_neutral_check_SUPER_COMMON=lambda *args:(False,None,None))
        result,control=super_common([('X',2)],author)
        self.assertFalse(result);self.assertTrue(control['legacy_checked']);self.assertEqual(control['legacy_assignments'],1)
        author.elec_neutral_check_SUPER_COMMON=lambda *args:(True,None,None)
        with self.assertRaisesRegex(ValueError,'disagrees'):super_common([('X',2)],author)


if __name__=='__main__':unittest.main()
