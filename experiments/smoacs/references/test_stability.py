import unittest
from assess import stable_status, hull_query


class StabilityTests(unittest.TestCase):
    def test_incomplete_hull_cannot_certify_stability(self):
        self.assertIsNone(stable_status(.01, False))
        self.assertIsNone(stable_status(None, True))
        self.assertTrue(stable_status(.09999, True))
        self.assertFalse(stable_status(.1, True))
        self.assertFalse(stable_status(.10001, True))
        self.assertFalse(stable_status(.2, False))

    def test_lp_matches_full_phase_diagram(self):
        from pymatgen.entries.computed_entries import ComputedEntry
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        references = [ComputedEntry('Li', 0), ComputedEntry('O', 0),
                      ComputedEntry('Fe', 0), ComputedEntry('Li2O', -6), ComputedEntry('FeO', -4)]
        for formula, energy in [('LiO', -2), ('LiFeO2', -5), ('LiFeO2', -12)]:
            candidate = ComputedEntry(formula, energy)
            subset = [r for r in references if set(r.composition.elements) <= set(candidate.composition.elements)]
            expected = PhaseDiagram(subset+[candidate]).get_e_above_hull(candidate)
            self.assertAlmostEqual(hull_query(subset, candidate)['hull_eV_atom'], expected, places=8)

    def test_nine_element_query_and_infeasible_composition(self):
        from pymatgen.entries.computed_entries import ComputedEntry
        elements = ['Li','Na','K','Ca','Mg','Al','Si','N','O']
        references = [ComputedEntry(e, -i) for i,e in enumerate(elements)]
        candidate = ComputedEntry(dict.fromkeys(elements, 1), -sum(range(9))+9*.2)
        self.assertAlmostEqual(hull_query(references, candidate)['hull_eV_atom'], .2, places=8)
        with self.assertRaises(ValueError):
            hull_query(references[:-1], candidate)


if __name__ == '__main__':
    unittest.main()
