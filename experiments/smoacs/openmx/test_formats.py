import unittest
from formats import HARTREE_EV, parse_output, restart_atoms, validate_log


def output(bands):
    return ('Utot. -10.0\nEigenvalues (Hartree) of SCF KS-eq.\n'
            'Chemical Potential (Hatree) = 0.0\n' + bands + '\n****\nTotal Computational Time\n')


class FormatsTests(unittest.TestCase):
    def test_insulator_and_metal(self):
        gapped = output('kloop=0\n1 -.1 -.1\n2 .1 .1\nkloop=1\n1 -.09 -.09\n2 .12 .12\n')
        result = parse_output(gapped)
        self.assertAlmostEqual(result['sampled_bandgap_eV'], .19*HARTREE_EV)
        metal = gapped.replace('1 -.09 -.09', '1 .01 .01')
        self.assertEqual(parse_output(metal)['sampled_bandgap_eV'], 0)

    def test_missing_or_truncated_bands(self):
        for bands in ('', 'kloop=0\n1 -.1 -.1\nkloop=1\n1 -.1 -.1\n2 .1 .1\n',
                      'kloop=0\n1 -.1 -.1\n3 .1 .1\n'):
            with self.assertRaises(ValueError):
                parse_output(output(bands))

    def test_convergence_is_explicit(self):
        good = 'The SCF was achieved at MD= 3\n|Maximum gradient| (Hartree/Bohr) =0.00001\n'
        self.assertEqual(validate_log(good, True)['final_md'], 3)
        for bad in ('', good+'The SCF was not fully achieved at MD=2', good.replace('.00001','.001')):
            with self.assertRaises(ValueError):
                validate_log(bad, True)

    def test_final_geometry_and_units(self):
        text = ('Atoms.SpeciesAndCoordinates.Unit Ang\nAtoms.UnitVectors.Unit Ang\n'
                '<Atoms.SpeciesAndCoordinates\n1 Si .5 .6 .7 2 2\nAtoms.SpeciesAndCoordinates>\n'
                '<Atoms.UnitVectors\n3 0 0\n0 3 0\n0 0 3\nAtoms.UnitVectors>\n')
        self.assertAlmostEqual(restart_atoms(text).positions[0, 0], .5)
        with self.assertRaises(ValueError):
            restart_atoms(text.replace('Unit Ang', 'Unit AU'))


if __name__ == '__main__':
    unittest.main()
