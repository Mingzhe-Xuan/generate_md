import unittest
from formats import HARTREE_EV, parse_output, restart_atoms, validate_log, block


def output(bands):
    return ('Utot. -10.0\nEigenvalues (Hartree) of SCF KS-eq.\n'
            'Chemical Potential (Hatree) = 0.0\n' + bands + '\n****\nTotal Computational Time\n')


class FormatsTests(unittest.TestCase):
    def test_hubbard_and_moments_match_basis_without_double_counting(self):
        from ase import Atoms
        from formats import input_text
        atoms = Atoms('FeO', positions=[[0,0,0],[1,1,1]], cell=[4,4,4], pbc=True)
        species = {'Fe':dict(valence=16,basis='Fe6.0H-s3p3d2f1',potential='Fe_PBE19H'),
                   'O':dict(valence=6,basis='O6.0-s3p3d2f1',potential='O_PBE19')}
        physics = dict(initial_moments_muB={'Fe':5,'O':.6}, hubbards={
            'Fe':dict(U_eff_eV=5.3,angular_momentum=2), 'O':dict(U_eff_eV=0,angular_momentum=-1)})
        text = input_text(atoms, species, '/data', [3,3,3], physics=physics)
        self.assertIn('scf.Hubbard.U on', text)
        self.assertIn('1d 5.30000000 2d 0.00000000', text)
        self.assertEqual(text.count('5.30000000'), 1)
        self.assertIn('10.500000 5.500000', text)
        self.assertTrue(all(len(row) == 8 and row[-1] == 'on' for row in block(text, 'Atoms.SpeciesAndCoordinates')))
        self.assertEqual(len(block(text, 'Hubbard.U.values')[0]), 1+2*(3+3+2+1))
    def test_insulator_and_metal(self):
        gapped = output('kloop=0\n1 -.1 -.1\n2 .1 .1\nkloop=1\n1 -.09 -.09\n2 .12 .12\n')
        result = parse_output(gapped)
        self.assertAlmostEqual(result['sampled_bandgap_eV'], .19*HARTREE_EV)
        self.assertEqual(parse_output(gapped.replace('Hatree', 'Hartree')), result)
        metal = gapped.replace('1 -.09 -.09', '1 .01 .01')
        self.assertEqual(parse_output(metal)['sampled_bandgap_eV'], 0)

    def test_missing_or_truncated_bands(self):
        for bands in ('', 'kloop=0\n1 -.1 -.1\nkloop=1\n1 -.1 -.1\n2 .1 .1\n',
                      'kloop=0\n1 -.1 -.1\n3 .1 .1\n'):
            with self.assertRaises(ValueError):
                parse_output(output(bands))

    def test_convergence_is_explicit(self):
        good = ('The SCF was achieved at MD= 3\nThe geometry optimization was achieved.\n'
                '|Maximum gradient| (Hartree/Bohr) =0.00001\n')
        self.assertEqual(validate_log(good, True)['final_md'], 3)
        self.assertEqual(validate_log(good.replace('gradient', 'dE/da'), True)['final_md'], 3)
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
