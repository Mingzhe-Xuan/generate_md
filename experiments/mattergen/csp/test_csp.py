"""Contract tests for CSP chemistry, source integrity and sampling configuration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml

from validation import digest, validate_composition, validate_csp_config, validate_sampling

HERE = Path(__file__).resolve().parent


class CSPContracts(unittest.TestCase):
    def test_published_config_matches_both_hashes(self):
        spec = json.loads((HERE / 'sources.json').read_text())
        self.assertEqual(digest(HERE / 'model_config.yaml'), spec['config_sha256'])
        self.assertEqual(digest(HERE / 'model_config.yaml', 'md5'), spec['config_md5'])

    def test_author_csp_configuration(self):
        validate_csp_config(yaml.safe_load((HERE / 'model_config.yaml').read_text()))

    def test_reject_atom_type_loss_or_corruption(self):
        for change in ('loss', 'corruption'):
            config = yaml.safe_load((HERE / 'model_config.yaml').read_text())
            module = config['lightning_module']['diffusion_module']
            if change == 'loss':
                module['loss_fn']['include_atomic_numbers'] = True
            else:
                module['corruption']['discrete_corruptions'] = {'atomic_numbers': {}}
            with self.assertRaises(ValueError):
                validate_csp_config(config)

    def test_reject_fractional_counts_before_upstream_truncates(self):
        for value in (0, -1, 1.2, float('nan'), float('inf'), True, '2'):
            with self.assertRaises(ValueError):
                validate_composition({'Si': value})

    def test_preserve_explicit_cell_multiplicity(self):
        self.assertEqual(validate_composition({'Si': 8}), {'Si': 8})
        self.assertEqual(validate_composition({'Sr': 2, 'Ti': 2, 'O': 6}), {'Sr': 2, 'Ti': 2, 'O': 6})
        with self.assertRaises(ValueError):
            validate_composition({'Si': 24})
        with self.assertRaises(ValueError):
            validate_composition({})

    def test_sampling_rejects_species_update_or_shortened_schedule(self):
        config = {'sampler_partial': dict(N=1000, n_steps_corrector=1, guidance_scale=0,
                   predictor_partials={'pos': {}, 'cell': {}}, corrector_partials={'pos': {}, 'cell': {}})}
        validate_sampling(config)
        for field in ('N', 'predictor_partials', 'corrector_partials'):
            modified = copy.deepcopy(config)
            if field == 'N':
                modified['sampler_partial'][field] = 20
            else:
                modified['sampler_partial'][field]['atomic_numbers'] = {}
            with self.assertRaises(ValueError):
                validate_sampling(modified)

    def test_hash_detects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint'
            path.write_bytes(b'publisher asset')
            before = digest(path)
            path.write_bytes(b'publisher asset changed')
            self.assertNotEqual(before, digest(path))


if __name__ == '__main__':
    unittest.main()
