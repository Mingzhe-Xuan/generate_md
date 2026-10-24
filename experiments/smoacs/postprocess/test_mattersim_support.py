from types import SimpleNamespace
import unittest

from mattersim_support import (UnsupportedSpeciesError, error_metadata, model_max_z,
                              needs_isolated_retry, validate_atomic_numbers)


class SupportTests(unittest.TestCase):
    def test_exact_loaded_encoder_boundary_before_gpu(self):
        maximum = model_max_z(SimpleNamespace(max_z=94))
        validate_atomic_numbers([1, 8, 94], maximum)
        with self.assertRaises(UnsupportedSpeciesError):
            validate_atomic_numbers([95], maximum)
        with self.assertRaises(UnsupportedSpeciesError) as error:
            validate_atomic_numbers([97, 98, 97], maximum)
        self.assertEqual(error.exception.numbers, [97, 98])
        self.assertEqual(error_metadata(error.exception)['error_type'], 'unsupported_species')
        self.assertFalse(error_metadata(error.exception)['retryable'])
        with self.assertRaises(UnsupportedSpeciesError):
            validate_atomic_numbers([0], maximum)
        validate_atomic_numbers([98], model_max_z(SimpleNamespace(max_z=98)))

    def test_context_failure_is_not_a_physical_failure(self):
        failure = error_metadata(RuntimeError('CUDA error: device-side assert triggered'))
        self.assertTrue(failure['infrastructure_failure'])
        self.assertTrue(failure['retryable'])
        self.assertEqual(error_metadata('CUDA out of memory')['error_type'], 'cuda_out_of_memory')
        self.assertEqual(error_metadata('Poisoned CUDA context after earlier case')['error_type'], 'cuda_context_error')
        self.assertFalse(error_metadata('FixSymmetry deformation gradient exceeds0.25')['retryable'])

    def test_repair_only_engine_errors_once(self):
        self.assertTrue(needs_isolated_retry(dict(status='failed', error='CUDA error: device-side assert triggered')))
        self.assertFalse(needs_isolated_retry(dict(status='failed', error='FixSymmetry guard')))
        self.assertFalse(needs_isolated_retry(dict(status='unconverged', error='CUDA error: device-side assert triggered')))
        self.assertFalse(needs_isolated_retry(dict(status='failed', error='CUDA error: device-side assert triggered', isolated_retry=True)))


if __name__ == '__main__':
    unittest.main()
