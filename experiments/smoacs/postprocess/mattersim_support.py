"""Validate the loaded potential's element encoder before touching CUDA."""


class UnsupportedSpeciesError(ValueError):
    def __init__(self, numbers, maximum):
        self.numbers = sorted(set(int(z) for z in numbers))
        self.maximum = int(maximum)
        super().__init__(f'Atomic numbers {self.numbers} outside MatterSim encoder range1..{self.maximum}.')


def model_max_z(model):
    maximum = int(model.max_z)
    if not 1 <= maximum <= 118:
        raise ValueError('Invalid loaded MatterSim element encoder size.')
    return maximum


def validate_atomic_numbers(numbers, maximum):
    unsupported = [int(z) for z in numbers if not 1 <= int(z) <= maximum]
    if unsupported:
        raise UnsupportedSpeciesError(unsupported, maximum)


def error_metadata(error):
    if isinstance(error, UnsupportedSpeciesError):
        return dict(error_type='unsupported_species', unsupported_atomic_numbers=error.numbers,
                    model_max_z=error.maximum, infrastructure_failure=False, retryable=False)
    text = str(error).lower()
    if any(token in text for token in ('device-side assert', 'illegal memory access', 'misaligned address', 'poisoned cuda context')):
        return dict(error_type='cuda_context_error', infrastructure_failure=True, retryable=True)
    if 'out of memory' in text and ('cuda' in text or 'gpu' in text):
        return dict(error_type='cuda_out_of_memory', infrastructure_failure=True, retryable=True)
    if 'fixsymmetry' in text or 'deformation gradient' in text:
        return dict(error_type='symmetry_guard', infrastructure_failure=False, retryable=False)
    return dict(error_type='relaxation_error', infrastructure_failure=False, retryable=False)


def needs_isolated_retry(record):
    if record.get('status') != 'failed' or record.get('isolated_retry'):
        return False
    return record.get('retryable', False) or error_metadata(record.get('error', ''))['retryable']
