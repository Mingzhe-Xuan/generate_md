"""Configuration and composition checks independent of GPU/model imports."""
import hashlib
import math
from pathlib import Path


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def byte_ranges(size, count=8):
    if size < 1 or count < 1 or count > size:
        raise ValueError('Invalid download partition.')
    return [(i * size // count, (i + 1) * size // count - 1) for i in range(count)]


def assemble_parts(root, size):
    """Called under Slurm; preserve the interrupted monolithic download."""
    import shutil
    ranges = byte_ranges(size)
    files = [root / 'parts' / f'part-{i:02d}' for i in range(len(ranges))]
    if any(p.stat().st_size != end - start + 1 for p, (start, end) in zip(files, ranges)):
        raise ValueError('Incomplete ranged checkpoint download.')
    target = root / 'checkpoints/last.ckpt.assembled'
    if target.exists():
        raise ValueError('Refuse to overwrite an existing assembled checkpoint.')
    with target.open('xb') as output:
        for part in files:
            with part.open('rb') as stream:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
    return target


def validate_composition(composition):
    if not isinstance(composition, dict) or not composition:
        raise ValueError('Provide explicit element counts for the simulation cell.')
    result = {}
    for element, count in composition.items():
        if not isinstance(element, str) or not element:
            raise ValueError('Element names must be nonempty strings.')
        if isinstance(count, bool) or not isinstance(count, (int, float)):
            raise ValueError('Atom counts must be positive integers.')
        if not math.isfinite(count) or count <= 0 or int(count) != count:
            raise ValueError('Fractional/zero atom counts are not supported.')
        result[element] = int(count)
    if sum(result.values()) > 20:
        raise ValueError('This checkpoint smoke is restricted to its MP20 atom-count domain.')
    return result


def validate_csp_config(config):
    module = config['lightning_module']['diffusion_module']
    loss = module['loss_fn']
    corruption = module['corruption']
    if loss.get('include_atomic_numbers', True) or loss.get('weights', {}).get('atomic_numbers', 0):
        raise ValueError('Checkpoint configuration includes atom-type training loss.')
    if corruption.get('discrete_corruptions') or set(corruption.get('sdes', {})) != {'pos', 'cell'}:
        raise ValueError('CSP must corrupt only positions and lattice.')
    if module['model'].get('property_embeddings') or module['model'].get('property_embeddings_adapt'):
        raise ValueError('Expected composition-only CSP, without property conditioning.')


def validate_sampling(config):
    sampler = config['sampler_partial']
    if sampler['N'] != 1000 or sampler['n_steps_corrector'] != 1 or sampler['guidance_scale'] != 0:
        raise ValueError('Use native 1000-step, one-corrector CSP with zero guidance.')
    for field in ('predictor_partials', 'corrector_partials'):
        if set(sampler[field]) != {'pos', 'cell'}:
            raise ValueError('Sampling must update only positions and lattice.')
