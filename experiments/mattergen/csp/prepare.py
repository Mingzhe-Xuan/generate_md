"""Slurm-only verification of downloaded author assets; never edits upstream code."""
import json
import os
from pathlib import Path
import shutil

from validation import assemble_parts, digest, validate_csp_config


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use Slurm for checkpoint verification.')
    import yaml
    here = Path(__file__).resolve().parent
    spec = json.loads((here / 'sources.json').read_text())
    root = Path('results/mattergen/csp-assets').resolve()
    candidate = root / 'checkpoints/last.ckpt.partial'
    final = candidate.with_name('last.ckpt')
    if final.exists():
        candidate = final
    elif (root / 'parts/download-complete.json').exists():
        candidate = assemble_parts(root, spec['checkpoint_size_bytes'])
    if candidate.stat().st_size != spec['checkpoint_size_bytes']:
        raise ValueError('Incomplete CSP checkpoint download.')
    if digest(candidate, 'md5') != spec['checkpoint_md5']:
        raise ValueError('CSP checkpoint differs from publisher checksum.')
    checkpoint_sha = digest(candidate)
    config = here / 'model_config.yaml'
    if digest(config) != spec['config_sha256']:
        raise ValueError('Author config hash mismatch.')
    validate_csp_config(yaml.safe_load(config.read_text()))
    if candidate != final:
        candidate.rename(final)
    target = root / 'config.yaml'
    if target.exists() and target.read_bytes() != config.read_bytes():
        raise ValueError('Refuse to overwrite an existing different CSP config.')
    shutil.copyfile(config, target)
    record = dict(source=spec, checkpoint_sha256=checkpoint_sha,
                  config_sha256=digest(target), job_id=os.environ['SLURM_JOB_ID'])
    (root / 'verified.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
