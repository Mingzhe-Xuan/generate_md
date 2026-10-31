"""Recover exact official base tensors from cached adapter only if all hashes match."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import numpy as np
import torch
from prepare import BASE_SHA, sha

MANIFEST_SHA = '0fe6a13c6a18114967e5929bff8ac924fb7e0ec6a69ae1d48ba064de4592f85a'
CONDITIONAL_SHA = '864ebbfd360e0a8287287d290a552da6b6bd92d67418a6a40be419ed3acd5e7e'


def matching_tensors(state, manifest):
    selected = {}; mismatches = []
    for name, expected in manifest['tensors'].items():
        value = state.get(name)
        if value is None or list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']:
            mismatches.append(name); continue
        digest = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        if digest != expected['sha256']:
            mismatches.append(name); continue
        selected[name] = value
    return selected, mismatches


def assemble_delta(directory, out):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['origin_base_sha256'] != BASE_SHA:
        raise ValueError('Delta targets another base checkpoint.')
    temp = out.with_suffix('.tmp')
    with temp.open('wb') as f:
        for index, item in enumerate(manifest['chunks']):
            if item['name'] != f'part-{index:04d}.bin' or not re.fullmatch(r'part-\d{4}\.bin', item['name']):
                raise ValueError('Invalid/noncontiguous chunk path.')
            p = directory/item['name']
            if p.stat().st_size != item['size'] or sha(p) != item['sha256']:
                raise ValueError('Delta chunk incomplete or corrupted: '+item['name'])
            f.write(p.read_bytes())
    if sha(temp) != manifest['delta_sha256']:
        raise ValueError('Assembled delta payload hash differs.')
    temp.replace(out)
    return manifest


def main(root):
    manifest_path = root/'assets/base-tensor-manifest.json'
    if sha(manifest_path) != MANIFEST_SHA: raise ValueError('Official-base tensor manifest changed.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['origin_sha256'] != BASE_SHA: raise ValueError('Unexpected full official checkpoint identity.')
    config = root/'assets/base/config.yaml'
    if sha(config) != manifest['config_sha256']: raise ValueError('Official base config differs.')
    conditional = Path('results/mattergen/checkpoints/dft_band_gap/checkpoints/last.ckpt')
    if sha(conditional) != CONDITIONAL_SHA: raise ValueError('Cached conditional checkpoint identity differs.')
    checkpoint = torch.load(conditional, map_location='cpu', weights_only=False)
    delta_manifest = None
    if (root/'assets/base-delta-parts/manifest.json').exists():
        delta_manifest = assemble_delta(root/'assets/base-delta-parts', root/'assets/base-delta.npz')
        with np.load(root/'assets/base-delta.npz', allow_pickle=False) as z:
            if set(z.files) - set(manifest['tensors']):
                raise ValueError('Unexpected tensors in official-base delta.')
            for name in z.files:
                checkpoint['state_dict'][name] = torch.from_numpy(z[name].copy())
    selected, mismatches = matching_tensors(checkpoint['state_dict'], manifest)
    event = dict(passed=not mismatches, matched=len(selected), total=len(manifest['tensors']), mismatches=mismatches,
        official_base_full_checkpoint_sha256=BASE_SHA, manifest_sha256=MANIFEST_SHA,
        cached_adapter_checkpoint_sha256=CONDITIONAL_SHA, slurm_job_id=os.environ['SLURM_JOB_ID'])
    if delta_manifest is not None:
        event['delta_transfer'] = dict(tensors=delta_manifest['tensors'], chunks=len(delta_manifest['chunks']),
                                      sha256=delta_manifest['delta_sha256'])
    if not mismatches:
        out = root/'assets/base-verified'; (out/'checkpoints').mkdir(parents=True, exist_ok=False)
        shutil.copyfile(config, out/'config.yaml')
        torch.save(dict(state_dict=selected), out/'checkpoints/last.ckpt')
        event.update(derived_checkpoint='assets/base-verified/checkpoints/last.ckpt',
            derived_checkpoint_sha256=sha(out/'checkpoints/last.ckpt'),
            config_sha256=sha(out/'config.yaml'),
            interpretation='Every base state tensor is bitwise identical to the fully downloaded, SHA-verified official base checkpoint; cached matching tensors and verified official delta only, adapter-only tensors excluded.')
    previous = root/'assets/base-recovery.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        saved = root/f'assets/base-recovery-{old["slurm_job_id"]}.json'
        if not saved.exists(): shutil.copyfile(previous, saved)
    (root/'assets/base-recovery.json').write_text(json.dumps(event, indent=2)+'\n')
    print(json.dumps(event), flush=True)
    if mismatches: raise ValueError('Cached backbone differs from official base: transfer exact missing tensors or full base, never relabel it.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): p.error('Checkpoint extraction/checks require Slurm.')
    main(a.run.resolve())
