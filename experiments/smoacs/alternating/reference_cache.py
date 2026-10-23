"""Reuse hash-matched reference phases and obtain benchmark controls in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Submit through Slurm.')
    manifest=json.loads((args.selection/'mattersim-inventory.json').read_text())
    output=args.selection/'mattersim-relaxation';output.mkdir(exist_ok=False)
    old=Path('results/smoacs/mp-public-reference/selected-261/mattersim-relaxation')
    root=Path(__file__).resolve().parents[1]
    checkpoint=json.loads((root/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    reused=0
    for candidate in manifest['candidates']:
        prior=old/Path(candidate['candidate_id']).stem
        if not (prior/'result.json').exists():continue
        record=json.loads((prior/'result.json').read_text())
        if record['input_sha256']!=candidate['sha256'] or record['checkpoint_sha256']!=checkpoint:continue
        if record['status']=='converged' and hashlib.sha256((prior/'relaxed.vasp').read_bytes()).hexdigest()!=record['relaxed_sha256']:
            raise ValueError('Corrupt reference cache.')
        destination=output/prior.name;destination.mkdir()
        shutil.copy2(prior/'result.json',destination/'result.json')
        if (prior/'relaxed.vasp').exists():shutil.copy2(prior/'relaxed.vasp',destination/'relaxed.vasp')
        reused+=1
    def controls():
        return sum(c['num_atoms']<=64 and (output/Path(c['candidate_id']).stem/'result.json').exists()
            and json.loads((output/Path(c['candidate_id']).stem/'result.json').read_text())['status']=='converged' for c in manifest['candidates'])
    if controls()<16:
        warm=[c['candidate_id'] for c in sorted(manifest['candidates'],key=lambda c:c['num_atoms'])
              if c['num_atoms']<=64 and not (output/Path(c['candidate_id']).stem/'result.json').exists()][:64]
        if warm:
            subprocess.run([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(root/'postprocess/relax.py'),
                '--inventory',str(args.selection/'mattersim-inventory.json'),'--output',str(output),
                '--checkpoint','results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth','--candidate-id',*warm],check=True)
    if controls()<16:raise ValueError('Insufficient converged reference benchmark controls.')
    (args.selection/'cache-reuse.json').write_text(json.dumps(dict(reused=reused,benchmark_controls=controls()))+'\n')


if __name__=='__main__':main()
