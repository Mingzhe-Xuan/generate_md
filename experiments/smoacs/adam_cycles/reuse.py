"""Seed small serial reference relaxations from verified existing results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selection',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm for batch processing.')
    manifest=json.loads((args.selection/'mattersim-inventory.json').read_text())
    output=args.selection/'mattersim-relaxation';output.mkdir(exist_ok=False)
    old=Path('results/smoacs/mp-public-reference/selected-261/mattersim-relaxation')
    model=json.loads((Path(__file__).resolve().parents[1]/'postprocess/sources.json').read_text())['mattersim']['checkpoint_sha256']
    reused=0
    for candidate in manifest['candidates']:
        prior=old/Path(candidate['candidate_id']).stem
        if not (prior/'result.json').exists():continue
        record=json.loads((prior/'result.json').read_text())
        if record['input_sha256']!=candidate['sha256'] or record['checkpoint_sha256']!=model:continue
        if record['status']=='converged' and hashlib.sha256((prior/'relaxed.vasp').read_bytes()).hexdigest()!=record['relaxed_sha256']:
            raise ValueError('Reference cache geometry hash mismatch.')
        destination=output/prior.name;destination.mkdir();shutil.copy2(prior/'result.json',destination/'result.json')
        if (prior/'relaxed.vasp').exists():shutil.copy2(prior/'relaxed.vasp',destination/'relaxed.vasp')
        reused+=1
    (args.selection/'cache-reuse.json').write_text(json.dumps(dict(reused=reused,mode='serial, no parallel benchmark needed'))+'\n')


if __name__=='__main__':main()
