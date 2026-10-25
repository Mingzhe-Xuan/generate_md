"""Create an immutable all128-parent experiment plan inside Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from selection import partition


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    root=args.run.resolve();source=Path('results/mattergen/gap2p5-task128-20261024/full128').resolve()
    inventory=source/'inventory-249/inventory.json';data=json.loads(inventory.read_text());ids=[r['candidate_id'] for r in data['candidates']]
    shards=partition(ids);digest=hashlib.sha256(inventory.read_bytes()).hexdigest()
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use task results.')
    if (root/'plan.json').exists():raise ValueError('Full experiment root already planned; preserve existing artifacts.')
    root.mkdir(parents=True,exist_ok=True)
    for i,parents in enumerate(shards):
        folder=root/'shards'/f'{i:02d}';folder.mkdir(parents=True,exist_ok=False)
        (folder/'parents.json').write_text(json.dumps(dict(parent_ids=parents,source_inventory_sha256=digest),indent=2)+'\n')
    plan=dict(parent_ids=ids,parent_count=128,N=10,amplitudes=[.1,.2,.3],shard_size=8,shard_count=16,
        CF_steps=20,CF_proposal_cap_A=.6,source=str(source),source_inventory_sha256=digest,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        candidate_B_attempts=128*11*3*2,selection='Legacy force+CF_Ef+valid and force+CF_Ef+separation diagnostic; CF ranking only, ALIGNN held out.',
        novelty='Both best-then-classify and best-among-novel; strict and looser no-rescale criteria.',
        masks='All parents optimized, including original nonneutral parents. No eligibility exclusion in generation.')
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(root/'SKIP_SUN').touch()
    print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
