"""Idempotent Slurm chain: three full GPU arms, frozen evaluations, shared SUN."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ARMS={'single':1,'alternating':3,'restart200':1}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output.absolute()
    if not root.resolve().is_relative_to((Path.cwd()/'results/smoacs').resolve()):raise ValueError('Use task results/smoacs.')
    registry=root.with_name(root.name+'-jobs.json');registry.parent.mkdir(parents=True,exist_ok=True)
    if registry.exists():jobs=json.loads(registry.read_text())
    else:
        if root.exists():raise ValueError('Fresh experiment required.')
        jobs=dict(root=str(root))
    if jobs['root']!=str(root):raise ValueError('Registry root differs.')
    script=Path(__file__).resolve().parent
    def submit(key,file,env,dependency=None):
        if key in jobs:return
        cmd=['sbatch','--parsable']+(['--dependency=afterok:'+dependency] if dependency else [])+[str(script/file)]
        value=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
        if not value.isdigit():raise ValueError('Invalid Slurm job ID.')
        jobs[key]=int(value);registry.write_text(json.dumps(jobs,indent=2)+'\n')
    for arm,cycle in ARMS.items():
        env=dict(os.environ,SMOACS_FULL_ROOT=str(root),SMOACS_FULL_RUN=str(root/arm),SMOACS_FULL_ARM=arm,SMOACS_FULL_CYCLE=str(cycle))
        submit(arm,'run.sbatch',env)
        submit(arm+'_evaluation','evaluate.sbatch',env,str(jobs[arm]))
    submit('SUN_and_report','references.sbatch',dict(os.environ,SMOACS_FULL_ROOT=str(root)),
           ':'.join(str(jobs[arm+'_evaluation']) for arm in ARMS))
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
