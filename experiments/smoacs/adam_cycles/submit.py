"""Idempotently submit GPU Adam cycles and their property/SUN evaluations."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.absolute()
    if not output.resolve().is_relative_to((Path.cwd()/'results/smoacs').resolve()):raise ValueError('Output must stay in task results/smoacs.')
    registry=output.with_name(output.name+'-jobs.json');registry.parent.mkdir(parents=True,exist_ok=True)
    if registry.exists():
        jobs=json.loads(registry.read_text())
        if jobs['run']!=str(output):raise ValueError('Registry points to another run.')
    else:
        if output.exists():raise ValueError('Refusing to overwrite an existing experiment.')
        jobs=dict(run=str(output),method='GPU Adam200 cosine x3 AB, cycle1 versus3')
    env=dict(os.environ,SMOACS_ADAM_RUN=str(output));root=Path(__file__).resolve().parent
    for key,script,prior in [('optimization','run.sbatch',None),('properties','evaluate.sbatch','optimization'),('SUN','references.sbatch','properties')]:
        if key in jobs:continue
        command=['sbatch','--parsable']
        if prior:command.append('--dependency=afterok:'+str(jobs[prior]))
        command.append(str(root/script))
        result=subprocess.check_output(command,env=env,text=True).strip().split(';')[0]
        if not result.isdigit():raise ValueError('Unrecognized Slurm response: '+result)
        jobs[key]=int(result);registry.write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
