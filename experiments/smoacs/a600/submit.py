"""Submit two GPU A600B controls and one paired comparison without duplicates."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.absolute()
    if not root.resolve().is_relative_to((Path.cwd()/'results/smoacs').resolve()):raise ValueError('Use task results/smoacs.')
    registry=root.with_name(root.name+'-jobs.json');registry.parent.mkdir(parents=True,exist_ok=True)
    if registry.exists():jobs=json.loads(registry.read_text())
    else:
        if root.exists():raise ValueError('Fresh experiment required.')
        jobs=dict(root=str(root))
    if jobs['root']!=str(root):raise ValueError('Registry differs.')
    script=Path(__file__).resolve().parent
    def submit(key,file,env,dependency=None):
        if key in jobs:return
        cmd=['sbatch','--parsable']+(['--dependency=afterok:'+dependency] if dependency else [])+[str(script/file)]
        result=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
        if not result.isdigit():raise ValueError('Invalid job ID.')
        jobs[key]=int(result);registry.write_text(json.dumps(jobs,indent=2)+'\n')
    for name,schedule in [('single','per_phase'),('restart200','restart200')]:
        env=dict(os.environ,SMOACS_A600_RUN=str(root/name),SMOACS_A600_SCHEDULE=schedule)
        submit(name,'run.sbatch',env);submit(name+'_evaluation','evaluate.sbatch',env,str(jobs[name]))
    submit('comparison','report.sbatch',dict(os.environ,SMOACS_A600_ROOT=str(root)),
           str(jobs['single_evaluation'])+':'+str(jobs['restart200_evaluation']))
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
