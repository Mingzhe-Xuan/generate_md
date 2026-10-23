"""Submit smoke, full128 task, and SUN verification with conditional job restoration."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--setup-job',required=True,type=int);p.add_argument('--paused-manifest',required=True,type=Path)
    args=p.parse_args();root=args.output.absolute();manifest=args.paused_manifest.absolute()
    if not root.resolve().is_relative_to((Path.cwd()/'results/mattergen').resolve()):raise ValueError('Use task MatterGen results root.')
    if not manifest.is_file():raise ValueError('Missing saved paused-job manifest.')
    registry=root.with_name(root.name+'-jobs.json')
    registry.parent.mkdir(parents=True,exist_ok=True)
    if registry.exists():jobs=json.loads(registry.read_text())
    else:
        if root.exists():raise ValueError('Fresh MatterGen run required.')
        jobs=dict(root=str(root),setup=args.setup_job,paused_manifest=str(manifest))
    if (jobs['root'],jobs['setup'],jobs['paused_manifest'])!=(str(root),args.setup_job,str(manifest)):raise ValueError('Registry differs.')
    script=Path(__file__).resolve().parent
    def submit(key,file,env,dependency):
        if key in jobs:return
        value=subprocess.check_output(['sbatch','--parsable','--dependency=afterok:'+str(dependency),str(script/file)],text=True,env=env).strip().split(';')[0]
        if not value.isdigit():raise ValueError('Invalid Slurm ID.')
        jobs[key]=int(value);registry.write_text(json.dumps(jobs,indent=2)+'\n')
    base=dict(os.environ,MATTERGEN_PAUSED_MANIFEST=str(manifest))
    submit('smoke','run.sbatch',dict(base,MATTERGEN_RUN=str(root/'smoke'),MATTERGEN_COUNT='4',MATTERGEN_BATCH='4'),args.setup_job)
    submit('full128','run.sbatch',dict(base,MATTERGEN_RUN=str(root/'full128'),MATTERGEN_COUNT='128',MATTERGEN_BATCH='16'),jobs['smoke'])
    submit('SUN_and_restore','references.sbatch',dict(base,MATTERGEN_RUN=str(root/'full128')),jobs['full128'])
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
