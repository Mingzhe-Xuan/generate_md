"""Release exactly the saved experiment chain after MatterGen verification."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from fetch import digest
from report import audit


def release(manifest,report_path):
    state=json.loads(Path(manifest).read_text());report_path=Path(report_path);record=json.loads(report_path.read_text())
    if record['denominator']!=128 or not record.get('verified_end_to_end'):raise ValueError('Full128 MatterGen verification required before restoration.')
    verified=audit(report_path.parent,record['reference_selection'],128)
    if verified['prediction_sha256']!=record['prediction_sha256'] or verified['sun_sha256']!=record['sun_sha256']:raise ValueError('MatterGen results changed.')
    root=Path(state['experiment_root']).resolve()
    if not root.is_relative_to((Path.cwd()/'results/smoacs').resolve()):raise ValueError('Paused experiment outside task root.')
    if digest(root/'alternating/checkpoint.pt')!=state['checkpoint_sha256']:raise ValueError('Paused checkpoint changed before restoration.')
    released=[]
    for identifier in state['held_jobs']:
        if not isinstance(identifier,int) or identifier<=0:raise ValueError('Invalid saved Slurm ID.')
        text=subprocess.check_output(['scontrol','show','job','-o',str(identifier)],text=True)
        owner=re.search(r'\bUserId=\S+\((\d+)\)',text)
        if owner is None or int(owner[1])!=os.getuid():raise ValueError('Saved job owner differs.')
        if 'JobName=full128-' not in text or 'WorkDir='+str(Path.cwd()) not in text:raise ValueError('Saved job is not this comparison.')
        if 'JobState=PENDING' in text:
            if not ('JobHeldUser' in text or 'held state' in text):raise ValueError('Pending saved job not held by user.')
            subprocess.run(['scontrol','release',str(identifier)],check=True);released.append(identifier)
        elif not any('JobState='+status in text for status in ('RUNNING','COMPLETING','COMPLETED')):
            raise ValueError('Saved job is not safely resumable: '+text)
    destination=report_path.parent/'resume-released.json'
    destination.write_text(json.dumps(dict(released_jobs=released,paused_manifest_sha256=digest(manifest),report_sha256=digest(report_path),
        checkpoint_sha256=state['checkpoint_sha256'],note='Release verified; actual checkpoint continuation must be checked in the resumed job.'),indent=2)+'\n')
    return released


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();print(json.dumps(dict(released=release(args.manifest,args.report))),flush=True)


if __name__=='__main__':main()
