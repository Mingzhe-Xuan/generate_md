"""Submit a validated alternating run and dependent evaluation jobs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess


def checked_pilot(path):
    complete=json.loads((path/'complete.json').read_text())
    data=json.loads((path/'results.json').read_text())
    protocol=data['protocol']
    if not complete['completed'] or not complete['pilot'] or protocol['initial_control_error']>1e-3:
        raise ValueError('Pilot initialization/completion gate failed.')
    if protocol['b_maxstep']!=.05 or protocol['b_max_steps']!=1600:
        raise ValueError('Pilot numerical profile differs from production.')
    history=[json.loads(line) for line in (path/'history.jsonl').read_text().splitlines() if line]
    good=[r for r in history if r['b_result']['status']=='converged'
          and r['b_result']['max_filter_force']<=.01 and r['feedback_gram_error']<=1e-3]
    if not good or max(r['a_steps'] for r in data['candidates'])<2:
        raise ValueError('Pilot has not exercised converged B feedback and a subsequent A update.')
    return dict(initial_control_error=protocol['initial_control_error'],converged_feedback_records=len(good),
                terminal_statuses=dict(Counter(r['status'] for r in data['candidates'])),
                note='Integration pass is not a claim that every pilot candidate jointly converged.')


def submit_job(script,env,dependency=None,walltime=None):
    command=['sbatch','--parsable']
    if dependency:command+=['--dependency=afterok:'+str(dependency)]
    if walltime:command+=['--time='+walltime]
    command.append(str(script))
    text=subprocess.check_output(command,text=True,env=env).strip().split(';')[0]
    if not text.isdigit():raise ValueError('Unrecognized sbatch response: '+text)
    return int(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=args.output.absolute();allowed=(Path.cwd()/'results/smoacs').resolve()
    if not output.resolve().is_relative_to(allowed):raise ValueError('Output must stay under task results/smoacs.')
    proof=checked_pilot(args.pilot_run)
    registry=output.with_name(output.name+'-jobs.json')
    if registry.exists():
        jobs=json.loads(registry.read_text())
        if jobs['run']!=str(output):raise ValueError('Registry path differs.')
    else:
        if output.exists():raise ValueError('Refuse to overwrite an existing experiment.')
        jobs=dict(run=str(output),pilot=str(args.pilot_run),pilot_proof=proof)
    env=dict(os.environ,SMOACS_AB_OUTPUT=str(output),SMOACS_AB_RUN=str(output),SMOACS_AB_PILOT='0')
    root=Path(__file__).resolve().parent
    for key,script,dependency,walltime in (
        ('optimization','run.sbatch',None,'24:00:00'),
        ('properties','evaluate.sbatch','optimization',None),
        ('SUN','references.sbatch','properties',None)):
        if key not in jobs:
            jobs[key]=submit_job(root/script,env,jobs[dependency] if dependency else None,walltime)
            registry.write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':main()
