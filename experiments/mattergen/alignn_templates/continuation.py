"""Preserve completed ALIGNN65 results and archive only verified pre-A failures."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inference_signature(source):
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=='predict':
            node.body=[item for item in node.body if not (isinstance(item,ast.If) and isinstance(item.test,ast.Name) and item.test.id=='native_control')]
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def pre_A_only(path):
    return not any((path/name).exists() for name in ('A-trace.jsonl','results.json','checkpoint.pt','complete.json')) and not any(path.glob('cycle-*'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--previous-job',required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Validate/archive/continue through Slurm.')
    root=args.run.resolve()
    allowed=(Path.cwd()/'results/mattergen').resolve()
    if not root.is_relative_to(allowed):
        raise ValueError('Continuation root outside task results.')
    status=load(Path('results/smoacs/status')/('job-'+args.previous_job+'.json'))
    if status['job_id']!=args.previous_job or status['exit_code']==0:
        raise ValueError('Previous job must have a confirmed failed terminal record.')
    preserved=root/'alignn200/neutral'
    protocol=load(preserved/'protocol.json')
    data=load(preserved/'results.json')
    if not load(preserved/'complete.json')['completed'] or not (preserved/'evaluation-cycle1/cross-prediction.complete').exists():
        raise ValueError('Only complete evaluated neutral65 results may be reused.')
    if len(protocol['ids'])!=65 or protocol.get('optimization_predictor')!='ALIGNN' or protocol['steps_per_A']!=200:
        raise ValueError('Unexpected completed scientific protocol.')
    if any(r['a_updates']!=200 for r in data['cycles'][0]['candidates']):
        raise ValueError('Completed cohort does not have200actualupdates each.')
    if protocol['template_initialization']['source_inventory_sha256']!=digest(args.source/'inventory-249/inventory.json'):
        raise ValueError('Raw input cohort changed.')
    old=protocol['source_commit']
    package=Path('experiments/mattergen/alignn_templates')
    comparisons={}
    for file in ('geometry.py','probe.py','device_control.py','run.py','model.py'):
        name=(package/file).as_posix()
        before=subprocess.check_output(['git','show',old+':'+name],text=True)
        after=Path(name).read_text()
        same=inference_signature(before)==inference_signature(after) if file=='model.py' else before==after
        if not same:
            raise ValueError('Optimization or geometry code changed; do not mix trajectories: '+name)
        comparisons[file]=dict(identical_optimization_code=True,control_only_difference=file=='model.py')
    adam='experiments/smoacs/adam_cycles/run.py'
    if subprocess.check_output(['git','show',old+':'+adam],text=True)!=Path(adam).read_text():
        raise ValueError('Shared optimization controller changed.')
    protected={str(p.relative_to(root)):digest(p) for p in preserved.rglob('*') if p.is_file() and p.suffix not in ('.pth','.zip','.npy','.npz') and 'loader-tmp' not in p.parts}
    failed=root/'alignn200/unmasked'
    archived=None
    if failed.exists() and not (failed/'complete.json').exists():
        if not pre_A_only(failed):
            raise ValueError('Partial optimization exists: refusing to archive/replay it.')
        destination=root/'failed-initialization'/('unmasked-before-'+os.environ['SLURM_JOB_ID'])
        if not destination.resolve().is_relative_to(root) or destination.exists():
            raise ValueError('Unsafe or occupied archive path.')
        destination.parent.mkdir(exist_ok=True)
        failed.rename(destination)
        archived=str(destination)
    event=dict(previous_job=status,current_job=os.environ['SLURM_JOB_ID'],preserved_source_commit=old,
        current_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        preserved='alignn200/neutral',protected_sha256=protected,optimization_code_comparisons=comparisons,
        archived_pre_A_failure=archived,note='Only invariant-feature control implementation changed; no completed scientific trajectory is replayed.')
    output=root/('continuation-'+os.environ['SLURM_JOB_ID']+'.json')
    output.write_text(json.dumps(event,indent=2)+'\n')
    print(json.dumps(dict(preserved65=True,archived_pre_A_failure=archived,protected_files=len(protected))),flush=True)


if __name__=='__main__':
    main()
