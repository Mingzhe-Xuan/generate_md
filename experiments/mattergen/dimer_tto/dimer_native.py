"""CPU physical feasibility with immutable GPU CF gradients on first4 parents."""
import argparse
from contextlib import redirect_stdout
import json
import inspect
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from dimer_policy import POLICY,PARENTS,ARMS,directions,ledger
from dimer_engine import Meter,point,directional_curvature,activate
from ase.mep import MinModeAtoms

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'annealed_bias'))
from anneal_experiment import source_proof
from anneal_shared import load,save,sha


def code_hashes():
    files={n:sha(HERE/n) for n in ('dimer_policy.py','dimer_engine.py','dimer_native.py')}
    smoacs=HERE.parents[1]/'smoacs'
    files.update({n:sha(smoacs/n) for n in ('alignn_checkpoint.py','postprocess/mattersim_support.py','postprocess/sources.json')})
    files['cached_proof_loader']=sha(HERE.parent/'annealed_bias/anneal_experiment.py')
    return files


def validate_inventory(data):
    expected=ledger(PARENTS);actual={r['id']:r for r in data['slots']}
    if len(data['slots'])!=len(expected) or len(actual)!=len(expected) or set(actual)!={r['id'] for r in expected}:raise ValueError('Missing/duplicate24 slots.')
    for spec in expected:
        row=actual[spec['id']]
        if any(row[k]!=v for k,v in spec.items()):raise ValueError('Future quench specification changed.')
        if row['status']=='candidate_ready' and sha(row['input'])!=row['input_sha256']:raise ValueError('Saved exit input changed.')
    attempts=data['attempts']
    if len(attempts)!=12 or {(r['parent_id'],r['arm']) for r in attempts}!={(p,a) for p in PARENTS for a in ARMS}:raise ValueError('Missing/duplicate12 activations.')
    for attempt in attempts:
        result=attempt['result'];certified=result['status']=='certified_candidate';signs=result.get('exit_signs_ready',[])
        if not 0<=result['NFE']<=POLICY['NFE_cap_per_activation'] or result.get('steps',0)>POLICY['steps']:raise ValueError('Activation budget changed.')
        if len(signs)!=len(set(signs)) or any(s not in (-1,1) for s in signs) or (signs and not certified):raise ValueError('Invalid certified exit inventory.')
        if certified and not result['certificate']['passed']:raise ValueError('Candidate lacks saddle certificate.')
        for row in actual.values():
            if (row['parent_id'],row['arm'])==(attempt['parent_id'],attempt['arm']):
                if row['status']!=('candidate_ready' if row['exit_sign'] in signs else 'activation_failed'):raise ValueError('Activation and exit status differ.')
    if data['NFE']!=data['preflight_NFE']+sum(r['result']['NFE'] for r in attempts):raise ValueError('Actual calculator ledger changed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm CPU; no CF inference runs here.')
    import torch
    import ase
    from ase.io import read,write
    from mattersim.forcefield import MatterSimCalculator,potential
    smoacs=HERE.parents[1]/'smoacs';sys.path[:0]=[str(smoacs),str(smoacs/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import model_max_z,validate_atomic_numbers
    source=a.source.resolve();cache=a.cache.resolve();proof=source_proof(source,cache);root=a.run.resolve();root.mkdir(parents=True,exist_ok=False)
    if [c['parent_id'] for c in proof['cases'][:4]]!=list(PARENTS) or '5090' not in proof['CF']['device']:raise ValueError('Original4/GPU CF proof changed.')
    plan=dict(policy=POLICY,parent_ids=list(PARENTS),source=str(source),cache=str(cache),cache_sha256=sha(cache/'native.json'),implementation_sha256=code_hashes(),
        cached_source_hashes=proof['source_hashes'],cached_CF=proof['CF'],cached_CF_gradients_used=4,fresh_CF_model_evaluations=0,reused_native_proof_NFE=proof['NFE'],
        role='CPU saddle physics with cachedGPUCF gradients; no final property efficacy.')
    save(root/'plan.json',plan)
    checkpoint=Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve();expected=load(smoacs/'postprocess/sources.json')['mattersim']['checkpoint_sha256']
    if sha(checkpoint)!=expected:raise ValueError('Potential checkpoint changed.')
    old=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected]);torch.set_num_threads(2)
    try:
        with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(checkpoint),device='cpu',load_training_state=False)
    finally:potential.torch=old
    if any(x.is_cuda for x in calc.potential.model.parameters()):raise RuntimeError('CPU dimer touched GPU.')
    calc.potential.model.eval();maximum=model_max_z(calc.potential.model);meter=Meter(calc);native=dict(passed=False,cases=[],checkpoint_sha256=expected,device='cpu',
        packages=dict(torch=torch.__version__,ASE=ase.__version__,numpy=np.__version__),
        dependency_sha256=dict(ASE_dimer=sha(inspect.getfile(MinModeAtoms)),MatterSimCalculator=sha(inspect.getfile(MatterSimCalculator))),
        plan_sha256=sha(root/'plan.json'));references={};modes={};energies={}
    try:
        for case in proof['cases'][:4]:
            parent=case['parent_id'];reference=read(case['seed_input'],format='vasp');reference.set_constraint(None);validate_atomic_numbers(reference.numbers,maximum)
            if sha(case['seed_input'])!=case['seed_sha256']:raise ValueError('Cached pose changed.')
            oldpoint=case['physics_seed']['record'];meter.bind(reference,oldpoint['energy_eV']);energy,force=point(reference,calc);stress=reference.copy();stress.calc=calc;stress=stress.get_stress(voigt=False,apply_constraint=False)
            errors=dict(energy_per_atom=abs(energy-oldpoint['energy_eV'])/len(reference),force=float(np.abs(force-oldpoint['forces_eV_A']).max()),stress=float(np.abs(stress-oldpoint['stress_eV_A3']).max()))
            if max(errors.values())>1e-4:raise ValueError('CPU versus cachedGPU physical control failed: '+str(errors))
            d=directions(case['CF_seed']['gradient'],len(reference),int(Path(parent).stem.split('-')[1]));curvature=directional_curvature(reference,calc,d['descent']) if d is not None else dict(passed=True,stationary_CF=True)
            native['cases'].append(dict(parent_id=parent,input_sha256=case['seed_sha256'],source_errors=errors,curvature=curvature,passed=curvature['passed']))
            save(root/'native.json',native)
            if not curvature['passed']:raise ValueError('Physical curvature two-scale gate failed.')
            references[parent]=reference;modes[parent]=d;energies[parent]=energy
        native.update(passed=True,NFE=meter.calls);save(root/'native.json',native);attempts=[];slots={r['id']:dict(r) for r in ledger(PARENTS)}
        for parent in PARENTS:
            for arm in ARMS:
                if modes[parent] is None:result,exits=dict(status='stationary_CF',NFE=0),[]
                else:result,exits=activate(references[parent],meter,modes[parent][arm],energies[parent])
                attempt=dict(parent_id=parent,arm=arm,initial_mode=modes[parent][arm] if modes[parent] is not None else None,result=result);attempts.append(attempt)
                folder=root/'activations'/Path(parent).stem/arm;folder.mkdir(parents=True,exist_ok=False);save(folder/'result.json',attempt)
                for spec in ledger([parent]):
                    if spec['arm']==arm:slots[spec['id']].update(status='activation_failed',activation_status=result['status'])
                for sign,atoms in exits:
                    id=f'{Path(parent).stem}-{arm}-exit{sign:+d}';path=folder/f'exit{sign:+d}.vasp';write(path,atoms,format='vasp',direct=True,sort=False)
                    slots[id].update(status='candidate_ready',input=str(path),input_sha256=sha(path))
                data=dict(attempts=attempts,slots=list(slots.values()),NFE=meter.calls,preflight_NFE=native['NFE'],goal_completion_proven=False)
                save(root/'results.json',data);print(json.dumps(dict(parent=parent,arm=arm,status=result['status'],NFE=result['NFE'],ready=len(exits))),flush=True)
        validate_inventory(data);save(root/'complete.json',dict(completed=True,plan_sha256=sha(root/'plan.json'),native_sha256=sha(root/'native.json'),results_sha256=sha(root/'results.json'),
            ready_slots=sum(s['status']=='candidate_ready' for s in slots.values()),goal_completion_proven=False,final_crystal_properties_evaluated=False))
    except Exception as error:
        save(root/'failed.json',dict(error=repr(error),traceback=traceback.format_exc(),NFE=meter.calls));raise


if __name__=='__main__':main()
