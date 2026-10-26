"""CPU tighter-force re-quenching of every material plain-CF endpoint and its parent."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from feedback import validate_source
from probe import load,save,sha
from attractors import validate_control
from proposals import mutate

TIGHT_POLICY=dict(fmax=.001,maxstep=.05,steps=1600,preserve_symmetry=False,
    trials=4,perturb_atomic_RMS_A=.02,perturb_strain=.003,seed=20261028,
    selection='ALL plain-descent physical gains>=.1eV beyond BOTH directB and seed-onlyB',
    role='Tighter-force paired diagnostic; does not replace frozen .01-force experiment or certify DFT/phonon stability')


def material_indices(plan,data,report):
    if len(plan['parent_ids'])!=16:raise ValueError('Expected original frozen16.')
    rows=report['arms']['plain-descent']['cases']
    if [r['parent_id'] for r in rows]!=plan['parent_ids']:raise ValueError('Material-case coverage changed.')
    selected=[]
    for i,(case,row) in enumerate(zip(plan['cases'],rows)):
        old=abs(case['predictions']['after']['crystalformer_gap']-2.5)
        seed=abs(data['selected']['seed'][i]['gap']-2.5)
        new=abs(data['selected']['plain-descent'][i]['gap']-2.5)
        if min(old,seed)-new>=.1 and row['physical_eligible']:selected.append(i)
    return selected


def perturb(reference,parent_index,trial):
    if trial==0:
        atoms=reference.copy();atoms.set_constraint(None);return atoms
    return mutate(reference,reference,np.random.default_rng([TIGHT_POLICY['seed'],parent_index,trial]),.02,strain=.003)


def tight_eligible(record):
    return bool(record.get('status')=='converged' and record.get('converged') is True and record.get('separated') is True
        and all(record.get(k) is not None and np.isfinite(record[k]) and record[k]<=TIGHT_POLICY['fmax'] for k in ('max_raw_atomic_force','max_filter_force')))


def validate_trials(parents,records):
    expected={(p,s,t) for p in parents for s in ('direct','selected') for t in range(4)}
    if len(records)!=len(expected) or {(r['parent_id'],r['stage'],r['trial']) for r in records}!=expected:
        raise ValueError('Missing/duplicate paired diagnostic trials.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('CPU physics must use Slurm.')
    import torch
    import ase.constraints
    from ase.filters import Filter,ExpCellFilter
    if not hasattr(ase.constraints,'Filter'):ase.constraints.Filter=Filter
    from ase.io import read,write
    from ase.neighborlist import neighbor_list
    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from mattersim.forcefield import MatterSimCalculator,potential
    source=a.source.resolve();root=a.run.resolve();plan=load(source/'plan.json');data=load(source/'direction-results.json');report=load(source/'report.json')
    validate_source(plan,data,load(source/'native-probe.json'));indices=material_indices(plan,data,report)
    if not indices:raise ValueError('No material CF endpoints to diagnose.')
    root.mkdir(parents=True,exist_ok=False);parents=[plan['parent_ids'][i] for i in indices]
    save(root/'tight-plan.json',dict(policy=TIGHT_POLICY,source=str(source),parent_ids=parents,indices=indices,
        source_hashes={n:sha(source/n) for n in ('plan.json','direction-results.json','report.json','native-probe.json')},
        mechanics_sha256=sha(__file__),evaluation_sha256=sha(Path(__file__).with_name('tight_evaluation.py'))))
    smoacs=Path(__file__).resolve().parents[2]/'smoacs';sys.path[:0]=[str(smoacs),str(smoacs/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import validate_atomic_numbers,model_max_z
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    checkpoint=Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve();expected=load(smoacs/'postprocess/sources.json')['mattersim']['checkpoint_sha256']
    if sha(checkpoint)!=expected:raise ValueError('Potential checkpoint changed.')
    original=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected]);torch.set_num_threads(2)
    try:
        with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(checkpoint),device='cpu',load_training_state=False)
    finally:potential.torch=original
    if any(v.is_cuda for v in calc.potential.model.parameters()):raise RuntimeError('CPU diagnostic accessed GPU.')
    calc.potential.model.eval();maximum=model_max_z(calc.potential.model);calls=[0];calculate=calc.calculate
    def counted(*args,**kwargs):calls[0]+=1;return calculate(*args,**kwargs)
    calc.calculate=counted;controls=[];templates={};records=[];begun=time.perf_counter()
    for i in indices:
        case=plan['cases'][i];parent=case['parent_id'];chosen=data['selected']['plain-descent'][i]
        for stage,path,B in (('direct',case['B']['relaxed_path'],case['B']),('selected',chosen['pose'],chosen['B'])):
            if sha(path)!=B['relaxed_sha256']:raise ValueError('Saved endpoint SHA changed.')
            atoms=read(path,format='vasp');atoms.set_constraint(None);validate_atomic_numbers(atoms.numbers,maximum);atoms.calc=calc
            energy=float(atoms.get_potential_energy());force=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max())
            controls.append(dict(parent_id=parent,stage=stage,**validate_control(energy,force,B,expected)));templates[parent,stage]=atoms.copy()
    save(root/'tight-controls.json',dict(passed=True,device='cpu',checkpoint_sha256=expected,cases=controls))
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    for parent in parents:
        index=int(Path(parent).stem.split('-')[1])
        for stage in ('direct','selected'):
            reference=templates[parent,stage];other=templates[parent,'selected' if stage=='direct' else 'direct']
            for trial in range(4):
                identifier=f'{Path(parent).stem}-{stage}-t{trial}.vasp';folder=root/Path(parent).stem/stage/f'trial-{trial}';folder.mkdir(parents=True,exist_ok=False)
                input_path=folder/'input.vasp';write(input_path,reference,format='vasp',direct=True,sort=False)
                row=dict(candidate_id=identifier,parent_id=parent,stage=stage,trial=trial,input=str(input_path),checkpoint_sha256=expected,status='failed')
                nfe=calls[0];started=time.perf_counter()
                try:
                    atoms=perturb(reference,index,trial);write(input_path,atoms,format='vasp',direct=True,sort=False);atoms.calc=calc
                    row.update(input_sha256=sha(input_path),perturbation_applied=trial!=0)
                    filtered=ExpCellFilter(atoms);opt=BFGS(filtered,logfile=str(folder/'optimizer.log'),maxstep=.05)
                    converged=bool(opt.run(fmax=.001,steps=1600));raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max());force=float(np.linalg.norm(filtered.get_forces(),axis=1).max());energy=float(atoms.get_potential_energy())
                    if not np.array_equal(atoms.numbers,reference.numbers) or not np.isfinite([raw,force,energy]).all():raise ValueError('Species/nonfinite tight result.')
                    path=folder/'relaxed.vasp';write(path,atoms,format='vasp',direct=True,sort=False);s=AseAtomsAdaptor.get_structure(atoms);own=AseAtomsAdaptor.get_structure(reference);opposite=AseAtomsAdaptor.get_structure(other)
                    row.update(status='converged' if converged else 'unconverged',converged=converged,num_atoms=len(atoms),final_energy_eV=energy,
                        max_filter_force=force,max_raw_atomic_force=raw,separated=not len(neighbor_list('d',atoms,.5-1e-9)),optimizer_steps=opt.get_number_of_steps(),
                        volume_A3=atoms.get_volume(),volume_ratio=atoms.get_volume()/reference.get_volume(),relaxed_path=str(path),relaxed_sha256=sha(path),
                        own_geometry=measure(own,s),other_geometry=measure(opposite,s),own_author_match=bool(matcher.fit(own,s)),other_author_match=bool(matcher.fit(opposite,s)))
                except Exception as error:row.update(error=repr(error),input_sha256=sha(input_path))
                row.update(NFE=calls[0]-nfe,elapsed_seconds=time.perf_counter()-started,tight_eligible=tight_eligible(row));records.append(row);save(folder/'result.json',row)
                save(root/'tight-physics.json',dict(policy=TIGHT_POLICY,parents=parents,records=records,controls=controls,checkpoint_sha256=expected,NFE=calls[0],elapsed_seconds=time.perf_counter()-begun))
                print(json.dumps({k:v for k,v in row.items() if k not in ('own_geometry','other_geometry')}),flush=True)
    validate_trials(parents,records);save(root/'tight-physics.complete.json',dict(completed=True,parents=len(parents),trials=len(records)))


if __name__=='__main__':main()
