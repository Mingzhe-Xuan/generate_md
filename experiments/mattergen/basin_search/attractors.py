"""Independent CPU small-perturbation re-quench diagnostic of distinct endpoints."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from policy import ACCEPTANCE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_control(energy,max_force,record,checkpoint_sha):
    if record['checkpoint_sha256']!=checkpoint_sha:raise ValueError('Archived potential changed.')
    delta=abs(energy-record['final_energy_eV'])/record['num_atoms']
    force_delta=abs(max_force-record['max_raw_atomic_force'])
    if not np.isfinite([energy,max_force,delta,force_delta]).all() or delta>1e-4 or force_delta>1e-4:raise ValueError(f'CPU/GPU endpoint control failed: energy error {delta} eV/atom, raw-fmax error {force_delta} eV/A.')
    return dict(energy_error_eV_atom=float(delta),raw_fmax_error_eV_A=float(force_delta),passed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('All mechanics must run in Slurm.')
    import torch
    import ase.constraints
    from ase.filters import Filter,ExpCellFilter
    if not hasattr(ase.constraints,'Filter'):ase.constraints.Filter=Filter
    from ase.io import read,write
    from ase.optimize import BFGS
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from mattersim.forcefield import MatterSimCalculator,potential
    from proposals import mutate
    smoacs=Path(__file__).resolve().parents[2]/'smoacs';sys.path[:0]=[str(smoacs),str(smoacs/'postprocess')]
    from alignn_checkpoint import TrustedCheckpointTorch
    from mattersim_support import validate_atomic_numbers,model_max_z
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    root=args.run.resolve();load=lambda path:json.loads(path.read_text())
    report=load(root/'report.json');plan=report['plan'];search=load(root/'search-results.json')
    if plan['acceptance']!=ACCEPTANCE or len(set(plan['parent_ids']))!=16:raise ValueError('Frozen cohort/criteria changed.')
    checkpoint=Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve()
    expected=load(smoacs/'postprocess/sources.json')['mattersim']['checkpoint_sha256']
    if sha(checkpoint)!=expected:raise ValueError('Potential checkpoint changed.')
    original=potential.torch;potential.torch=TrustedCheckpointTorch(torch,[expected])
    try:
        with redirect_stdout(sys.stderr):calc=MatterSimCalculator(load_path=str(checkpoint),device='cpu',load_training_state=False)
    finally:potential.torch=original
    if any(parameter.is_cuda for parameter in calc.potential.model.parameters()):raise RuntimeError('CPU diagnostic accessed GPU.')
    maximum=model_max_z(calc.potential.model);calls=[0];calculate=calc.calculate
    def counted(*a,**kw):calls[0]+=1;return calculate(*a,**kw)
    calc.calculate=counted
    output=root/f'attractors-{os.environ["SLURM_JOB_ID"]}';output.mkdir(exist_ok=False)
    cases={c['parent_id']:c for c in plan['cases']};points={c['parent_id']:p for c,p in zip(plan['cases'],search['selected']['beam'])}
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    selected=[r for r in report['methods']['beam']['cases'] if r['material_CF_improvement']]
    controls=[];templates={};records=[];start=time.perf_counter()
    for row in selected:
        parent=row['parent_id'];point=points[parent];baseline=cases[parent]['B']
        for stage,path,record in (('direct',baseline['relaxed_path'],baseline),('selected',point['pose'],point['B'])):
            if sha(path)!=record['relaxed_sha256']:raise ValueError('Original endpoint SHA changed.')
            atoms=read(path,format='vasp');atoms.set_constraint(None);validate_atomic_numbers(atoms.numbers,maximum);atoms.calc=calc
            energy=float(atoms.get_potential_energy());raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max())
            control=validate_control(energy,raw,record,expected);controls.append(dict(parent_id=parent,stage=stage,**control))
            templates[(parent,stage)]=atoms.copy()
    (output/'controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    print(json.dumps(dict(CPU_controls_passed=True,endpoint_count=len(controls))),flush=True)
    for index,row in enumerate(selected):
        parent=row['parent_id']
        for stage_index,stage in enumerate(('direct','selected')):
            reference=templates[(parent,stage)];other=templates[(parent,'selected' if stage=='direct' else 'direct')]
            for trial in range(3):
                folder=output/Path(parent).stem/stage/f'trial-{trial}';folder.mkdir(parents=True,exist_ok=False)
                nfe=calls[0];begun=time.perf_counter();event=dict(parent_id=parent,stage=stage,trial=trial,seed=[20261028,index,stage_index,trial])
                try:
                    atoms=mutate(reference,reference,np.random.default_rng(event['seed']),.02,strain=.003)
                    write(folder/'input.vasp',atoms,format='vasp',direct=True,sort=False);atoms.set_constraint(None);atoms.calc=calc
                    filtered=ExpCellFilter(atoms);opt=BFGS(filtered,logfile=str(folder/'optimizer.log'),maxstep=.05)
                    converged=bool(opt.run(fmax=.01,steps=1600));force=float(np.linalg.norm(filtered.get_forces(),axis=1).max())
                    raw=float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max());energy=float(atoms.get_potential_energy())
                    if not np.array_equal(atoms.numbers,reference.numbers):raise ValueError('Ordered species changed.')
                    if not np.isfinite([energy,force,raw]).all():raise ValueError('Nonfinite endpoint.')
                    write(folder/'relaxed.vasp',atoms,format='vasp',direct=True,sort=False)
                    result=AseAtomsAdaptor.get_structure(atoms);own=AseAtomsAdaptor.get_structure(reference);different=AseAtomsAdaptor.get_structure(other)
                    event.update(converged=converged,max_filter_force=force,max_raw_force=raw,energy_eV_atom=energy/len(atoms),steps=opt.get_number_of_steps(),
                        own_author_match=bool(matcher.fit(result,own)),other_author_match=bool(matcher.fit(result,different)),
                        own_geometry=measure(own,result),other_geometry=measure(different,result),relaxed_sha256=sha(folder/'relaxed.vasp'),status='converged' if converged else 'unconverged')
                except Exception as error:event.update(status='failed',error=repr(error))
                event.update(NFE=calls[0]-nfe,CPU_seconds=time.perf_counter()-begun);records.append(event)
                (folder/'result.json').write_text(json.dumps(event,indent=2,allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in event.items() if k not in ('own_geometry','other_geometry')}),flush=True)
    result=dict(device='cpu',checkpoint_sha256=expected,search_seed=report['search_seed'],controls=controls,records=records,NFE=calls[0],elapsed_seconds=time.perf_counter()-start,
        note='Diagnostic only: 3 small .02A RMS/.003strain perturbations per endpoint, unconstrained same-fmax quench. Does not change original criteria, establish global basins, or certify DFT/phonon stability.')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),trials=len(records),CPU_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
