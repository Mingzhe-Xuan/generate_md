"""Separate frozen-pose potential consistency from raw-start relaxation path repeatability."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from geometry import Geometry
from runtime import Mechanics
from probe import load,save,sha
from attractors import validate_control


def compare(case,point,replay,match):
    reference=case['B'];raw=float(np.linalg.norm(point['forces_eV_A'],axis=1).max())
    try:control=validate_control(point['energy_eV'],raw,reference,point['checkpoint_sha256'])
    except ValueError as error:control=dict(passed=False,error=repr(error))
    delta=abs(replay['final_energy_eV']-reference['final_energy_eV'])/reference['num_atoms'] if replay['status']=='converged' else None
    return dict(parent_id=case['parent_id'],fixed_pose_control=control,raw_replay_status=replay['status'],raw_replay_energy_error_eV_atom=delta,
        raw_replay_strict_match=bool(match),raw_path_reproduced=bool(delta is not None and delta<=1e-4 and match),
        old_steps=reference.get('optimizer_steps'),new_steps=replay.get('optimizer_steps'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Potential diagnostics must use Slurm GPU.')
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    root=a.run.resolve();plan=load(root/'validation-plan.json');out=root/('baseline-diagnostic-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    rpc=out/'rpc';rpc.mkdir();client=Mechanics(rpc);controls=[]
    strict=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
    try:
        for case in plan['cases']:
            parent=case['parent_id'];path=case['B']['relaxed_path'];replay_path=root/'baseline-replay'/Path(parent).stem/'result.json'
            if sha(path)!=case['B']['relaxed_sha256']:raise ValueError('Frozen original pose changed.')
            replay=load(replay_path)
            if replay['input_sha256']!=case['B']['input_sha256'] or replay['checkpoint_sha256']!=case['B']['checkpoint_sha256'] or any(k in replay and replay[k]!=v for k,v in (('maxstep',.05),('step_limit',1600))):
                raise ValueError('Raw replay source/model/profile changed.')
            s=Structure.from_file(path);geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords)
            point=client.points(geom,np.zeros((1,geom.size)))[0]
            fit=replay['status']=='converged' and bool(strict.fit(s,Structure.from_file(replay['relaxed_path'])))
            row=compare(case,point,replay,fit);row['source_sha256']=case['B']['relaxed_sha256'];row['raw_replay_record_sha256']=sha(replay_path);controls.append(row)
        result=dict(role='Diagnostic only; no gradient proposals or gate amendment authorized by this file.',plan_sha256=sha(root/'validation-plan.json'),
            parents=len(controls),all_fixed_pose_controls_passed=all(r['fixed_pose_control']['passed'] for r in controls),
            raw_path_mismatches=sum(not r['raw_path_reproduced'] for r in controls),cases=controls,NFE=client.NFE,mechanics=client.ready)
        save(out/'report.json',result);print(json.dumps(result),flush=True)
    finally:client.close()


if __name__=='__main__':main()
