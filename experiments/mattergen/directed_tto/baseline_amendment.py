"""Pre-outcome baseline amendment pinned to the complete48 fixed-pose diagnostic."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import numpy as np
from probe import load,save,sha,code_hashes
from attractors import validate_control

DIAGNOSTIC_SHA='f1434cd09bd16e9209654ea7fc6a3592c7cb4ff40ee174b6b9793497831f4cee'
AMENDMENT_POLICY=dict(version=1,failed_job=420,diagnostic_job=421,
    hard_gate='All48 original frozen poses: same-checkpoint energy/atom and raw-fmax errors <=1e-4, plus unchanged CF/precision/FD gates.',
    raw_path='Preserve raw-replay energy and strict geometry mismatch as diagnostics, never relabel them as reproduced.',
    original_selection='Original manifest, cohort, gradient, seeds, ray budgets and both selection policies unchanged.',
    extra_control='Score ALL48 cached fresh raw-B endpoints with frozen CF/ALIGNN; report paired contrasts and material gains beyond this additional baseline.',
    additional_robustness=dict(min_material_parents=4,min_gain_beyond_fresh_rawB_eV=.1,require_negative_paired_CF_CI95_vs_fresh_rawB=True),
    interpretation='Directional evidence alone does not prove practical benefit unless gains survive fresh raw-B sensitivity and physical retention checks.')


def validate_diagnostic(plan,diagnostic,digest):
    from validation_run import PLAN_SHA
    if digest!=DIAGNOSTIC_SHA or diagnostic['plan_sha256']!=PLAN_SHA:raise ValueError('Pinned diagnostic/plan changed.')
    rows=diagnostic['cases']
    if diagnostic['parents']!=48 or [r['parent_id'] for r in rows]!=plan['parent_ids'] or not diagnostic['all_fixed_pose_controls_passed']:
        raise ValueError('Complete48 diagnostic required.')
    for case,row in zip(plan['cases'],rows):
        c=row['fixed_pose_control']
        if row['source_sha256']!=case['B']['relaxed_sha256'] or not c['passed'] or not np.isfinite([c['energy_error_eV_atom'],c['raw_fmax_error_eV_A']]).all() or max(c['energy_error_eV_atom'],c['raw_fmax_error_eV_A'])>1e-4:
            raise ValueError('Same-pose potential consistency failed.')
        if row['raw_replay_status']!='converged':raise ValueError('Fresh raw-B sensitivity requires all48 cached terminal poses.')


def cached_replays(source,plan,diagnostic):
    results=[]
    for case,row in zip(plan['cases'],diagnostic['cases']):
        folder=source/'baseline-replay'/Path(case['parent_id']).stem;path=folder/'result.json';r=load(path)
        expected=dict(id=case['parent_id'],action='relax',source=case['raw_path'],output=str(folder))
        if sha(path)!=row['raw_replay_record_sha256'] or load(folder/'task.json')!=expected:raise ValueError('Cached raw-B record/task changed.')
        if (r['status']!='converged' or r.get('converged') is not True or r['input_sha256']!=case['B']['input_sha256']
            or r['checkpoint_sha256']!=case['B']['checkpoint_sha256'] or r.get('maxstep')!=.05 or r.get('step_limit')!=1600
            or r.get('preserve_symmetry') is not True or r['num_atoms']!=case['B']['num_atoms']):raise ValueError('Cached raw-B source/model/profile changed.')
        if sha(case['raw_path'])!=r['input_sha256'] or sha(r['relaxed_path'])!=r['relaxed_sha256']:raise ValueError('Cached raw-B pose changed.')
        if not np.isfinite([r['final_energy_eV'],r['max_raw_atomic_force'],r['max_filter_force']]).all() or r['NFE']<1:raise ValueError('Invalid cached raw-B physics/cost.')
        results.append(r)
    return results


def load_amendment(root,plan):
    amendment=load(root/'baseline-amendment.json');source=Path(amendment['source_run']);path=source/'baseline-diagnostic-421/report.json'
    if amendment['policy']!=AMENDMENT_POLICY or amendment['native_math_sha256']!=code_hashes():raise ValueError('Baseline amendment/source changed.')
    if amendment['plan_sha256']!=sha(root/'validation-plan.json') or amendment['diagnostic_sha256']!=sha(path):raise ValueError('Amendment provenance changed.')
    diagnostic=load(path);validate_diagnostic(plan,diagnostic,sha(path));records=cached_replays(source,plan,diagnostic)
    links=[dict(parent_id=c['parent_id'],record_sha256=r['raw_replay_record_sha256']) for c,r in zip(plan['cases'],diagnostic['cases'])]
    if amendment['raw_replay_records']!=links or amendment['cached_raw_B_NFE']!=sum(r['NFE'] for r in records) or amendment['raw_path_mismatches']!=diagnostic['raw_path_mismatches'] or amendment['original_raw_path_gate_satisfied'] is not False:
        raise ValueError('Amendment cache ledger changed.')
    return amendment,diagnostic,records


def fixed_pose_control(case,point):
    return validate_control(point['energy_eV'],float(np.linalg.norm(point['forces_eV_A'],axis=1).max()),case['B'],point['checkpoint_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    from validation_run import validate_manifest
    source=a.source.resolve();root=a.run.resolve();plan=load(source/'validation-plan.json');validate_manifest(plan,sha(source/'validation-plan.json'))
    diagnostic_path=source/'baseline-diagnostic-421/report.json';diagnostic=load(diagnostic_path);validate_diagnostic(plan,diagnostic,sha(diagnostic_path))
    records=cached_replays(source,plan,diagnostic)
    if any((source/name).exists() for name in ('validation-results.json','validation.complete.json')):raise ValueError('Amendment must precede scientific candidate outcomes.')
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(source/'validation-plan.json',root/'validation-plan.json')
    save(root/'baseline-amendment.json',dict(policy=AMENDMENT_POLICY,created_at=datetime.now(timezone.utc).isoformat(),
        source_run=str(source),plan_sha256=sha(root/'validation-plan.json'),diagnostic_sha256=sha(diagnostic_path),native_math_sha256=code_hashes(),
        raw_replay_records=[dict(parent_id=c['parent_id'],record_sha256=r['raw_replay_record_sha256']) for c,r in zip(plan['cases'],diagnostic['cases'])],
        cached_raw_B_NFE=sum(r['NFE'] for r in records),raw_path_mismatches=diagnostic['raw_path_mismatches'],original_raw_path_gate_satisfied=False))
    load_amendment(root,plan);print('BASELINE_AMENDMENT_RECORDED_BEFORE_PROPOSALS',flush=True)


if __name__=='__main__':main()
