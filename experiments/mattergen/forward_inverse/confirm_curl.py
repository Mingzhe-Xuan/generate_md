"""Frozen-window cross-system local score/force test-plane confirmation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from csp_model import CSP,sha
from physics_client import Physics
from relations import centered_basis,cosine,relative_error
from response_report import curl_evidence


POLICY=dict(seed=20261055,times=[.001,.05,.15,.30],steps_A=[.0005,.001],
    vector_QC_relative=.02,native_replay_relative=1e-5,native_value_equivalence=.005,
    curl_signal_vs_discrepancy=5,prior_systems=['Si','C','TiO2','CaTiO3'],
    interpretation='Frozen two-dimensional random-plane check, not full-Jacobian certification or population prevalence.')


def random_plane(q,rng):
    v=rng.normal(size=q.shape[1]);v/=np.linalg.norm(v)
    w=rng.normal(size=q.shape[1]);w-=v*(v@w);w/=np.linalg.norm(w)
    return q@np.stack([v,w],axis=1)


def projected_adjoints(plane,adjoints):
    return np.asarray(adjoints)@plane


def save(path,obj):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=default)+'\n')


def execute(source,root):
    if root.exists():raise ValueError('Preserve previous confirmation.')
    root.mkdir(parents=True)
    complete=json.loads((source/'complete.json').read_text());plan=json.loads((source/'plan.json').read_text())
    if not complete['execution_complete']:raise ValueError('Source incomplete.')
    if [c['id'] for c in complete['cases']]!=[c['id'] for c in plan['selection']['selected']]:raise ValueError('Source coverage mismatch.')
    physics=None;started=time.perf_counter()
    try:
        model=CSP();physics=Physics(root/'physics')
        report=dict(policy=POLICY,source_plan_sha256=sha(source/'plan.json'),source_complete_sha256=sha(source/'complete.json'),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            script_sha256=sha(Path(__file__)),model=model.metadata,physics=physics.metadata,cases=[])
        save(root/'plan.json',report)
        for index,case in enumerate(complete['cases']):
            row=dict(id=case['id'],system=case['system'],prior_system=case['system'] in POLICY['prior_systems'],status='pending',times=[])
            if 'probe' not in case:
                row['status']='source_probe_unavailable';report['cases'].append(row);continue
            numbers=np.array(case['numbers']);cell=np.array(case['cell']);cart=np.array(case['probe'])
            rng=np.random.default_rng([POLICY['seed'],index]);plane=random_plane(centered_basis(len(numbers)),rng)
            tasks=[dict(id=case['id']+'-base',action='point',numbers=numbers.tolist(),cell=cell.tolist(),frac=(cart@np.linalg.inv(cell)).tolist())]
            for h in POLICY['steps_A']:
                for direction in plane.T:
                    for sign in (1,-1):
                        pos=cart+sign*h*direction.reshape(-1,3)
                        tasks.append(dict(id=case['id']+'-'+str(len(tasks)),action='point',numbers=numbers.tolist(),cell=cell.tolist(),frac=(pos@np.linalg.inv(cell)).tolist()))
            physical=physics.request(tasks)
            if any(r['status']!='completed' for r in physical):
                row.update(status='physical_point_failed',physics=physical);report['cases'].append(row);continue
            forces=np.array([r['forces_eV_A'] for r in physical[1:]]).reshape(2,2,2,-1)
            physical_j=np.stack([np.stack([plane.T@((pair[0]-pair[1])/(2*h)) for pair in fields],axis=1) for h,fields in zip(POLICY['steps_A'],forces)])
            physical_energies=np.array([r['energy_eV'] for r in physical[1:]]).reshape(2,2,2)
            fbase=np.asarray(physical[0]['forces_eV_A']).ravel();expected=-plane.T@fbase
            physical_controls=[]
            for j,h in enumerate(POLICY['steps_A']):
                fd=(physical_energies[j,:,0]-physical_energies[j,:,1])/(2*h)
                physical_controls.append(dict(step_A=h,AD=expected,FD=fd,
                    passed=np.abs(fd-expected)<=.005+.02*np.maximum(np.abs(fd),np.abs(expected))))
            row.update(physical_J=physical_j,physical_curl=[float(j[1,0]-j[0,1]) for j in physical_j],
                physical_FD_relative_change=relative_error(*physical_j),energy_force_controls=physical_controls)
            folder=root/case['id'];folder.mkdir()
            for t in POLICY['times']:
                dtypes={};raw={}
                for dtype in ('single','double'):
                    field=lambda positions,**kw:model.score(numbers,cell,positions,t,dtype=dtype,**kw)
                    base=field(cart);repeat=field(cart)
                    ad=[field(cart,adjoint=u) for u in plane.T]
                    matrix=projected_adjoints(plane,[r['adjoint'] for r in ad])
                    modes={}
                    for mode in ('inference','grad_enabled'):
                        fields=[];graphs=[]
                        for h in POLICY['steps_A']:
                            pairs=[];hashes=[]
                            for direction in plane.T:
                                values=[field(cart+sign*h*direction.reshape(-1,3),
                                    **({'adjoint':np.zeros(cart.size)} if mode=='grad_enabled' else {})) for sign in (1,-1)]
                                pairs.append([r['score'] for r in values]);hashes.append([r['graph_hash'] for r in values])
                            fields.append(pairs);graphs.append(hashes)
                        fields=np.asarray(fields)
                        finite=np.stack([np.stack([plane.T@((pair[0].astype(float)-pair[1].astype(float))/(2*h)) for pair in pairs],axis=1) for h,pairs in zip(POLICY['steps_A'],fields)])
                        ad_graph_ok=all(r['graph_hash']==base['graph_hash'] for r in ad)
                        graph_ok=[ad_graph_ok and all(a==b==base['graph_hash'] for a,b in hashes) for hashes in graphs]
                        evidence=curl_evidence(matrix,*np.eye(2),finite[:,:,0],finite[:,:,1],graph_ok)
                        modes[mode]=dict(evidence,pair_graphs=graphs)
                        raw[dtype+'_'+mode+'_fields']=fields
                    replay=relative_error(base['score'],repeat['score'])
                    if replay>POLICY['native_replay_relative']:
                        for evidence in modes.values():evidence['corroborated_nonzero_curl']=False
                    dtypes[dtype]=dict(modes=modes,matrix=matrix,replay_relative_error=replay,
                        grad_vs_inference_relative_error=relative_error(base['score'],ad[0]['score']),
                        base_graph=base['graph_hash'],adjoint_graphs=[a['graph_hash'] for a in ad],
                        cos_score_force=cosine(base['score'],fbase))
                    raw.update({dtype+'_score':base['score'],dtype+'_grad_score':ad[0]['score'],dtype+'_repeat':repeat['score'],
                                dtype+'_adjoints':np.array([r['adjoint'] for r in ad])})
                filename=f'time-{t}.npz'
                np.savez_compressed(folder/filename,plane=plane,numbers=numbers,cell=cell,cart=cart,physical_forces=forces,force_base=fbase,**raw)
                value_error=relative_error(raw['single_score'],raw['double_score'])
                row['times'].append(dict(t=t,dtypes=dtypes,fp32_fp64_value_relative_error=value_error,
                    fp64_value_equivalence_passed=value_error<=POLICY['native_value_equivalence'],
                    vectors=case['id']+'/'+filename,vectors_sha256=sha(folder/filename)))
            row['status']='completed';report['cases'].append(row);save(root/'progress.json',report)
            print('CONFIRM_CASE',case['id'],flush=True)
        report.update(execution_complete=True,seconds=time.perf_counter()-started,physics_NFE=physics.NFE)
        save(root/'complete.json',report)
    finally:
        if physics is not None:physics.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    execute(args.source.resolve(),args.run.resolve())


if __name__=='__main__':main()
