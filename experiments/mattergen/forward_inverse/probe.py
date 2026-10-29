"""Local score-Jacobian/physical-Hessian study with native and finite-difference gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from csp_model import CSP, sha
from physics_client import Physics
from relations import centered_basis, centered, cartesian_noise_covariance, cosine, fd_jacobian, matrix_diagnostics, relative_error, projected_adjoint_jacobian
from prepare_anchors import symmetric_cell

HERE = Path(__file__).resolve().parent
POLICY = dict(seed=20261052, times=[.001,.05,.15,.30], FD_steps_A=[.001,.002],
    adjoint_FD_steps_A=[.002,.004], adjoint_abs=.005, adjoint_rel=.02,
    force_energy_FD_steps_A=[.002,.004], force_energy_abs=.005, force_energy_rel=.02,
    native_dtype_relative=.005, fp32_replay_relative=1e-5, matrix_derivative_QC_relative=.02, probe_RMS_A=.01,
    primary_score_dtype='Original native FP32; full autograd Jacobian, checked against rebuilt-graph finite differences.',
    fp64='Secondary diagnostic only; equivalence criterion remains.005 and failures are explicit.',
    anchor_relaxation=dict(fmax=.001,max_steps=800,cell_relax=True,preserve_symmetry=False),
    coordinates='Fixed cell/species Cartesian covectors; remove three translations only.',
    units=dict(force='eV/angstrom',H='eV/angstrom^2',score='1/angstrom',score_J='1/angstrom^2',covariance='angstrom^2'),
    interpretation='Conditional local slice of joint CSP score, not a claim of thermal sampling or physical temperature.')


def save(path,obj):
    def default(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=default)+'\n')


def close(first,second,absolute,relative):
    return bool(abs(first-second)<=absolute+relative*max(abs(first),abs(second)))


def execute(args):
    import torch
    from ase import Atoms
    from ase.io import write
    root=args.run.resolve()
    selection=json.loads((root/'selection.json').read_text())
    if (root/'plan.json').exists():raise ValueError('Preserve prior attempt; use a new run root.')
    started=time.perf_counter();physics=None
    try:
        model=CSP();physics=Physics(root/'physics')
        torch.manual_seed(POLICY['seed']);torch.cuda.manual_seed_all(POLICY['seed'])
        code={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}
        save(root/'plan.json',dict(policy=POLICY,selection_sha256=sha(root/'selection.json'),code_sha256=code,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model=model.metadata,
            physics=physics.metadata,selection=selection))
        results=[]
        for number,case in enumerate(selection['selected']):
            if sha(case['source'])!=case['source_sha256']:raise ValueError('Reference changed.')
            folder=root/case['id'];folder.mkdir()
            initialNFE=physics.NFE;initialCalls=model.metadata['calls_fp64']+model.metadata['calls_fp32'];tic=time.perf_counter()
            numbers=np.array(case['numbers']);cell=np.array(case['cell']);frac=np.array(case['frac'])
            tasks=[dict(id=case['id']+'-anchor',action='relax',numbers=numbers.tolist(),cell=cell.tolist(),frac=frac.tolist(),
                        output=str(folder/'anchor'),fmax=.001,max_steps=800,cell_relax=True)]
            relaxed=physics.request(tasks)[0]
            row=dict(id=case['id'],system=case['system'],num_atoms=len(numbers),relaxation=relaxed,status='pending')
            if relaxed['status']!='converged':
                row.update(status='anchor_unconverged',seconds=time.perf_counter()-tic)
                results.append(row);save(folder/'result.json',row);save(root/'progress.json',results);continue
            cell=symmetric_cell(np.array(relaxed['cell']));frac=np.array(relaxed['frac']);anchor=frac@cell
            q=centered_basis(len(numbers));d=q.shape[1]
            rng=np.random.default_rng([POLICY['seed'],number]);offset=q@rng.normal(size=d);offset*=POLICY['probe_RMS_A']*np.sqrt(len(numbers))/np.linalg.norm(offset)
            cart=anchor+offset.reshape(-1,3)
            atoms=Atoms(numbers=numbers,cell=cell,positions=cart,pbc=True);write(folder/'probe.vasp',atoms,format='vasp',direct=True,sort=False)
            serial=[0]
            def points(carts,prefix):
                tasks=[]
                for coordinates in carts:
                    serial[0]+=1
                    tasks.append(dict(id=case['id']+'-'+prefix+'-'+str(serial[0]),action='point',numbers=numbers.tolist(),
                                      cell=cell.tolist(),frac=(coordinates@np.linalg.inv(cell)).tolist()))
                rows=physics.request(tasks)
                if any(r['status']!='completed' for r in rows):raise RuntimeError('Physical point failed: '+str(rows))
                return rows
            anchor_point,point=points([anchor,cart],'center')
            if abs(anchor_point['energy_eV']-relaxed['energy_eV'])>1e-4:raise ValueError('Rigid cell reorientation changed physical energy.')
            v=rng.normal(size=d);v/=np.linalg.norm(v);w=rng.normal(size=d);w/=np.linalg.norm(w)
            vc=(q@v).reshape(-1,3);wc=q@w
            controls=[]
            predicted=-float(np.array(point['forces_eV_A']).ravel()@vc.ravel())
            for step in POLICY['force_energy_FD_steps_A']:
                plus,minus=points([cart+step*vc,cart-step*vc],'energyFD')
                finite=(plus['energy_eV']-minus['energy_eV'])/(2*step)
                rec=dict(kind='MatterSim_energy_force',step_A=step,predicted=predicted,FD=finite,
                    passed=close(predicted,finite,POLICY['force_energy_abs'],POLICY['force_energy_rel']))
                controls.append(rec)
                if not rec['passed']:raise ValueError('Energy-force gate failed: '+str(rec))
            force_raw={};hessians={}
            for label,center in [('anchor',anchor),('probe',cart)]:
                matrices=[]
                for step in POLICY['FD_steps_A']:
                    displacements=[center+sign*step*q[:,j].reshape(-1,3) for j in range(d) for sign in (1,-1)]
                    fields=np.array([r['forces_eV_A'] for r in points(displacements,label+'H')]).reshape(d,2,-1)
                    matrix=-q.T@fd_jacobian(fields[:,0],fields[:,1],step)
                    force_raw[label+'_'+str(step)]=fields;matrices.append(matrix)
                hessians[label]=np.stack(matrices)
            np.savez_compressed(folder/'force-data.npz',numbers=numbers,cell=cell,anchor=anchor,probe=cart,Q=q,offset=offset,
                H_anchor=hessians['anchor'],H_probe=hessians['probe'],v=v,w=w,
                H_control_force=np.asarray(point['forces_eV_A']),**force_raw)
            score_fields=[];score_jac=[];score_ad=[];score_centers=[];standards=[];topology=[];per_time=[]
            primary=lambda coordinates,t,**kw:model.score(numbers,cell,coordinates,t,dtype='single',**kw)
            for t in POLICY['times']:
                base=primary(cart,t,adjoint=wc)
                precision=model.score(numbers,cell,cart,t)
                repeat=primary(cart,t)
                native_error=relative_error(base['score'],precision['score']);replay_error=relative_error(base['score'],repeat['score'])
                if replay_error>POLICY['fp32_replay_relative']:
                    raise ValueError('Native FP32 repeatability gate failed: '+str(replay_error))
                if base['std_range']>1e-12:raise ValueError('Expected constant positional marginal std within a structure.')
                for step in POLICY['adjoint_FD_steps_A']:
                    plus=primary(cart+step*vc,t);minus=primary(cart-step*vc,t)
                    same=plus['graph_hash']==minus['graph_hash']==base['graph_hash']
                    finite=float(wc@(plus['score']-minus['score'])/(2*step));expected=float(vc.ravel()@base['adjoint'])
                    passed=close(finite,expected,POLICY['adjoint_abs'],POLICY['adjoint_rel']) if same else None
                    controls.append(dict(kind='CSP_adjoint_FD',t=t,step_A=step,FD=finite,predicted=expected,same_graph=same,passed=passed))
                ad_rows=[primary(cart,t,adjoint=q[:,j]) for j in range(d)]
                ad_matrix=projected_adjoint_jacobian(q,[r['adjoint'] for r in ad_rows])
                ad_same_graph=all(r['graph_hash']==base['graph_hash'] for r in ad_rows)
                fields_at_time=[];jac_at_time=[];hashes_at_time=[]
                for step in POLICY['FD_steps_A']:
                    fields=[];fingerprints=[]
                    for j in range(d):
                        pair=[primary(cart+sign*step*q[:,j].reshape(-1,3),t) for sign in (1,-1)]
                        fields.append([p['score'] for p in pair]);fingerprints.append([p['graph_hash'] for p in pair])
                    fields=np.asarray(fields)
                    matrix=q.T@fd_jacobian(fields[:,0],fields[:,1],step)
                    fields_at_time.append(fields);jac_at_time.append(matrix);hashes_at_time.append(fingerprints)
                at_anchor=primary(anchor,t)
                covariance=cartesian_noise_covariance(cell,base['std'],len(numbers))
                sr=q.T@base['score'];force=q.T@np.asarray(point['forces_eV_A']).ravel()
                radial=-np.linalg.solve(covariance,q.T@offset)
                drift=centered(base['fractional_score']@cell)
                delta=q.T@(base['score']-at_anchor['score'])
                details=dict(t=t,fractional_std=base['std'],fp64_relative_error=native_error,replay_relative_error=replay_error,
                    fp64_same_graph=precision['graph_hash']==base['graph_hash'],fp64_equivalence_passed=native_error<=POLICY['native_dtype_relative'],
                    cos_score_force=cosine(sr,force),
                    cos_drift_force=cosine(q.T@drift,force),cos_score_noise_restoration=cosine(sr,radial),
                    cos_bias_removed_score_noise=cosine(delta,radial),score_bias_norm=float(np.linalg.norm(at_anchor['score'])),
                    jacobian_FD_relative_change=relative_error(jac_at_time[0],jac_at_time[1]),
                    jacobian_ad_FD_errors=[relative_error(ad_matrix,j) for j in jac_at_time],ad_same_graph=ad_same_graph,
                    ad_diagnostics=matrix_diagnostics(ad_matrix),
                    diagnostics=[matrix_diagnostics(j) for j in jac_at_time],
                    graph_unchanged_counts=[sum(a==b==base['graph_hash'] for a,b in hs) for hs in hashes_at_time],
                    coordinate_directions=d)
                details['native_derivative_QC_passed']=bool(ad_same_graph and
                    max(details['jacobian_ad_FD_errors'])<=POLICY['matrix_derivative_QC_relative'] and
                    details['jacobian_FD_relative_change']<=POLICY['matrix_derivative_QC_relative'] and
                    all(count==d for count in details['graph_unchanged_counts']) and
                    all(r['passed'] is True for r in controls if r['kind']=='CSP_adjoint_FD' and r['t']==t))
                per_time.append(details);score_fields.append(fields_at_time);score_jac.append(jac_at_time)
                score_ad.append(ad_matrix)
                score_centers.append([base['score'],at_anchor['score']]);standards.append(base['std']);topology.append(hashes_at_time)
                np.savez_compressed(folder/f'time-{t}.npz',score_fields=np.asarray(fields_at_time),J_fd=np.asarray(jac_at_time),
                    J_ad=ad_matrix,adjoints=np.stack([r['adjoint'] for r in ad_rows]),scalar_adjoint=base['adjoint'],
                    score_probe=base['score'],score_anchor=at_anchor['score'],score_fp64=precision['score'],std=base['std'])
                save(folder/f'time-{t}.json',details)
                print('CURVATURE_TIME',case['id'],t,details['jacobian_FD_relative_change'],flush=True)
            vectors=folder/'vectors.npz'
            np.savez_compressed(vectors,numbers=numbers,cell=cell,anchor=anchor,probe=cart,Q=q,offset=offset,
                H_anchor=hessians['anchor'],H_probe=hessians['probe'],score_jacobian=np.asarray(score_jac),
                score_jacobian_ad=np.asarray(score_ad),
                score_fields=np.asarray(score_fields),score_centers=np.asarray(score_centers),fractional_std=np.asarray(standards),
                force_anchor=np.asarray(anchor_point['forces_eV_A']),force_probe=np.asarray(point['forces_eV_A']),**force_raw)
            row.update(status='completed',controls=controls,times=per_time,graph_fingerprints=topology,
                physical_H_anchor=[matrix_diagnostics(m) for m in hessians['anchor']],physical_H_probe=[matrix_diagnostics(m) for m in hessians['probe']],
                physical_H_anchor_FD_relative_change=relative_error(*hessians['anchor']),physical_H_probe_FD_relative_change=relative_error(*hessians['probe']),
                anchor_point=anchor_point,probe_point=point,vectors_sha256=sha(vectors),
                NFE=physics.NFE-initialNFE,score_calls=model.metadata['calls_fp64']+model.metadata['calls_fp32']-initialCalls,
                seconds=time.perf_counter()-tic)
            save(folder/'result.json',row);results.append(row);save(root/'progress.json',results)
            print('CURVATURE_CASE',case['id'],row['seconds'],flush=True)
        save(root/'complete.json',dict(passed=all(r['status']=='completed' for r in results),cases=results,
            native_derivative_QC_all_passed=bool(results) and all(r['status']=='completed' and all(t['native_derivative_QC_passed'] for t in r['times']) for r in results),
            physics_NFE=physics.NFE,physics_requests=physics.requests,model=model.metadata,seconds=time.perf_counter()-started,
            device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
    except Exception:
        save(root/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-started))
        raise
    finally:
        if physics is not None:physics.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    execute(args)


if __name__=='__main__':main()
