"""Native derivative controls and immutable direction operators for frozen16."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from geometry import Geometry
from directions import normalize,hessian,soft_descent,matched_random,forward,unrolled
from settings import POLICY,SEED
from runtime import Mechanics
from cf_gradient import GradientScorer


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def code_hashes():
    return {p:sha(Path(__file__).with_name(p)) for p in ('geometry.py','directions.py','cf_gradient.py','mechanics.py','runtime.py','settings.py','probe.py')}


def common_seed(geom,index,seed=SEED):
    for attempt in range(20):
        rng=np.random.default_rng([seed,index,attempt]);q=geom.project(rng.normal(size=geom.size))
        q[:-6]*=.03/max(np.linalg.norm(q[:-6]),1e-30)
        strain=np.einsum('a,aij->ij',q[-6:]/geom.scale,geom.basis);q[-6:]*=.003/max(np.linalg.norm(strain,2),1e-30)
        if geom.feasible(q):return q,attempt
    return np.zeros(geom.size),20


def close(fd,predicted,absolute,relative):return bool(abs(fd-predicted)<=absolute+relative*max(abs(fd),abs(predicted)))


def controls(geom,q,g,H,eta,client,scorer,index):
    rng=np.random.default_rng([SEED,index,777]);records=[]
    value_gradient=client.gradients(geom,np.array([q]))[0]
    callback=lambda Q:client.gradients(geom,Q)
    glook,states=unrolled(q,callback,lambda v:scorer.evaluate(geom,v,True)['gradient'],eta,8,geom.project)
    for block in ('atomic','strain','both'):
        v=geom.project(rng.normal(size=geom.size))
        if block=='atomic':v[-6:]=0
        if block=='strain':v[:-6]=0
        v=normalize(v)
        for eps in (.004,.002):
            plus=scorer.evaluate(geom,q+eps*v);minus=scorer.evaluate(geom,q-eps*v);fd=(plus['loss']-minus['loss'])/(2*eps);pred=float(g@v)
            records.append(dict(kind='CF_gradient',block=block,epsilon=eps,FD=fd,predicted=pred,passed=close(fd,pred,.001,.02)))
            energy=client.points(geom,np.array([q+eps*v,q-eps*v]));fd=(energy[0]['energy_eV']-energy[1]['energy_eV'])/(2*eps);pred=float(value_gradient@v)
            records.append(dict(kind='physical_gradient',block=block,epsilon=eps,FD=fd,predicted=pred,passed=close(fd,pred,.005,.01)))
            if block=='both':
                qp=forward(q+eps*v,callback,eta,8,geom.project)[-1];qm=forward(q-eps*v,callback,eta,8,geom.project)[-1]
                fd=(scorer.evaluate(geom,qp)['loss']-scorer.evaluate(geom,qm)['loss'])/(2*eps);pred=float(glook@v)
                records.append(dict(kind='unrolled8_gradient',block=block,epsilon=eps,FD=fd,predicted=pred,passed=close(fd,pred,.002,.05)))
    return records,glook,states


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    root=args.run.resolve();plan=load(root/'plan.json')
    if plan['policy']!=POLICY or len(plan['cases'])!=16:raise ValueError('Frozen protocol changed.')
    if (root/'native-probe.json').exists():raise ValueError('Use a fresh run for a new native gate.')
    operator_dir=root/'operators';operator_dir.mkdir(exist_ok=False);rpc=root/'probe-rpc';rpc.mkdir(exist_ok=False)
    scorer=GradientScorer();client=Mechanics(rpc);checks=[];operators=[]
    try:
        replays=client.request([dict(id=c['parent_id'],action='relax',source=c['raw_path'],output=str(root/'baseline-replay'/Path(c['parent_id']).stem)) for c in plan['cases']])
        strict=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
        for index,(case,replay) in enumerate(zip(plan['cases'],replays)):
            parent=case['parent_id'];reference=Structure.from_file(case['B']['relaxed_path']);geom=Geometry(reference.atomic_numbers,reference.lattice.matrix,reference.frac_coords)
            expected=case['predictions']['after'];direct=scorer.evaluate(geom,np.zeros(geom.size));native=scorer.predict(Path(case['B']['relaxed_path']))
            cf_error=max(abs(direct['gap']-expected['crystalformer_gap']),abs(direct['eform']-expected['crystalformer_eform']),abs(direct['gap']-native[0]),abs(direct['eform']-native[1]))
            energy_error=abs(replay['final_energy_eV']-case['B']['final_energy_eV'])/geom.n if replay['status']=='converged' else None
            fit=replay['status']=='converged' and bool(strict.fit(reference,Structure.from_file(replay['relaxed_path'])))
            baseline_pass=bool(cf_error<=1e-5 and energy_error is not None and energy_error<=1e-4 and fit)
            if not baseline_pass:raise ValueError('Native baseline control failed: '+str((parent,cf_error,energy_error,fit)))
            global_index=int(Path(parent).stem.split('-')[1]);q,seed_attempt=common_seed(geom,global_index,plan['search_seed'])
            score=scorer.evaluate(geom,q,True);g=geom.project(score['gradient']);H,asymmetry=hessian(lambda Q:client.gradients(geom,Q),q)
            if asymmetry>POLICY['Hessian_asymmetry_relative_max']:raise ValueError(f'Hessian asymmetry {parent}: {asymmetry}')
            eigenvalues=np.linalg.eigvalsh(H);eta=.3/max(np.max(np.abs(eigenvalues)),.1)
            native_checks=[]
            if index<4:native_checks,glook,states=controls(geom,q,g,H,eta,client,scorer,global_index)
            else:glook,states=unrolled(q,lambda Q:client.gradients(geom,Q),lambda v:scorer.evaluate(geom,v,True)['gradient'],eta,8,geom.project)
            stationary=np.linalg.norm(g)<1e-10
            plain=np.zeros_like(g) if stationary else normalize(-g)
            soft,soft_meta=(np.zeros_like(g),dict(stationary=True)) if stationary else soft_descent(g,H,geom.project)
            look=np.zeros_like(g) if np.linalg.norm(glook)<1e-10 else normalize(-geom.project(glook))
            matrix=np.array([plain,soft,look]);randoms=np.array([matched_random(d,np.random.default_rng([plan['search_seed'],global_index,f,555]),geom.project) for f,d in enumerate(matrix)])
            path=operator_dir/(Path(parent).stem+'.npz');np.savez_compressed(path,qseed=q,descent=matrix,random=randoms,Hessian=H,gradient=g,unrolled_gradient=glook,eta=eta,lookahead_states=np.array(states))
            entry=dict(parent_id=parent,baseline_CF_error=cf_error,baseline_energy_error_eV_atom=energy_error,baseline_match=fit,
                Hessian_asymmetry=asymmetry,native_checks=native_checks,passed=all(r['passed'] for r in native_checks),seed_draw_attempt=seed_attempt,
                initial_gap=score['gap'],initial_eform=score['eform'],gradient_norm=float(np.linalg.norm(g)),unrolled_gradient_norm=float(np.linalg.norm(glook)),
                eta=eta,soft_metadata=soft_meta,operator_path=str(path),operator_sha256=sha(path),source_sha256=sha(case['B']['relaxed_path']))
            checks.append(entry);operators.append(entry)
            save(root/'probe-progress.json',dict(cases=checks,CF=scorer.metadata,NFE=client.NFE))
            print(json.dumps({k:v for k,v in entry.items() if k not in ('native_checks','soft_metadata')}),flush=True)
            if not entry['passed']:raise ValueError('Native derivative gate failed; inspect probe-progress before any scientific proposals.')
        save(root/'native-probe.json',dict(passed=True,code_sha256=code_hashes(),operators=operators,CF=scorer.metadata,mechanics=client.ready,NFE=client.NFE,mechanics_seconds=client.seconds,
            note='16frozen-baseline controls, fullcanonicalB replay, 4predefined-parent FD input/physical/unrolled-gradient tests. Finite8-step derivative is not full-relaxation derivative.'))
    finally:client.close()


if __name__=='__main__':main()
