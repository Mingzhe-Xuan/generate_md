"""Native-gated CF-guided estimates of the fully relaxed property objective."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from guided_settings import POLICY,PARENTS,FAMILIES,ARMS,SCALES,SEED,OUTPUTS
from guided_math import unit_or_zero,projected_normal_draws,estimate
from guided_common import (load,save,sha,common_seed,close,GradientScorer,Geometry,Mechanics,promote,precision_score,matched_random,
    point,best,validate_control,export_evaluation,six_checks_pass,implementation_hashes,source_plan,validate_plan,utility,geometry,family_noise,ray_directions)
from guided_validate import validate_study


def native_gate(root,plan,scorer,double,client):
    from pymatgen.core import Structure
    records=[];baselines=[]
    for i,case in enumerate(plan['cases']):
        parent=case['parent_id'];path=Path(case['B']['relaxed_path'])
        if sha(path)!=case['B']['relaxed_sha256'] or sha(case['raw_path'])!=case['B']['input_sha256']:raise ValueError('Source geometry changed.')
        s=Structure.from_file(path);geom=Geometry(s.atomic_numbers,s.lattice.matrix,s.frac_coords);base=point(parent,case['raw_path'],case['B'],scorer);baselines.append(base)
        direct=scorer.evaluate(geom,np.zeros(geom.size));expected=case['predictions']['after']
        cf_error=max(abs(base['gap']-expected['crystalformer_gap']),abs(base['eform']-expected['crystalformer_eform']),abs(direct['gap']-base['gap']),abs(direct['eform']-base['eform']))
        physical=client.points(geom,np.zeros((1,geom.size)))[0]
        control=validate_control(physical['energy_eV'],float(np.linalg.norm(physical['forces_eV_A'],axis=1).max()),case['B'],physical['checkpoint_sha256'])
        index=int(Path(parent).stem.split('-')[1]);q,draw=common_seed(geom,index,SEED);value=scorer.evaluate(geom,q,True);frozen=scorer.evaluate(geom,q)
        g=geom.project(value['gradient']);precise=precision_score(double,geom,q,scorer.torch.float64,True);legacy=precision_score(double,geom,q,scorer.torch.float64)
        adapter=max(abs(value[k]-frozen[k]) for k in ('gap','eform'));forward=max(abs(value[k]-precise[k]) for k in ('gap','eform'))
        relative=float(np.linalg.norm(g-geom.project(precise['gradient']))/max(np.linalg.norm(precise['gradient']),1e-10));double_adapter=max(abs(precise[k]-legacy[k]) for k in ('gap','eform'));checks=[]
        if i<4:
            rng=np.random.default_rng([SEED,index,1003])
            for block in ('atomic','strain','both'):
                v=geom.project(rng.normal(size=geom.size))
                if block=='atomic':v[-6:]=0
                if block=='strain':v[:-6]=0
                v=unit_or_zero(v)
                for eps in (1e-5,1e-6):
                    plus=precision_score(double,geom,q+eps*v,scorer.torch.float64);minus=precision_score(double,geom,q-eps*v,scorer.torch.float64)
                    fd=(plus['loss']-minus['loss'])/(2*eps);ad=float(g@v);checks.append(dict(block=block,epsilon=eps,FD=fd,predicted=ad,passed=close(fd,ad,.001,.02)))
        guide=unit_or_zero(g);random=matched_random(guide,np.random.default_rng([SEED,index,331]),geom.project)
        normal,scalar=projected_normal_draws(geom.project,geom.size,POLICY['pairs'],np.random.default_rng([SEED,index,330]))
        passed=bool(cf_error<=1e-5 and control['passed'] and adapter<=1e-5 and forward<=1e-5 and relative<=.001 and double_adapter<=1e-10 and (six_checks_pass(checks) if i<4 else True))
        record=dict(parent_id=parent,passed=passed,baseline_CF_error=cf_error,fixed_pose_control=control,adapter_forward_error=adapter,precision_forward_error=forward,
            precision_gradient_relative_error=relative,double_adapter_error=double_adapter,derivative_checks=checks,source_sha256=case['B']['relaxed_sha256'],
            geometry=dict(numbers=geom.numbers.tolist(),cell=geom.cell.tolist(),frac=geom.frac.tolist()),qseed=q.tolist(),seed_draw=draw,
            cf_gradient_norm=float(np.linalg.norm(g)),cf_guide=guide.tolist(),random_guide=random.tolist(),normal=normal.tolist(),scalar=scalar.tolist())
        records.append(record);save(root/'native-progress.json',dict(operators=records,CF=scorer.metadata,NFE=client.NFE))
        print(json.dumps(dict(stage='native',completed=i+1,parent_id=parent,passed=passed)),flush=True)
        if not passed:raise ValueError('Native guided gate failed before scientific proposals.')
    proof=dict(passed=True,operators=records,baseline=baselines,plan_sha256=sha(root/'plan.json'),code_sha256=implementation_hashes(),CF=scorer.metadata,
        double_CF=double.metadata,mechanics=client.ready,NFE=client.NFE)
    save(root/'native.json',proof);return proof


def evaluate_tasks(root,stem,geom,taskspecs,scorer,client,base):
    from ase import Atoms
    from ase.io import write
    rows=[];pending=[];tasks=[];energy=base['B']['final_energy_eV']/base['B']['num_atoms'];folder=root/'inputs'/stem;folder.mkdir(parents=True,exist_ok=True)
    for spec,q,stationary in taskspecs:
        row=dict(spec,q=np.asarray(q).tolist(),utility=1.)
        if stationary or not geom.feasible(q):
            row['status']='stationary_or_geometric_rejection';rows.append(row);continue
        cell,frac=geom.pose(q);path=folder/(row['id']+'.vasp');write(path,Atoms(numbers=geom.numbers,cell=cell,scaled_positions=frac,pbc=True),format='vasp',direct=True,sort=False)
        row.update(input=str(path),input_sha256=sha(path));pending.append(row)
        tasks.append(dict(id=row['id'],action='relax',source=str(path),output=str(root/'B'/stem/row['id'])))
    results=client.request(tasks)
    for row,B in zip(pending,results):
        row.update(status=B['status'],B=B)
        if B['status']=='converged':
            try:q=point(row['id'],row['input'],B,scorer,base['B']['volume_A3'])
            except ValueError as error:row.update(status='CF_prediction_failed',error=repr(error))
            else:row.update(point=q,utility=utility(q,energy))
        rows.append(row)
    byid={r['id']:r for r in rows}
    return [byid[spec['id']] for spec,_,_ in taskspecs]


def search(root,plan,proof,scorer,client):
    attempts=[];seeds=[];estimates={f:[] for f in FAMILIES};selected={k:[] for k in OUTPUTS};baseline=proof['baseline'];start=client.NFE
    for i,(case,operator,base) in enumerate(zip(plan['cases'],proof['operators'],baseline)):
        parent=case['parent_id'];stem=Path(parent).stem;geom=geometry(operator);qseed=np.array(operator['qseed']);index=int(stem.split('-')[1]);energy=base['B']['final_energy_eV']/base['B']['num_atoms']
        seed=evaluate_tasks(root,stem,geom,[(dict(id=stem+'-seed',parent_id=parent,kind='seed'),qseed,False)],scorer,client,base)[0];seeds.append(seed)
        for family_index,family in enumerate(FAMILIES):
            noise,meta=family_noise(operator,family);specs=[]
            for pair,delta in enumerate(noise):
                for side,sign in (('plus',1),('minus',-1)):
                    identifier=f'{stem}-{family}-p{pair}-{side}'
                    specs.append((dict(id=identifier,parent_id=parent,family=family,kind='probe',pair=pair,side=side),qseed+sign*delta,False))
            probes=evaluate_tasks(root,stem,geom,specs,scorer,client,base);attempts.extend(probes)
            plus=[r['utility'] for r in probes if r['side']=='plus'];minus=[r['utility'] for r in probes if r['side']=='minus'];gradient=estimate(noise,plus,minus,POLICY['sigma_RMS_A'],geom.project)
            directions=ray_directions(gradient,operator['cf_guide'],geom,np.random.default_rng([SEED,index,family_index,445]))
            pool=[base]+[r['point'] for r in [seed]+probes if 'point' in r];control=best(pool,energy)
            estimates[family].append(dict(parent_id=parent,noise=noise.tolist(),sampling=meta,plus=plus,minus=minus,gradient=gradient.tolist(),
                gradient_norm=float(np.linalg.norm(gradient)),directions={k:v.tolist() for k,v in directions.items()},probe_control=control))
            specs=[]
            for arm in ARMS:
                d=directions[arm]
                for k,scale in enumerate(SCALES):
                    identifier=f'{stem}-{family}-{arm}-s{k}'
                    specs.append((dict(id=identifier,parent_id=parent,family=family,kind='ray',arm=arm,scale=scale),qseed+scale*d,np.linalg.norm(d)<1e-10))
            rays=evaluate_tasks(root,stem,geom,specs,scorer,client,base);attempts.extend(rays)
            selected[family+'-probes'].append(control)
            for arm in ARMS:selected[family+'-'+arm].append(best(pool+[r['point'] for r in rays if r['arm']==arm and 'point' in r],energy))
        selected['baseline'].append(base)
        save(root/'results.json',dict(baseline=baseline,seeds=seeds,attempts=attempts,estimates=estimates,selected=selected,completed_parents=i+1,
            NFE_native=proof['NFE'],NFE_search=client.NFE-start,NFE_total=client.NFE,CF=scorer.metadata,mechanics_seconds=client.seconds))
        print(json.dumps(dict(stage='search',completed=i+1,parent_id=parent,attempts=len(attempts),NFE=client.NFE,
            errors={k:abs(v[-1]['gap']-2.5) for k,v in selected.items()})),flush=True)
    return load(root/'results.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--run',type=Path,required=True);p.add_argument('--export-only',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('All native CF/physics/search must use Slurm GPU.')
    root=a.run.resolve()
    if not root.exists():
        if a.source is None:p.error('Fresh runs require --source.')
        plan=source_plan(a.source.resolve());root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',plan)
    plan=load(root/'plan.json');validate_plan(plan)
    if (root/'complete.json').exists():
        done=load(root/'complete.json');proof=load(root/'native.json');data=load(root/'results.json')
        if done['results_sha256']!=sha(root/'results.json') or done['native_sha256']!=sha(root/'native.json') or proof['code_sha256']!=implementation_hashes():raise ValueError('Completed guided data/source changed.')
        validate_study(plan,proof,data);print('GUIDED_ALREADY_COMPLETE',flush=True);return
    scorer=GradientScorer()
    if a.export_only:proof=load(root/'native.json');data=load(root/'results.json')
    else:
        double=GradientScorer();promote(double,scorer.torch.float64);rpc=root/'rpc';rpc.mkdir(exist_ok=False);client=Mechanics(rpc)
        try:proof=native_gate(root,plan,scorer,double,client);data=search(root,plan,proof,scorer,client)
        finally:client.close()
    if proof['plan_sha256']!=sha(root/'plan.json') or proof['code_sha256']!=implementation_hashes():raise ValueError('Native proof belongs to a different guided plan/source.')
    validate_study(plan,proof,data)
    for method in OUTPUTS:export_evaluation(root,method,data['selected'][method],plan,scorer)
    save(root/'complete.json',dict(completed=True,parents=len(PARENTS),attempts=1152,results_sha256=sha(root/'results.json'),native_sha256=sha(root/'native.json'),plan_sha256=sha(root/'plan.json'),code_sha256=implementation_hashes()))


if __name__=='__main__':main()
