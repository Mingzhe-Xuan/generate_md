"""Three frozen-budget post-quench searches; all CF inference is CUDA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

from policy import METHODS,ROUNDS,WIDTH,physical,best,search_loss
from proposals import mutate
from scorer import Scorer

HERE=Path(__file__).resolve().parent
SMOACS=HERE.parents[1]/'smoacs'
sys.path.insert(0,str(SMOACS/'alternating'))
from rpc import read_response


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class Workers:
    def __init__(self,root):
        self.root=root;self.workers=[];self.logs=[];self.metadata=[]
        for i in range(2):
            log=(root/f'mechanics-{i}.err').open('a');self.logs.append(log)
            process=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(HERE/'mechanics.py'),
                '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve())],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MATTERGEN_CF_COHORT=f'basin_mechanics_{i}'))
            self.workers.append(process);meta=read_response(process.stdout,root/f'mechanics-{i}.stdout')
            if not meta['ready']:raise RuntimeError('Mechanics worker unavailable.')
            self.metadata.append(meta)
        save(root/'mechanics-metadata.json',self.metadata)

    def request(self,tasks):
        finished={};pending=[]
        for task in tasks:
            folder=Path(task['output']);record=folder/'result.json'
            if record.exists():
                r=load(record)
                if load(folder/'task.json')!=task or r['input_sha256']!=sha(task['source']):raise ValueError('Stale mechanics request.')
                if not r.get('infrastructure_failure'):finished[task['id']]=r;continue
            if folder.exists():folder.rename(folder.with_name(folder.name+f'-interrupted-{time.time_ns()}'))
            pending.append(task)
        def call(i,subset):
            if not subset:return []
            process=self.workers[i];process.stdin.write(json.dumps(dict(tasks=subset))+'\n');process.stdin.flush()
            rows=read_response(process.stdout,self.root/f'mechanics-{i}.stdout')['results']
            if len(rows)!=len(subset):raise ValueError('Mechanics attempt missing.')
            for task in subset:save(Path(task['output'])/'task.json',task)
            return rows
        with ThreadPoolExecutor(max_workers=2) as pool:
            batches=list(pool.map(lambda i:call(i,pending[i::2]),range(2)))
        for rows in batches:
            for row in rows:
                if row.get('infrastructure_failure'):raise RuntimeError('Mechanics infrastructure failure; preserve artifacts and restart isolated workers.')
                finished[row['candidate_id']]=row
        return [finished[t['id']] for t in tasks]

    def close(self):
        for p in self.workers:
            if p.poll() is None:
                try:p.stdin.write('{"stop":true}\n');p.stdin.flush();p.wait(timeout=30)
                except (BrokenPipeError,subprocess.TimeoutExpired):p.terminate();p.wait(timeout=20)
        for log in self.logs:log.close()


def point(identifier,input_path,B,scorer,reference_volume=None):
    from pymatgen.core import Structure
    pose=Path(B['relaxed_path']);s=Structure.from_file(pose);gap,ef=scorer.predict(pose)
    b=dict(B)
    b.update(volume_A3=s.volume,volume_ratio=s.volume/reference_volume if reference_volume is not None else 1.)
    if 'separated' not in b:b['separated']=len(s.get_neighbor_list(.5-1e-9)[0])==0
    return dict(id=identifier,input=str(input_path),pose=str(pose),gap=gap,eform=ef,B=b)


def archive(points,base_energy):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher=StructureMatcher(ltol=.1,stol=.2,angle_tol=5,primitive_cell=False,scale=False,attempt_supercell=False)
    chosen=[];structures=[]
    for p in sorted([p for p in points if physical(p['B'],base_energy)],key=lambda p:(search_loss(p),p['id'])):
        structure=Structure.from_file(p['pose'])
        if any(matcher.fit(structure,other) for other in structures):continue
        chosen.append(p);structures.append(structure)
        if len(chosen)==WIDTH:break
    return chosen or [points[0]]


def export(root,method,selected,plan,scorer):
    destination=root/f'evaluation-{method}'
    if (destination/'inventory-249/inventory.json').exists():return
    destination.mkdir(exist_ok=True);(destination/'inventory-249').mkdir(exist_ok=True);rows=[];inventory=[]
    for case,p in zip(plan['cases'],selected):
        identifier=case['parent_id'];source=Path(p['input']);B=dict(p['B']);gap,ef=scorer.predict(source)
        if B['input_sha256']!=sha(source):raise ValueError('Selected B input provenance lost.')
        folder=destination/'relaxation'/Path(identifier).stem;folder.mkdir(parents=True,exist_ok=False)
        shutil.copy2(p['pose'],folder/'relaxed.vasp');B.update(candidate_id=identifier,relaxed_path=str((folder/'relaxed.vasp').resolve()))
        save(folder/'result.json',B)
        inventory.append(dict(case['inventory'],candidate_id=identifier,source=str(source),sha256=sha(source),original_gap=gap,original_eform=ef,original_valid=str(case['neutral']),search_point_id=p['id']))
        rows.append(dict(original_fnames=identifier,bandgap_onehot=gap,ef_onehot=ef,is_neutral_cmn=str(case['neutral']),**{'minbond_less_than_0.5':'True'},valid_structure=str(case['neutral'])))
    with (destination/'result.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(destination/'inventory-249/inventory.json',dict(role='Frozen postquench basin search choices',denominator=16,candidates=inventory,source_csv_sha256=sha(destination/'result.csv')))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--seed',type=int,default=20261026);p.add_argument('--probe-only',action='store_true');args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    from ase.io import read,write
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    root=args.run.resolve();plan=load(root/'plan.json');cases=plan['cases'];scorer=Scorer();workers=Workers(root)
    baseline=[];controls=[];base_energy={};all_points={};selected={};history=[]
    try:
        for case in cases:
            key=case['parent_id'];p0=point(key,case['raw_path'],case['B'],scorer);expected=case['predictions']['after']
            error=max(abs(p0['gap']-expected['crystalformer_gap']),abs(p0['eform']-expected['crystalformer_eform']))
            if error>1e-5:raise ValueError('Frozen CUDA CF baseline control failed.')
            baseline.append(p0);base_energy[key]=case['B']['final_energy_eV']/case['B']['num_atoms'];controls.append(dict(parent=key,CF_error=error))
        replay=workers.request([dict(id=c['parent_id'],source=c['raw_path'],output=str(root/'baseline-replay'/Path(c['parent_id']).stem),action='relax') for c in cases])
        matcher=StructureMatcher(ltol=.02,stol=.05,angle_tol=1,primitive_cell=False,scale=False,attempt_supercell=False)
        for c,b,control in zip(cases,replay,controls):
            energy=abs(b['final_energy_eV']-c['B']['final_energy_eV'])/c['B']['num_atoms'] if b['status']=='converged' else None
            fit=b['status']=='converged' and matcher.fit(Structure.from_file(b['relaxed_path']),Structure.from_file(c['B']['relaxed_path']))
            control.update(replay_energy_error_eV_atom=energy,replay_match=bool(fit),passed=energy is not None and energy<=1e-4 and bool(fit))
        save(root/'native-controls.json',dict(passed=all(c['passed'] for c in controls),cases=controls,CF=scorer.metadata))
        if not all(c['passed'] for c in controls):raise ValueError('Canonical mechanics replay gate failed.')
        if args.probe_only:
            md_tasks=[dict(id=Path(c['parent_id']).stem+'-probe',source=c['B']['relaxed_path'],reference=c['B']['relaxed_path'],
                output=str(root/'native-probe'/Path(c['parent_id']).stem/'MD'),action='md',seed=[args.seed,9999,i],temperature=600) for i,c in enumerate(cases[:4])]
            md=workers.request(md_tasks)
            save(root/'native-probe.json',dict(passed=all(r['status']=='escaped' for r in md),MD=md,CF=scorer.metadata))
            if not all(r['status']=='escaped' for r in md):raise ValueError('The four predetermined MD numerical controls failed.')
            quenches=workers.request([dict(id=t['id'],source=r['source'],output=str(Path(t['output']).parent/'B'),action='relax') for t,r in zip(md_tasks,md)])
            save(root/'native-probe.json',dict(passed=True,MD=md,B=quenches,CF=scorer.metadata,
                note='Numerical MD controls on first4 frozen parents. Quench success is reported, not required for mathematical MD control.'))
            print(json.dumps(dict(native_gate_passed=True,parents=16,MD_controls=4)),flush=True)
            return
        release_tasks=[]
        for index,c in enumerate(cases):
            reference=read(c['B']['relaxed_path']);trial=mutate(reference,reference,np.random.default_rng([args.seed,index,999]),.05)
            path=root/'release-inputs'/c['parent_id'];path.parent.mkdir(exist_ok=True);write(path,trial,format='vasp',direct=True,sort=False)
            release_tasks.append(dict(id=c['parent_id'],source=str(path),output=str(root/'release'/Path(c['parent_id']).stem),action='relax',preserve_symmetry=False))
        release=workers.request(release_tasks);selected['symmetry_release']=[]
        for c,task,b,p0 in zip(cases,release_tasks,release,baseline):
            selected['symmetry_release'].append(point(c['parent_id']+'-release',task['source'],b,scorer,p0['B']['volume_A3']) if b['status']=='converged' else p0)
        for mindex,method in enumerate(METHODS):
            pools={c['parent_id']:[p0] for c,p0 in zip(cases,baseline)}
            for generation in range(ROUNDS):
                proposals=[];md_tasks=[]
                for index,(case,p0) in enumerate(zip(cases,baseline)):
                    parent=case['parent_id'];elite=archive(pools[parent],base_energy[parent])
                    for child in range(WIDTH):
                        identifier=Path(parent).stem+f'-{method}-g{generation:02}-c{child}'
                        start=p0 if method=='iid' else elite[child%len(elite)]
                        rng=np.random.default_rng([args.seed,mindex,index,generation,child]);reference=read(p0['pose'])
                        if method=='soft_md':
                            md_tasks.append(dict(id=identifier,source=start['pose'],reference=p0['pose'],output=str(root/method/f'g{generation}'/'MD'/identifier),action='md',
                                seed=[args.seed,mindex,index,generation,child],temperature=min(2400,600*1.5**generation)))
                            continue
                        path=root/method/f'g{generation}'/'proposals'/(identifier+'.vasp');path.parent.mkdir(parents=True,exist_ok=True)
                        try:
                            trial=mutate(read(start['pose']),reference,rng,.3 if method=='iid' else (.3,.45,.6,.8)[generation],collective=method=='beam',strain=.04 if method=='beam' else 0.)
                            write(path,trial,format='vasp',direct=True,sort=False)
                            proposals.append(dict(id=identifier,parent=parent,source=str(path)))
                        except ValueError as error:history.append(dict(method=method,parent=parent,id=identifier,status='proposal_rejected',error=str(error)))
                if md_tasks:
                    md_results=workers.request(md_tasks)
                    for task,row in zip(md_tasks,md_results):
                        parent=task['id'].split('-soft_md-')[0]+'.vasp'
                        history.append(dict(method=method,parent=parent,MD=row))
                        if row['status']=='escaped':proposals.append(dict(id=task['id'],parent=parent,source=row['source']))
                b_tasks=[dict(id=x['id'],source=x['source'],output=str(root/method/f'g{generation}'/'B'/x['id']),action='relax') for x in proposals]
                b_results=workers.request(b_tasks)
                for proposal,b in zip(proposals,b_results):
                    history.append(dict(method=method,parent=proposal['parent'],B=b))
                    if b['status']=='converged':
                        pnt=point(proposal['id'],proposal['source'],b,scorer,pools[proposal['parent']][0]['B']['volume_A3']);pools[proposal['parent']].append(pnt)
                chosen=[best(pools[c['parent_id']],base_energy[c['parent_id']]) for c in cases]
                save(root/'progress.json',dict(method=method,generation=generation+1,selected=chosen,CF=scorer.metadata,history=history))
                print(json.dumps(dict(method=method,round=generation+1,mean_error=float(np.mean([abs(c['gap']-2.5) for c in chosen])),selected_new=sum(c['id']!=b['id'] for c,b in zip(chosen,baseline)))),flush=True)
            all_points[method]=pools;selected[method]=chosen
            save(root/'search-results.json',dict(baseline=baseline,selected=selected,pools=all_points,history=history,CF=scorer.metadata,search_seed=args.seed))
        for method,points in selected.items():export(root,method,points,plan,scorer)
        save(root/'search.complete.json',dict(completed=True,parents=16,methods=list(selected),CF=scorer.metadata))
    finally:workers.close()


if __name__=='__main__':main()
