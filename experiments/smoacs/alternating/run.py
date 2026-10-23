"""One Crystalformer Adam update followed by converged MatterSim relaxation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import types

from policy import schedule,stop_status,geometry_change

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Captured(BaseException):pass


def read_response(stream, diagnostic):
    for line in stream:
        prefix,found,body=line.partition('__AB_RPC__')
        if prefix:
            with Path(diagnostic).open('a') as log:log.write(prefix)
        if found:return json.loads(body)
    raise RuntimeError('B worker exited without a protocol response; inspect its stdout/stderr logs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--anneal',type=int,default=200)
    parser.add_argument('--max-rounds',type=int,default=300)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Submit through Slurm.')
    import numpy as np
    import torch
    from pymatgen.core import Structure
    torch.set_num_threads(4)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=args.resume)
    sys.path.insert(0,str(ROOT))
    from run_section21 import SOURCE_MEMBER,SOURCE_SHA256
    archive=Path('results/smoacs/downloads/reproducibility.tar.gz')
    assets=Path('results/smoacs/assets-199').resolve()
    manifest=json.loads((ROOT/'assets.json').read_text())
    if digest(archive)!=manifest['reproducibility.tar.gz']['sha256']:raise ValueError('Author archive changed.')
    for name,info in json.loads((ROOT/'checkpoints.json').read_text()).items():
        if digest(assets/'models/crystalformer'/name)!=info['sha256']:raise ValueError('Predictor changed.')
    with tarfile.open(archive) as tar:original=tar.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA256:raise ValueError('Author source changed.')
    os.environ.update(SMOACS_ROOT=str(assets),SMOACS_DEVICE='cpu',MPLBACKEND='Agg')
    source=Path('third_party/SMOACS').resolve();sys.path[:0]=[str(source),str(source/'src')]
    author=types.ModuleType('alternating_original_author');author.__file__=str(assets/SOURCE_MEMBER)
    sys.modules[author.__name__]=author
    exec(compile(original,author.__file__,'exec',dont_inherit=True),author.__dict__)
    captured={}
    def intercept(**kwargs):
        captured.update(kwargs)
        raise Captured()
    author.optimize_solution_Crystalformer=intercept
    try:
        author.main_experiment(num_steps=200,num_candidate=128,num_batch_crystal=128,
            bandgap_margin=.04,target_bandgap=2.5,atom_lr=.007,lattice_lr=.002,coords_lr=.005,
            limit_coords_displacement=None,initialization='Struct-AtomicMask',method='grad',
            optimize_mode='pytorch_optimizer___steps_cm-None',seed=42,
            exp_name='capture-'+os.environ['SLURM_JOB_ID'],output_root=str(output))
    except Captured:pass
    if not captured:raise RuntimeError('Original initialization was not captured.')
    data=captured['mini_batch_inputs_dict'];original_scalers=captured['scalers']
    all_ids=[Path(str(v)).name for v in data['fnames']]
    baseline_path=Path('results/smoacs/optimization/section21-Struct-AtomicMask-grad-206/result.npz')
    with np.load(baseline_path,allow_pickle=False) as z:
        baseline_ids=z['original_fnames'].tolist()
        baseline_gap=dict(zip(baseline_ids,z['gap_history'][0]))
        baseline_ef=dict(zip(baseline_ids,z['ef_history'][0]))
    if set(all_ids)!=set(baseline_ids) or len(all_ids)!=128:raise ValueError('Original template IDs changed.')
    sizes=data['size'].long();total=int(sizes.sum())
    chosen=list(range(128))
    if args.pilot:
        chosen=[]
        for target in (2,4,8,10):
            available=[i for i in range(128) if i not in chosen]
            chosen.append(min(available,key=lambda i:(abs(int(sizes[i])-target),all_ids[i])))
        chosen.sort()
    atom_indices=[];offset=0
    for i,n in enumerate(sizes.tolist()):
        if i in chosen:atom_indices.extend(range(offset,offset+n))
        offset+=n
    selected={}
    for key,value in data.items():
        if torch.is_tensor(value) and value.ndim:
            selected[key]=value[chosen].clone() if value.shape[0]==128 else value[atom_indices].clone() if value.shape[0]==total else value
        else:selected[key]=value
    data=selected;ids=[all_ids[i] for i in chosen];size=data['size'].long();n=len(ids)
    batch=torch.repeat_interleave(torch.arange(n),size);data['batch']=batch
    slices=[];offset=0
    for count in size.tolist():slices.append(slice(offset,offset+count));offset+=count
    base=original_scalers['abc_scaler'].base_length[chosen].clone()
    class LinearLengths:
        def rescale(self,value):return value*base
    class LinearAngles:
        def rescale(self,value):return value*60+60
    scalers=dict(abc_scaler=LinearLengths(),angle_scaler=LinearAngles())
    targets=[torch.nn.Parameter(v.clone().float()) for v in (
        data['batch_abc']/base,(data['batch_angle']-60)/60,data['batch_dir_coords'],
        data['atomic_distribution'],data['ox_mask_learnable_tensor_per_crystal'])]
    lower=torch.full((n,3),2.);upper=torch.full((n,3),10.)
    angle_lower=torch.full((n,3),60.);angle_upper=torch.full((n,3),120.)
    gap_model,ef_model=captured['bandgap_model'],captured['e_form_model']
    for model in (gap_model,ef_model):
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
    optimizers=[torch.optim.Adam(targets[:2],lr=.002),torch.optim.Adam(targets[3:],lr=.007),torch.optim.Adam([targets[2]],lr=.005)]
    def forward(temperature):
        gap,ef,out=author.forward_propagation_Crystalformer(optimization_targets=targets,fixed_inputs=data,
            scalers=scalers,temperature=temperature,bandgap_model=gap_model,e_form_model=ef_model,
            adding_noise_scale=None,limit_coords_displacement=None,
            abc_range=(lower,upper),angle_range=(angle_lower,angle_upper))
        gap,ef=gap.reshape(-1),ef.reshape(-1)
        return gap,ef,torch.clamp(torch.abs(gap-2.5)-.04,min=0)+ef,out
    def hard_predictions(out):
        onehot=torch.nn.functional.one_hot(out['normalized_dist'].argmax(1),98).float()
        return author.Crystalformer_prediction(gap_model,ef_model,onehot,out['pos'],batch,out['lattice_vectors'],size)
    with torch.no_grad():g,e,_,_=forward(.01)
    error=max(max(abs(float(g[i])-float(baseline_gap[k])),abs(float(e[i])-float(baseline_ef[k]))) for i,k in enumerate(ids))
    if error>1e-3:raise ValueError('Original initialization control failed: '+str(error))
    protocol=dict(pilot=args.pilot,anneal=args.anneal,max_rounds=args.max_rounds,ids=ids,
        initial_control_error=error,baseline_npz_sha256=digest(baseline_path),author_source_sha256=SOURCE_SHA256,
        a_parameters=['scaled lengths','scaled angles','fractional coordinates','element logits','oxidation logits'],
        a_rates=[.002,.007,.005],lr_after_anneal_fraction=.1,b_fmax=.01,b_max_steps=400,
        joint_gradient_max=.001,joint_update_max=.001,geometry_rms_A=.001,cell_relative=1e-4,
        property_delta=.001,patience=5,cycle_patience=10,
        boundary_policy='Expand original cell bounds only as necessary to contain actual B geometry; preserve B without clipping.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    states=[dict(candidate_id=k,status='active',a_steps=0,stable_rounds=0) for k in ids]
    previous=[None]*n;last=None;iteration=0;phase='A_ready';tasks=[];info={}
    checkpoint=output/'checkpoint.pt'
    if args.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['ids']!=ids or saved['anneal']!=args.anneal:raise ValueError('Resume protocol mismatch.')
        with torch.no_grad():
            for p,v in zip(targets,saved['targets']):p.copy_(v)
            for p,v in zip((lower,upper,angle_lower,angle_upper),saved['bounds']):p.copy_(v)
        for opt,value in zip(optimizers,saved['optimizers']):opt.load_state_dict(value)
        states,previous,last,iteration,phase,tasks,info=[saved[k] for k in ('states','previous','last','iteration','phase','tasks','info')]
    def save():
        payload=dict(ids=ids,anneal=args.anneal,targets=[p.detach() for p in targets],
            bounds=[lower,upper,angle_lower,angle_upper],optimizers=[o.state_dict() for o in optimizers],
            states=states,previous=previous,last=last,iteration=iteration,phase=phase,tasks=tasks,info=info)
        torch.save(payload,output/'checkpoint.tmp');os.replace(output/'checkpoint.tmp',checkpoint)
        (output/'results.json').write_text(json.dumps(dict(protocol=protocol,candidates=states),indent=2)+'\n')
    workers=[];handles=[]
    worker_script=Path(__file__).with_name('worker.py').resolve()
    try:
        for i in range(args.workers):
            log=(output/('worker-'+str(i)+'-'+os.environ['SLURM_JOB_ID']+'.err')).open('w');handles.append(log)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            workers.append(subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(worker_script),
                '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve())],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env,bufsize=1))
        ready=[]
        for index,w in enumerate(workers):
            ready.append(read_response(w.stdout,output/('worker-'+str(index)+'.stdout')))
        if not all(v['ready'] for v in ready):raise RuntimeError('B worker initialization failed.')
        (output/'workers.json').write_text(json.dumps(ready,indent=2)+'\n')
        while any(s['status']=='active' for s in states):
            temperature,fraction=schedule(iteration,args.anneal)
            if phase=='A_ready':
                active=torch.tensor([s['status']=='active' for s in states])
                started=time.perf_counter();gap,ef,loss,out=forward(temperature)
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite A input; preserve checkpoint for diagnosis.')
                gradients=torch.autograd.grad((loss*active).sum(),targets)
                residual=[];atom_residual=[]
                for i,sl in enumerate(slices):
                    blocks=[gradients[0][i],gradients[1][i],gradients[2][sl],gradients[3][sl],gradients[4][i]]
                    residual.append(max(float(v.abs().max()) for v in blocks))
                    atom_residual.append(max(float(blocks[j].abs().max()) for j in (3,4)))
                    if states[i]['status']=='active' and previous[i] is not None:
                        status=stop_status(residual[i],atom_residual[i],states[i].get('a_step_max',1.),
                            states[i]['stable_rounds'],temperature==.0001 and states[i].get('last_temperature')==temperature,
                            states[i].get('b_converged',False),atom_step=states[i].get('atom_step_max',1.),
                            a_rms=states[i].get('a_geometry_rms',1.),a_cell=states[i].get('a_cell_relative',1.))
                        if status:states[i].update(status=status,a_gradient_residual=residual[i])
                if iteration>=args.max_rounds:
                    for i,state in enumerate(states):
                        if state['status']=='active':state.update(status='outer_limit',a_gradient_residual=residual[i])
                    break
                active=torch.tensor([s['status']=='active' for s in states])
                if not active.any():break
                masks=[active[:,None],active[:,None],active[batch,None],active[batch,None],active[:,None]]
                before=[p.detach().clone() for p in targets]
                before_cell=out['lattice_vectors'].detach().clone()
                before_frac=out['normed_batch_dir_coords'].detach().clone()
                for opt,rate in zip(optimizers,(.002,.007,.005)):
                    opt.zero_grad();opt.param_groups[0]['lr']=rate*fraction
                for p,g,m in zip(targets,gradients,masks):p.grad=g*m/n
                for opt in optimizers:opt.step()
                with torch.no_grad():
                    for p,b,m in zip(targets,before,masks):p.copy_(torch.where(m,p,b))
                    _,_,_,out=forward(temperature);hg,he=hard_predictions(out)
                if not all(torch.isfinite(v).all() for v in (hg,he,out['lattice_vectors'])):
                    raise ValueError('Nonfinite A proposal; no invalid B job submitted.')
                iteration+=1;tasks=[];info={}
                for i,sl in enumerate(slices):
                    if states[i]['status']!='active':continue
                    folder=output/('round-'+str(iteration).zfill(4))/Path(ids[i]).stem;folder.mkdir(parents=True,exist_ok=True)
                    numbers=(out['normalized_dist'][sl].argmax(1)+1).tolist()
                    structure=Structure(out['lattice_vectors'][i].detach().numpy(),numbers,out['normed_batch_dir_coords'][sl].detach().numpy())
                    structure.to(filename=str(folder/'A.vasp'),fmt='poscar')
                    tasks.append(dict(candidate_id=ids[i],source=str(folder/'A.vasp'),output=str(folder/'B')))
                    increments=[targets[0][i]-before[0][i],targets[1][i]-before[1][i],targets[2][sl]-before[2][sl],
                                targets[3][sl]-before[3][sl],targets[4][i]-before[4][i]]
                    info[ids[i]]=dict(a_gradient_residual=residual[i],atom_gradient_residual=atom_residual[i],
                        a_step_max=max(float(v.detach().abs().max()) for v in increments),
                        atom_step_max=max(float(v.detach().abs().max()) for v in increments[3:]),
                        a_gap=float(hg.reshape(-1)[i]),a_eform=float(he.reshape(-1)[i]),temperature=temperature,
                        a_batch_seconds=time.perf_counter()-started)
                    rms,relative=geometry_change(before_cell[i].numpy(),before_frac[sl].numpy(),
                        out['lattice_vectors'][i].detach().numpy(),out['normed_batch_dir_coords'][sl].detach().numpy())
                    info[ids[i]].update(a_geometry_rms=rms,a_cell_relative=relative)
                    states[i].update(info[ids[i]],a_steps=iteration,a_source=str(folder/'A.vasp'))
                phase='B_pending';save()
            groups=[tasks[i::len(workers)] for i in range(len(workers))]
            for w,group in zip(workers,groups):w.stdin.write(json.dumps(dict(tasks=group))+'\n');w.stdin.flush()
            returned=[]
            for index,w in enumerate(workers):
                returned.extend(read_response(w.stdout,output/('worker-'+str(index)+'.stdout'))['results'])
            results={r['candidate_id']:r for r in returned}
            if set(results)!={t['candidate_id'] for t in tasks}:raise ValueError('B response IDs mismatch.')
            current={}
            with torch.no_grad():
                for i,sl in enumerate(slices):
                    if ids[i] not in results:continue
                    record=results[ids[i]];states[i]['b_result']=record
                    if record['status']!='converged':
                        states[i].update(status='B_'+record['status'],b_converged=False);continue
                    structure=Structure.from_file(record['relaxed_path'])
                    proposal=Structure.from_file(states[i]['a_source'])
                    if list(structure.atomic_numbers)!=list(proposal.atomic_numbers):raise ValueError('B atom identity changed.')
                    lengths=torch.tensor(structure.lattice.abc,dtype=torch.float32)
                    angles=torch.tensor(structure.lattice.angles,dtype=torch.float32)
                    expanded=bool(((lengths<lower[i])|(lengths>upper[i])|(angles<angle_lower[i])|(angles>angle_upper[i])).any())
                    lower[i]=torch.minimum(lower[i],lengths);upper[i]=torch.maximum(upper[i],lengths)
                    angle_lower[i]=torch.minimum(angle_lower[i],angles);angle_upper[i]=torch.maximum(angle_upper[i],angles)
                    targets[0][i].copy_(lengths/base[i]);targets[1][i].copy_((angles-60)/60)
                    targets[2][sl].copy_(torch.tensor(structure.frac_coords,dtype=torch.float32))
                    states[i]['boundary_expanded']=states[i].get('boundary_expanded',False) or expanded
                    current[ids[i]]=structure
                _,_,_,feedback=forward(info[tasks[0]['candidate_id']]['temperature'])
                bg,be=hard_predictions(feedback)
            for i,sl in enumerate(slices):
                if ids[i] not in current:continue
                structure=current[ids[i]];cell=feedback['lattice_vectors'][i].detach().numpy()
                gram_error=float(np.max(np.abs(cell@cell.T-structure.lattice.matrix@structure.lattice.matrix.T)))
                if gram_error>1e-3:raise ValueError('B geometry was clipped/distorted during A feedback.')
                frac=feedback['normed_batch_dir_coords'][sl].detach().numpy()
                numbers=(feedback['normalized_dist'][sl].argmax(1)+1).tolist()
                if numbers!=list(structure.atomic_numbers):raise ValueError('B feedback changed composition.')
                point=dict(cell=cell.tolist(),frac=frac.tolist(),numbers=numbers,gap=float(bg.reshape(-1)[i]),eform=float(be.reshape(-1)[i]))
                stable=False
                if previous[i] is not None and point['numbers']==previous[i]['numbers']:
                    rms,relative=geometry_change(previous[i]['cell'],previous[i]['frac'],point['cell'],point['frac'])
                    stable=rms<=.001 and relative<=1e-4 and abs(point['gap']-previous[i]['gap'])<=.001 and abs(point['eform']-previous[i]['eform'])<=.001
                temperature=info[ids[i]]['temperature']
                count=states[i]['stable_rounds']+1 if stable and temperature==.0001 and states[i].get('last_temperature')==temperature else 0
                states[i].update(info[ids[i]],stable_rounds=count,last_temperature=temperature,b_converged=True,
                    b_gap=point['gap'],b_eform=point['eform'],feedback_gram_error=gram_error)
                previous[i]=point
                with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(dict(round=iteration,**states[i]))+'\n')
            last=results;phase='A_ready';save()
            print(json.dumps(dict(round=iteration,active=sum(s['status']=='active' for s in states),
                                  statuses={k:sum(s['status']==k for s in states) for k in {s['status'] for s in states}})),flush=True)
        save()
        (output/'complete.json').write_text(json.dumps(dict(completed=True,pilot=args.pilot,initial_control_passed=error<=1e-3,
            denominator=n,rounds=iteration,all_records=len(states)))+'\n')
    finally:
        for w in workers:
            if w.poll() is None:
                try:w.stdin.write('{"stop":true}\n');w.stdin.flush();w.wait(timeout=20)
                except Exception:w.terminate();w.wait(timeout=10)
        for h in handles:h.close()


if __name__=='__main__':main()
