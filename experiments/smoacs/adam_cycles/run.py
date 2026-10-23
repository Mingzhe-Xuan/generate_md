"""GPU Adam/MatterSim fixed-budget cycles from original four or128 templates."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from adam_policy import BASE_RATES,flags,freeze_inactive,set_rates,valid_proposal
from gpu import GPUAudit,prepare_cuda

BLOCK=Path(__file__).resolve().parents[1]/'block_alternating'
sys.path.insert(0,str(BLOCK))
spec=importlib.util.spec_from_file_location('block_initialization',BLOCK/'run.py')
block=importlib.util.module_from_spec(spec);spec.loader.exec_module(block)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cycles(context,hard_predictions,prepared_protocol,steps=200,cycles=3,schedule='per_phase'):
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from solver import gram_check
    from policy import geometry_change
    output=context['output'];args=context['args'];ids=context['ids'];n=len(ids)
    if args.resume or n!=(4 if args.pilot else 128) or args.max_rounds!=cycles or (steps,cycles) not in ((200,3),(600,1)):
        raise ValueError('Expected a fresh four- or128-template Adam200x3 or Adam600x1 experiment.')
    if schedule not in ('per_phase','restart200'):raise ValueError('Unknown LR schedule.')
    gpu_info=prepare_cuda(context)
    targets=context['targets'];optimizers=context['optimizers'];forward=context['forward'];slices=context['slices']
    micro=None
    if n>4:
        from microbatch import Microbatch
        micro=Microbatch(context,size=16)
        with torch.no_grad():
            a,b,_,_=forward(.01);x,y,_,_=micro.forward(.01)
        gpu_info['microbatch_forward_error']=max(float((a-x).abs().max()),float((b-y).abs().max()))
        if gpu_info['microbatch_forward_error']>1e-3:raise ValueError('Microbatch predictor equivalence failed.')
        active_probe=torch.ones(n,dtype=torch.bool,device=targets[0].device)
        _,g16=micro.gradients(.01,active_probe);_,g8=Microbatch(context,size=8).gradients(.01,active_probe)
        discrepancy=max(float((a-b).abs().max())/max(1e-6,float(a.abs().max()),float(b.abs().max())) for a,b in zip(g16,g8))
        gpu_info.update(microbatch_size=16,gradient_normalization=n,microbatch_gradient_error=discrepancy)
        if discrepancy>1e-3:raise ValueError('Microbatch gradient partition equivalence failed.')
        del g16,g8
        forward=micro.forward
    batch=context['batch'];base=context['base'];lower,upper,angle_lower,angle_upper=[context[k] for k in ('lower','upper','angle_lower','angle_upper')]
    protocol={k:prepared_protocol[k] for k in ('ids','initial_control_error','baseline_npz_sha256','author_source_sha256',
        'source_commit','a_parameters','a_dtype','precision_control_error','gaussian_derivative','adapter_control_error','adapter_control_calls')}
    protocol.update(method='Adam_cosine_then_MatterSim',cycles=cycles,steps_per_A=steps,
        a_base_rates=BASE_RATES,temperature=.01,lr_cycle_policy=schedule,lr_period=200 if schedule=='restart200' else steps,
        b_fmax=.01,b_maxstep=.05,b_max_steps=1600,denominator=n,
        comparison='Fixed-budget GPU experiment; no joint convergence claim.')
    protocol.update(gpu_info)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    active=torch.ones(n,dtype=torch.bool,device=targets[0].device);states=[dict(candidate_id=k,status='active') for k in ids]
    rounds=[];neutral_cache={};workers=[];logs=[]

    def points(out,folder):
        hg,he=hard_predictions(out);answer=[]
        for i,sl in enumerate(slices):
            structure=Structure(out['lattice_vectors'][i].detach().cpu().numpy(),
                (out['normalized_dist'][sl].argmax(1)+1).tolist(),out['normed_batch_dir_coords'][sl].detach().cpu().numpy())
            counts=Counter(s.specie.symbol for s in structure);key=tuple(sorted(counts.items()))
            if key not in neutral_cache:
                neutral,_,_=context['author'].elec_neutral_check_SUPER_COMMON(0,n,[e for e,c in key],[[c] for e,c in key],True)
                neutral_cache[key]=bool(neutral)
            separated=not any(neighbor.nn_distance<.5 for group in structure.get_all_neighbors(.5) for neighbor in group)
            destination=folder/Path(ids[i]).stem;destination.mkdir(parents=True,exist_ok=True)
            path=destination/'structure.vasp';structure.to(filename=str(path),fmt='poscar')
            gap=float(hg.reshape(-1)[i]);ef=float(he.reshape(-1)[i]);valid=neutral_cache[key] and separated
            answer.append(dict(candidate_id=ids[i],source=str(path),sha256=digest(path),num_atoms=len(structure),
                formula=structure.composition.formula,elements=sorted(counts),gap=gap,eform=ef,
                neutral=neutral_cache[key],separated=separated,valid=valid,flags=flags(gap,ef,valid),
                cell=structure.lattice.matrix.tolist(),frac=structure.frac_coords.tolist(),
                atomic_numbers=list(structure.atomic_numbers),volume=structure.volume))
        return answer

    def save(cycle,phase):
        payload=dict(protocol=protocol,cycles=rounds,states=states,current_cycle=cycle,phase=phase)
        path=output/'results.tmp';path.write_text(json.dumps(payload,indent=2)+'\n');os.replace(path,output/'results.json')
        torch.save(dict(targets=[p.detach() for p in targets],optimizers=[o.state_dict() for o in optimizers],
            bounds=[lower,upper,angle_lower,angle_upper],active=active,cycle=cycle,phase=phase),output/'checkpoint.tmp')
        os.replace(output/'checkpoint.tmp',output/'checkpoint.pt')

    with torch.no_grad():_,_,_,out=forward(.01);initial=points(out,output/'initial')
    previous=initial;save(0,'initial')
    audit=GPUAudit(context,output/'gpu-audit.json')
    try:
        for cycle in range(1,cycles+1):
            cycle_dir=output/('cycle-'+str(cycle));cycle_dir.mkdir()
            eligible=active.clone();updates=[0]*n;step_times=[];started=time.perf_counter()
            for step in range(steps):
                if not active.any():break
                step_started=time.perf_counter()
                set_rates(optimizers,step%200 if schedule=='restart200' else step,200 if schedule=='restart200' else steps)
                if micro is None:
                    _,_,loss,_=forward(.01)
                    if not torch.isfinite(loss[active]).all():raise ValueError('Nonfinite active A input; retain last checkpoint.')
                    gradients=torch.autograd.grad(loss[active].sum()/n,targets)
                else:loss,gradients=micro.gradients(.01,active)
                if not all(g.is_cuda for g in gradients):raise RuntimeError('Adam input gradients are not on CUDA.')
                masks=[active[:,None],active[:,None],active[batch,None],active[batch,None],active[:,None]]
                before=[p.detach().clone() for p in targets]
                for optimizer in optimizers:optimizer.zero_grad()
                for parameter,gradient,mask in zip(targets,gradients,masks):parameter.grad=torch.where(mask,gradient,0.)
                for optimizer in optimizers:optimizer.step()
                for optimizer in optimizers:freeze_inactive(optimizer,targets,before,masks)
                with torch.no_grad():_,_,after_loss,after=forward(.01)
                for i,sl in enumerate(slices):
                    if not active[i]:continue
                    finite=valid_proposal(after_loss[i],after['lattice_vectors'][i],after['normed_batch_dir_coords'][sl])
                    if not finite:
                        with torch.no_grad():
                            for p,b,selection in zip(targets,before,(i,i,sl,sl,i)):p[selection].copy_(b[selection])
                        active[i]=False;states[i].update(status='A_failed',error='Nonfinite or singular Adam proposal rolled back.',failed_cycle=cycle,failed_step=step+1)
                    else:updates[i]+=1
                masks=[active[:,None],active[:,None],active[batch,None],active[batch,None],active[:,None]]
                for optimizer in optimizers:freeze_inactive(optimizer,targets,before,masks)
                elapsed=time.perf_counter()-step_started;step_times.append(elapsed)
                if step==0 or (step+1)%25==0:
                    record=dict(cycle=cycle,A_step=step+1,active=int(active.sum()),loss=loss.detach().tolist(),
                                rates=[o.param_groups[0]['lr'] for o in optimizers],step_seconds=elapsed)
                    with (output/'A-trace.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
            set_rates(optimizers,200 if schedule=='restart200' else steps,200 if schedule=='restart200' else steps)
            with torch.no_grad():_,_,_,out=forward(.01);a_points=points(out,cycle_dir/'A')
            rows=[]
            for i in range(n):
                rows.append(dict(candidate_id=ids[i],cycle=cycle,A=a_points[i],a_updates=updates[i],eligible_at_start=bool(eligible[i]),
                    status='B_pending' if active[i] else 'not_eligible',reason=None if active[i] else dict(states[i]),
                    a_phase_seconds=time.perf_counter()-started,a_step_seconds=step_times))
            rounds.append(dict(cycle=cycle,candidates=rows));save(cycle,'B_pending')
            tasks=[dict(candidate_id=ids[i],source=rows[i]['A']['source'],output=str(cycle_dir/'B'/Path(ids[i]).stem)) for i in range(n) if active[i]]
            if tasks:
                torch.cuda.empty_cache()
                if not workers:
                    for wi in range(args.workers):
                        log=(output/('worker-'+str(wi)+'.err')).open('w');logs.append(log)
                        worker=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(block.OLD/'worker.py'),
                            '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').absolute())],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,
                            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),bufsize=1)
                        workers.append(worker)
                        if not block.legacy.read_response(worker.stdout,output/('worker-'+str(wi)+'.stdout'))['ready']:raise RuntimeError('B worker not ready.')
                for wi,worker in enumerate(workers):worker.stdin.write(json.dumps(dict(tasks=tasks[wi::len(workers)]))+'\n');worker.stdin.flush()
                returned=[]
                for wi,worker in enumerate(workers):returned.extend(block.legacy.read_response(worker.stdout,output/('worker-'+str(wi)+'.stdout'))['results'])
                records={r['candidate_id']:r for r in returned}
                if set(records)!={t['candidate_id'] for t in tasks}:raise ValueError('B response IDs differ.')
            else:records={}
            before_feedback=[p.detach().clone() for p in targets[:3]];accepted=[]
            for i,sl in enumerate(slices):
                if ids[i] not in records:continue
                record=records[ids[i]];rows[i]['b_result']=record
                if record['input_sha256']!=rows[i]['A']['sha256']:raise ValueError('B input hash differs from immutable A endpoint.')
                if record['status']!='converged':
                    active[i]=False;states[i].update(status='B_'+record['status'],failed_cycle=cycle)
                    rows[i]['status']=states[i]['status'];continue
                structure=Structure.from_file(record['relaxed_path'])
                if list(structure.atomic_numbers)!=rows[i]['A']['atomic_numbers']:raise ValueError('B changed ordered composition.')
                if not np.isfinite(structure.lattice.matrix).all() or structure.volume<=1e-6:
                    active[i]=False;states[i].update(status='feedback_failed',failed_cycle=cycle)
                    rows[i].update(status='B_converged_feedback_failed',feedback_error='Numerically singular B cell.');continue
                lengths=torch.tensor(structure.lattice.abc,dtype=targets[0].dtype,device=targets[0].device)
                angles=torch.tensor(structure.lattice.angles,dtype=targets[0].dtype,device=targets[0].device)
                with torch.no_grad():
                    lower[i]=torch.minimum(lower[i],lengths);upper[i]=torch.maximum(upper[i],lengths)
                    angle_lower[i]=torch.minimum(angle_lower[i],angles);angle_upper[i]=torch.maximum(angle_upper[i],angles)
                    targets[0][i].copy_(lengths/base[i]);targets[1][i].copy_((angles-60)/60)
                    targets[2][sl].copy_(torch.tensor(structure.frac_coords,dtype=targets[0].dtype,device=targets[0].device))
                accepted.append(i)
            with torch.no_grad():_,_,feedback_loss,out=forward(.01)
            for i in accepted:
                sl=slices[i];check=gram_check(out['lattice_vectors'][i].cpu().numpy(),Structure.from_file(records[ids[i]]['relaxed_path']).lattice.matrix)
                if not check['passed'] or not valid_proposal(feedback_loss[i],out['lattice_vectors'][i],out['normed_batch_dir_coords'][sl]):
                    with torch.no_grad():
                        for p,old,selection in zip(targets[:3],before_feedback,(i,i,sl)):p[selection].copy_(old[selection])
                    active[i]=False;states[i].update(status='feedback_failed',failed_cycle=cycle)
                    rows[i].update(status='B_converged_feedback_failed',feedback_error='B reconstruction/finite-value check failed.',feedback=check)
                else:rows[i]['feedback']=check
            with torch.no_grad():_,_,_,out=forward(.01);b_points=points(out,cycle_dir/'B-evaluation')
            for i in range(n):
                if not active[i]:continue
                check=rows[i]['feedback']
                rows[i].update(status='completed',B=b_points[i],feedback=check,
                    volume_ratio_to_initial=b_points[i]['volume']/initial[i]['volume'])
                rms,relative=geometry_change(previous[i]['cell'],previous[i]['frac'],b_points[i]['cell'],b_points[i]['frac'])
                rows[i].update(change_from_previous_B=dict(rms=rms,relative_cell=relative))
                previous[i]=b_points[i]
            save(cycle,'cycle_completed')
            print(json.dumps(dict(cycle_completed=cycle,statuses=dict(Counter(r['status'] for r in rows)))),flush=True)
        for state in states:
            if state['status']=='active':state['status']='fixed_cycles_completed_not_a_convergence_claim'
        save(cycles,'complete')
        audit.close()
        if not audit.report['passed']:raise RuntimeError('No actual CUDA forward/backward optimization was observed.')
        (output/'complete.json').write_text(json.dumps(dict(completed=True,cycles=cycles,denominator=n))+'\n')
    finally:
        audit.close()
        for worker in workers:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=20)
                except Exception:worker.terminate();worker.wait(timeout=10)
        for log in logs:log.close()


if __name__=='__main__':block.legacy.main(driver=lambda context:block.run_blocks(context,driver=run_cycles))
