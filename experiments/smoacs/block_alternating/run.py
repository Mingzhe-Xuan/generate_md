"""Small-sample A-to-stationarity/B-to-force-convergence alternating driver."""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from solver import gram_check,hinge_subgradient,projected_gradient,solve
from gaussian import ExactGaussianAverage

OLD=Path(__file__).resolve().parents[1]/'alternating'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('alternating_initialization',OLD/'run.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)


def run_blocks(context):
    import numpy as np
    import torch
    from pymatgen.core import Structure
    from policy import geometry_change
    args=context['args'];output=context['output'];ids=context['ids'];targets=context['targets']
    if not args.pilot or len(ids)!=4:raise ValueError('This feasibility experiment is restricted to the four-template pilot.')
    forward=context['forward'];slices=context['slices'];base=context['base']
    with torch.no_grad():original_gap,original_ef,_,_=forward(.01)
    for parameter in targets:parameter.data=parameter.data.double()
    for model in (context['gap_model'],context['ef_model']):model.double()
    for key,value in context['data'].items():
        if torch.is_tensor(value) and value.is_floating_point():context['data'][key]=value.double()
    with torch.no_grad():double_gap,double_ef,_,_=forward(.01)
    precision_error=max(float((double_gap-original_gap).abs().max()),float((double_ef-original_ef).abs().max()))
    if precision_error>1e-3:raise ValueError('Float64 forward differs from pinned float32 control: '+str(precision_error))
    from models import indexed_lattice_multi_head_attention as attention
    attention.AverageGaussiansBySoftmaxXYFunc_=ExactGaussianAverage
    with torch.no_grad():exact_gap,exact_ef,_,_=forward(.01)
    adapter_error=max(float((exact_gap-double_gap).abs().max()),float((exact_ef-double_ef).abs().max()))
    if adapter_error>1e-8:raise ValueError('Gaussian derivative adapter changed predictor values: '+str(adapter_error))
    def hard_predictions(out):
        onehot=torch.nn.functional.one_hot(out['normalized_dist'].argmax(1),98).to(targets[0].dtype)
        return context['author'].Crystalformer_prediction(context['gap_model'],context['ef_model'],onehot,
            out['pos'],context['batch'],out['lattice_vectors'],context['size'])
    lower,upper,angle_lower,angle_upper=[context[k] for k in ('lower','upper','angle_lower','angle_upper')]
    protocol=dict(context['protocol'],method='A_LBFGSB_until_stationary_then_B',temperature=.01,
        anneal=None,a_maxiter=1200,a_maxfun=3000,a_projected_gradient_tolerance=.001,
        joint_patience=3,gram_rtol=2e-6,gram_atol=1e-5,a_dtype='float64',precision_control_error=precision_error,
        elemental_oxidation_lower_bound=1e-6,
        gaussian_derivative='ordinary_autograd_including_distance_basis',adapter_control_error=adapter_error,
        hinge_subgradient_boundary_eV=1e-7,
        note='Fixed-T feasibility method, not original Adam/annealing. Constrained and raw stationarity reported separately.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    states=[dict(candidate_id=k,status='active',a_phases=0,stable_rounds=0) for k in ids]
    previous=[None]*len(ids);temperature=.01
    with torch.no_grad():_,_,_,initial=forward(temperature)
    initial_volumes=[float(abs(np.linalg.det(v))) for v in initial['lattice_vectors'].numpy()]
    worker=None;log=None;round_index=0

    def dump():
        path=output/'results.tmp'
        path.write_text(json.dumps(dict(protocol=protocol,round=round_index,candidates=states),indent=2)+'\n')
        os.replace(path,output/'results.json')

    def selection(i):return [i,i,slices[i],slices[i],i]

    def variables(i):
        selections=selection(i);parts=[p[s].detach().numpy().astype(float).ravel() for p,s in zip(targets,selections)]
        lo=[(lower[i]/base[i]).numpy(),((angle_lower[i]-60)/60).numpy()]
        hi=[(upper[i]/base[i]).numpy(),((angle_upper[i]-60)/60).numpy()]
        lo=[v.astype(float).ravel() for v in lo]+[np.full(len(v),-np.inf) for v in parts[2:]]
        hi=[v.astype(float).ravel() for v in hi]+[np.full(len(v),np.inf) for v in parts[2:]]
        lo[3][:]=1e-6;lo[4][:]=1e-6
        return selections,parts,np.concatenate(lo),np.concatenate(hi)

    def objective(i,selections,parts):
        def evaluate(x):
            offset=0
            with torch.no_grad():
                for p,s,part in zip(targets,selections,parts):
                    count=len(part);p[s].copy_(torch.as_tensor(x[offset:offset+count],dtype=p.dtype).reshape(p[s].shape));offset+=count
            gap,ef,loss,_=forward(temperature)
            at_hinge=abs(abs(float(gap[i].detach())-2.5)-.04)<=1e-7
            gradients=torch.autograd.grad(loss[i],targets,retain_graph=at_hinge)
            flat=np.concatenate([g[s].detach().numpy().astype(float).ravel() for g,s in zip(gradients,selections)])
            evaluate.raw_autograd_gradient=float(np.max(np.abs(flat)))
            evaluate.hinge_coefficient=None
            if at_hinge:
                gap_grad=torch.autograd.grad(gap[i],targets,retain_graph=True)
                ef_grad=torch.autograd.grad(ef[i],targets)
                gap_flat=np.concatenate([g[s].detach().numpy().ravel() for g,s in zip(gap_grad,selections)])
                ef_flat=np.concatenate([g[s].detach().numpy().ravel() for g,s in zip(ef_grad,selections)])
                _,_,lo,hi=variables(i)
                flat,evaluate.hinge_coefficient=hinge_subgradient(x,float(gap[i].detach()),gap_flat,ef_flat,lo,hi)
            value=float(loss[i].detach())
            if not np.isfinite(value) or not np.isfinite(flat).all():raise ValueError('Nonfinite A objective/gradient.')
            return value,flat
        return evaluate

    def point(i):
        with torch.no_grad():soft_gap,soft_ef,_,out=forward(temperature);gap,ef=hard_predictions(out)
        return dict(cell=out['lattice_vectors'][i].numpy(),frac=out['normed_batch_dir_coords'][slices[i]].numpy(),
            numbers=(out['normalized_dist'][slices[i]].argmax(1)+1).tolist(),
            gap=float(gap.reshape(-1)[i]),eform=float(ef.reshape(-1)[i]),
            continuous_gap=float(soft_gap[i]),continuous_eform=float(soft_ef[i]),
            minimum_assignment_probability=float(out['normalized_dist'][slices[i]].max(1).values.min()))

    try:
        for round_index in range(1,args.max_rounds+1):
            if not any(r['status']=='active' for r in states):break
            for i,state in enumerate(states):
                if state['status']!='active':continue
                folder=output/('round-'+str(round_index).zfill(4))/Path(ids[i]).stem;folder.mkdir(parents=True)
                before=point(i);selections,parts,lo,hi=variables(i);evaluate=objective(i,selections,parts)
                evaluations=0;started=time.perf_counter()
                def traced(x):
                    nonlocal evaluations
                    value,gradient=evaluate(x);evaluations+=1
                    if evaluations==1 or evaluations%20==0:
                        record=dict(round=round_index,candidate_id=ids[i],evaluation=evaluations,loss=value,
                            projected_gradient=float(np.max(np.abs(projected_gradient(x,gradient,lo,hi)))))
                        with (output/'A-trace.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                        print(json.dumps(record),flush=True)
                    return value,gradient
                try:
                    x,report=solve(traced,np.concatenate(parts),lo,hi)
                    evaluate(x)
                except Exception as error:
                    state.update(status='A_failed',error=repr(error));dump();continue
                report.update(elapsed_seconds=time.perf_counter()-started,actual_evaluations=evaluations)
                state.update(a_report=report,a_phases=round_index)
                (folder/'A-result.json').write_text(json.dumps(report,indent=2)+'\n')
                after_a=point(i)
                report.update(continuous_gap=after_a['continuous_gap'],continuous_eform=after_a['continuous_eform'],
                              hinge_boundary_distance=abs(abs(after_a['continuous_gap']-2.5)-.04))
                final_parts=[p[s].detach().numpy().astype(float).ravel() for p,s in zip(targets,selections)]
                final_x=np.concatenate(final_parts);final_value,final_gradient=evaluate(final_x)
                report.update(raw_autograd_gradient=evaluate.raw_autograd_gradient,hinge_coefficient=evaluate.hinge_coefficient)
                pg=projected_gradient(final_x,final_gradient,lo,hi);offset=0;blocks=[]
                for part in final_parts:
                    end=offset+len(part)
                    blocks.append(dict(raw=float(np.abs(final_gradient[offset:end]).max()),
                                       projected=float(np.abs(pg[offset:end]).max())))
                    offset=end
                report['gradient_blocks_lengths_angles_coords_elements_oxidation']=blocks
                if not report['converged']:
                    direction=-pg/max(float(np.abs(pg).max()),1e-12);probes=[]
                    for step in (1e-3,1e-4,1e-5,1e-6,1e-7):
                        trial=np.clip(final_x+step*direction,lo,hi)
                        try:trial_value,_=evaluate(trial);probes.append(dict(step=step,loss_change=trial_value-final_value))
                        except ValueError:probes.append(dict(step=step,status='invalid_trial'))
                    evaluate(final_x)
                    report['finite_descent_probe']=probes
                torch.save(dict(targets=[p[s].detach().clone() for p,s in zip(targets,selections)],
                                lower=lo.tolist(),upper=hi.tolist(),candidate_id=ids[i]),folder/'A-state.pt')
                (folder/'A-result.json').write_text(json.dumps(report,indent=2)+'\n')
                Structure(after_a['cell'],after_a['numbers'],after_a['frac']).to(filename=str(folder/'A.vasp'),fmt='poscar')
                state.update(a_source=str(folder/'A.vasp'),a_gap=after_a['gap'],a_eform=after_a['eform'])
                if not report['converged']:
                    state['status']='A_unconverged';dump();continue
                if worker is None:
                    log=(output/'worker.err').open('w')
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    worker=subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'),str(OLD/'worker.py'),
                        '--checkpoint',str(Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').absolute())],
                        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env,bufsize=1)
                    ready=legacy.read_response(worker.stdout,output/'worker.stdout')
                    if not ready['ready']:raise RuntimeError('B not ready.')
                task=dict(candidate_id=ids[i],source=str(folder/'A.vasp'),output=str(folder/'B'))
                worker.stdin.write(json.dumps(dict(tasks=[task]))+'\n');worker.stdin.flush()
                result=legacy.read_response(worker.stdout,output/'worker.stdout')['results'][0]
                state['b_result']=result
                if result['status']!='converged':
                    state['status']='B_'+result['status'];dump();continue
                structure=Structure.from_file(result['relaxed_path'])
                if list(structure.atomic_numbers)!=after_a['numbers']:
                    state.update(status='feedback_failed',error='B changed ordered atomic numbers.');dump();continue
                lengths=torch.tensor(structure.lattice.abc,dtype=targets[0].dtype)
                angles=torch.tensor(structure.lattice.angles,dtype=targets[0].dtype)
                with torch.no_grad():
                    lower[i]=torch.minimum(lower[i],lengths);upper[i]=torch.maximum(upper[i],lengths)
                    angle_lower[i]=torch.minimum(angle_lower[i],angles);angle_upper[i]=torch.maximum(angle_upper[i],angles)
                    targets[0][i].copy_(lengths/base[i]);targets[1][i].copy_((angles-60)/60)
                    targets[2][slices[i]].copy_(torch.tensor(structure.frac_coords,dtype=targets[0].dtype))
                after_b=point(i);check=gram_check(structure.lattice.matrix,after_b['cell']);state['feedback_check']=check
                if not check['passed']:
                    state.update(status='feedback_failed',error='Scale-aware cell Gram check failed.');dump();continue
                selections,parts,lo,hi=variables(i);endpoint=objective(i,selections,parts)
                value,gradient=endpoint(np.concatenate(parts))
                residual=float(np.max(np.abs(projected_gradient(np.concatenate(parts),gradient,lo,hi))))
                raw=float(np.max(np.abs(gradient)))
                a_rms,a_cell=geometry_change(before['cell'],before['frac'],after_a['cell'],after_a['frac'])
                b_rms,b_cell=geometry_change(after_a['cell'],after_a['frac'],after_b['cell'],after_b['frac'])
                stable=False
                if previous[i] is not None:
                    old=previous[i];rms,relative=geometry_change(old['cell'],old['frac'],after_b['cell'],after_b['frac'])
                    stable=(old['numbers']==after_b['numbers'] and rms<=.001 and relative<=1e-4
                        and abs(old['gap']-after_b['gap'])<=.001 and abs(old['eform']-after_b['eform'])<=.001)
                qualifies=(stable and before['numbers']==after_a['numbers']==after_b['numbers'] and residual<=.001
                    and a_rms<=.001 and a_cell<=1e-4 and b_rms<=.001 and b_cell<=1e-4
                    and result['max_filter_force']<=.01)
                state.update(stable_rounds=state['stable_rounds']+1 if qualifies else 0,
                    b_endpoint_a_projected_gradient=residual,b_endpoint_a_raw_gradient=raw,
                    b_endpoint_a_autograd_gradient=endpoint.raw_autograd_gradient,
                    b_endpoint_hinge_coefficient=endpoint.hinge_coefficient,
                    a_geometry_rms=a_rms,a_cell_relative=a_cell,b_geometry_rms=b_rms,b_cell_relative=b_cell,
                    b_gap=after_b['gap'],b_eform=after_b['eform'],minimum_assignment_probability=after_b['minimum_assignment_probability'],
                    volume_ratio_to_initial=float(structure.volume/initial_volumes[i]))
                previous[i]=after_b
                if state['stable_rounds']>=3:
                    state.update(status='jointly_converged' if endpoint.raw_autograd_gradient<=.001
                                 else 'jointly_constrained_or_hinge_stationary')
                with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(dict(round=round_index,**state))+'\n')
                print(json.dumps(dict(round=round_index,**state)),flush=True);dump()
            dump()
        for state in states:
            if state['status']=='active':state['status']='outer_limit'
        dump()
        (output/'complete.json').write_text(json.dumps(dict(completed=True,denominator=len(ids),statuses=dict(Counter(s['status'] for s in states))))+'\n')
    finally:
        if worker is not None and worker.poll() is None:
            try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=20)
            except Exception:worker.terminate();worker.wait(timeout=10)
        if log is not None:log.close()


if __name__=='__main__':legacy.main(driver=run_blocks)
