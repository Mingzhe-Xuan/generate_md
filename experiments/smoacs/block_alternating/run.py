"""Small-sample A-to-stationarity/B-to-force-convergence alternating driver."""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from solver import gram_check,projected_gradient,solve

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
    forward=context['forward'];hard_predictions=context['hard_predictions'];slices=context['slices'];base=context['base']
    lower,upper,angle_lower,angle_upper=[context[k] for k in ('lower','upper','angle_lower','angle_upper')]
    protocol=dict(context['protocol'],method='A_LBFGSB_until_stationary_then_B',temperature=.01,
        anneal=None,a_maxiter=1200,a_maxfun=3000,a_projected_gradient_tolerance=.001,
        joint_patience=3,gram_rtol=2e-6,gram_atol=1e-5,
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
        return selections,parts,np.concatenate(lo),np.concatenate(hi)

    def objective(i,selections,parts):
        def evaluate(x):
            offset=0
            with torch.no_grad():
                for p,s,part in zip(targets,selections,parts):
                    count=len(part);p[s].copy_(torch.as_tensor(x[offset:offset+count],dtype=p.dtype).reshape(p[s].shape));offset+=count
            _,_,loss,_=forward(temperature)
            gradients=torch.autograd.grad(loss[i],targets)
            flat=np.concatenate([g[s].detach().numpy().astype(float).ravel() for g,s in zip(gradients,selections)])
            value=float(loss[i].detach())
            if not np.isfinite(value) or not np.isfinite(flat).all():raise ValueError('Nonfinite A objective/gradient.')
            return value,flat
        return evaluate

    def point(i):
        with torch.no_grad():_,_,_,out=forward(temperature);gap,ef=hard_predictions(out)
        return dict(cell=out['lattice_vectors'][i].numpy(),frac=out['normed_batch_dir_coords'][slices[i]].numpy(),
            numbers=(out['normalized_dist'][slices[i]].argmax(1)+1).tolist(),
            gap=float(gap.reshape(-1)[i]),eform=float(ef.reshape(-1)[i]),
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
                lengths=torch.tensor(structure.lattice.abc,dtype=torch.float32)
                angles=torch.tensor(structure.lattice.angles,dtype=torch.float32)
                with torch.no_grad():
                    lower[i]=torch.minimum(lower[i],lengths);upper[i]=torch.maximum(upper[i],lengths)
                    angle_lower[i]=torch.minimum(angle_lower[i],angles);angle_upper[i]=torch.maximum(angle_upper[i],angles)
                    targets[0][i].copy_(lengths/base[i]);targets[1][i].copy_((angles-60)/60)
                    targets[2][slices[i]].copy_(torch.tensor(structure.frac_coords,dtype=torch.float32))
                after_b=point(i);check=gram_check(structure.lattice.matrix,after_b['cell']);state['feedback_check']=check
                if not check['passed']:
                    state.update(status='feedback_failed',error='Scale-aware cell Gram check failed.');dump();continue
                selections,parts,lo,hi=variables(i);value,gradient=objective(i,selections,parts)(np.concatenate(parts))
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
                    a_geometry_rms=a_rms,a_cell_relative=a_cell,b_geometry_rms=b_rms,b_cell_relative=b_cell,
                    b_gap=after_b['gap'],b_eform=after_b['eform'],minimum_assignment_probability=after_b['minimum_assignment_probability'],
                    volume_ratio_to_initial=float(structure.volume/initial_volumes[i]))
                previous[i]=after_b
                if state['stable_rounds']>=3:
                    state.update(status='jointly_converged' if raw<=.001 else 'jointly_constrained_converged')
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
