"""Multisystem forward-property response through fixed-cell physical relaxation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from csp_model import sha
from physics_client import Physics
from property_model import Properties
from prepare_anchors import symmetric_cell
from pullback_math import rms_direction, response
from relations import centered


POLICY = dict(seed=20261053, target_gap_eV=2.5, probe_RMS_A=.03, amplitudes_RMS_A=[.005,.02,.05],
    directions=['random', 'gap_loss_descent', 'eform_descent'], tolerances_eV_A=[.001,.0002],
    anchor_fmax=.001, max_steps=800, FD_steps_A=[1e-4,5e-5], FD_abs=1e-5, FD_rel=.002,
    fixed_species=True, relaxation_cell='Fixed after full-cell anchor relaxation; no symmetry constraints.',
    property_order=['gap_eV','eform_eV_atom'], ratio_denominator_min=1e-5,
    distance='Ordered-site MIC with center-of-mass translation removed; not a StructureMatcher basin label.',
    interpretation='Finite-amplitude empirical pullback, not differentiation through BFGS or an optimization efficacy benchmark.')


def save(path, obj):
    def default(x):
        if isinstance(x, np.ndarray):return x.tolist()
        if isinstance(x, np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=default)+'\n')


def execute(root):
    from ase.geometry import find_mic
    if (root/'plan.json').exists():raise ValueError('Preserve previous attempt.')
    selection = json.loads((root/'selection.json').read_text())
    physics = None;started = time.perf_counter();cases = []
    try:
        properties = Properties(root);physics = Physics(root/'physics')
        files = [Path(__file__), Path(__file__).with_name('property_model.py'), Path(__file__).with_name('pullback_math.py'),
                 Path(__file__).with_name('physics_client.py'), Path(__file__).with_name('physics_worker.py')]
        plan = dict(policy=POLICY, selection=selection, selection_sha256=sha(root/'selection.json'),
                    commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    code_sha256={p.name:sha(p) for p in files}, properties=properties.metadata, physics=physics.metadata)
        save(root/'plan.json', plan)
        for index, selected in enumerate(selection['selected']):
            folder=root/selected['id'];folder.mkdir();tic=time.perf_counter();before=physics.NFE
            if sha(selected['source']) != selected['source_sha256']:raise ValueError('Source reference changed.')
            numbers=np.array(selected['numbers']);n=len(numbers);cell=np.array(selected['cell'])
            serial=[0]
            def task(coordinates, label, tolerance, fullcell=False):
                serial[0]+=1
                return dict(id=selected['id']+'-'+str(serial[0]), action='relax', numbers=numbers.tolist(),
                    cell=cell.tolist(), frac=(coordinates@np.linalg.inv(cell)).tolist(), fmax=tolerance,
                    max_steps=POLICY['max_steps'], cell_relax=fullcell, output=str(folder/label))
            anchor=physics.request([task(np.array(selected['frac'])@cell, 'anchor', POLICY['anchor_fmax'], True)])[0]
            case=dict(id=selected['id'], system=selected['system'], num_atoms=n, anchor=anchor, status='pending', pairs=[])
            if anchor['status'] != 'converged':
                case.update(status='anchor_'+anchor['status'], seconds=time.perf_counter()-tic, NFE=physics.NFE-before)
                save(folder/'result.json',case);cases.append(case);save(root/'progress.json',cases);continue
            cell=symmetric_cell(np.array(anchor['cell']));anchor_cart=np.array(anchor['frac'])@cell
            rng=np.random.default_rng([POLICY['seed'],index]);probe=anchor_cart+POLICY['probe_RMS_A']*rms_direction(rng.normal(size=(n,3)),n)
            original,original_graphs=properties.evaluate(numbers,cell,probe,gradient=True,native=True)
            directions=dict(random=rms_direction(rng.normal(size=(n,3)),n),
                gap_loss_descent=rms_direction(-(original['values'][0]-POLICY['target_gap_eV'])*original['gradients'][0],n),
                eform_descent=rms_direction(-original['gradients'][1],n))
            case.update(cell=cell, numbers=numbers, probe=probe, original=original,
                        direction_status={k:'zero_gradient' if d is None else 'active' for k,d in directions.items()})
            references={}
            for tolerance in POLICY['tolerances_eV_A']:
                label='reference-'+str(tolerance)
                relaxed=physics.request([task(probe,label,tolerance)])[0]
                reference=dict(physics=relaxed)
                if relaxed['status']=='converged':
                    cart=np.array(relaxed['frac'])@cell
                    reference['property'],_=properties.evaluate(numbers,cell,cart,gradient=True)
                references[str(tolerance)]=reference
            case['references']=references
            if any(r['physics']['status']!='converged' for r in references.values()):
                case.update(status='reference_unconverged',seconds=time.perf_counter()-tic,NFE=physics.NFE-before)
                save(folder/'result.json',case);cases.append(case);save(root/'progress.json',cases);continue
            controls=[]
            for name,direction in directions.items():
                if direction is None:continue
                unit=direction/np.sqrt(n)
                for step in POLICY['FD_steps_A']:
                    p,_=properties.evaluate(numbers,cell,probe+step*unit)
                    m,_=properties.evaluate(numbers,cell,probe-step*unit)
                    graph_ok=p['graph_hash']==m['graph_hash']==original['graph_hash']
                    expected=original['gradients'].reshape(2,-1)@unit.ravel()
                    finite=(p['values']-m['values'])/(2*step)
                    fixedp,_=properties.evaluate(numbers,cell,probe+step*unit,graphs=original_graphs)
                    fixedm,_=properties.evaluate(numbers,cell,probe-step*unit,graphs=original_graphs)
                    fixed_finite=(fixedp['values']-fixedm['values'])/(2*step)
                    fixed_passed=np.abs(expected-fixed_finite)<=POLICY['FD_abs']+POLICY['FD_rel']*np.maximum(np.abs(expected),np.abs(fixed_finite))
                    passed=np.abs(expected-finite)<=POLICY['FD_abs']+POLICY['FD_rel']*np.maximum(np.abs(expected),np.abs(finite))
                    controls.append(dict(direction=name,step_A=step,AD=expected,FD=finite,same_graph=graph_ok,
                                         passed=passed if graph_ok else None, fixed_graph_FD=fixed_finite,fixed_graph_passed=fixed_passed))
                for amplitude in POLICY['amplitudes_RMS_A']:
                    plus=probe+amplitude*direction;minus=probe-amplitude*direction
                    yp,_=properties.evaluate(numbers,cell,plus);ym,_=properties.evaluate(numbers,cell,minus)
                    for tolerance in POLICY['tolerances_eV_A']:
                        label=f'{name}-{amplitude}-{tolerance}'
                        endpoints=physics.request([task(plus,label+'-plus',tolerance),task(minus,label+'-minus',tolerance)])
                        pair=dict(direction=name,amplitude_RMS_A=amplitude,tolerance_eV_A=tolerance,
                                  raw_plus=yp,raw_minus=ym,physics=endpoints,status='pending')
                        if all(r['status']=='converged' for r in endpoints):
                            carts=[np.array(r['frac'])@cell for r in endpoints]
                            if any(not np.array_equal(r['numbers'],numbers) or not np.allclose(r['cell'],cell,atol=1e-10) for r in endpoints):
                                raise ValueError('Fixed species/cell invariant failed.')
                            evaluated=[properties.evaluate(numbers,cell,c)[0] for c in carts]
                            difference=find_mic(carts[0]-carts[1],cell,pbc=True)[0]
                            reference=references[str(tolerance)]
                            reference_cart=np.array(reference['physics']['frac'])@cell
                            distances=[float(np.sqrt(np.mean(np.sum(centered(find_mic(c-reference_cart,cell,pbc=True)[0]).reshape(n,3)**2,axis=1)))) for c in carts]
                            pair.update(status='completed',relaxed_plus=evaluated[0],relaxed_minus=evaluated[1],
                                endpoint_RMS_to_reference_A=distances,
                                response=response(yp['values'],ym['values'],evaluated[0]['values'],evaluated[1]['values'],
                                    original['gradients'],reference['property']['gradients'],direction,difference,amplitude))
                        else:pair['status']='relaxation_unconverged_or_failed'
                        case['pairs'].append(pair)
                    save(folder/'partial.json',case)
            case.update(status='completed',controls=controls,directions=directions,seconds=time.perf_counter()-tic,NFE=physics.NFE-before)
            save(folder/'result.json',case);cases.append(case);save(root/'progress.json',cases)
            print('PROPERTY_CASE',selected['id'],case['seconds'],case['NFE'],flush=True)
        save(root/'complete.json',dict(execution_complete=True,cases=cases,selected=len(selection['selected']),
            physics_NFE=physics.NFE,properties=properties.metadata,native_controls=properties.predictor.native_controls,
            seconds=time.perf_counter()-started))
    except Exception:
        save(root/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-started))
        raise
    finally:
        if physics is not None:physics.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    execute(args.run.resolve())


if __name__=='__main__':main()
