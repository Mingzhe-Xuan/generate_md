"""Audit saved post-quench endpoints and attempted-budget scaling, without models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
from policy import ACCEPTANCE,METHODS,ROUNDS,WIDTH,best,physical
from report import audit_attempts,interval

PREFIXES=(0,1,4,8,12,16)


def attempt_number(identifier,parent,method):
    match=re.fullmatch(re.escape(Path(parent).stem+'-'+method)+r'-g(\d{2})-c(\d+)',identifier)
    if not match:raise ValueError('Candidate ID does not belong to this parent/method.')
    generation,child=map(int,match.groups())
    if generation>=ROUNDS or child>=WIDTH:raise ValueError('Candidate index exceeds frozen budget.')
    return generation*WIDTH+child+1


def prefix_pool(points,parent,method,n):
    if not 0<=n<=ROUNDS*WIDTH:raise ValueError('Invalid attempted prefix.')
    if points[0]['id']!=parent:raise ValueError('Baseline fallback is missing.')
    if len({p['id'] for p in points})!=len(points):raise ValueError('Duplicate candidate.')
    return [points[0]]+[p for p in points[1:] if attempt_number(p['id'],parent,method)<=n]


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inspect(root):
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_best_n'))
    from effect_geometry import measure
    root=root.resolve();report=load(root/'report.json');plan=report['plan'];data=load(root/'search-results.json')
    parents=plan['parent_ids']
    if plan['acceptance']!=ACCEPTANCE or len(parents)!=16 or len(set(parents))!=16:raise ValueError('Frozen criteria/cohort changed.')
    if [c['parent_id'] for c in plan['cases']]!=parents or plan!=load(root/'plan.json'):raise ValueError('Frozen plan differs.')
    if not load(root/'native-controls.json')['passed'] or not load(root/'search.complete.json')['completed']:raise ValueError('Incomplete native run.')
    if data['search_seed']!=report['search_seed']:raise ValueError('Seed provenance differs.')
    ledger=audit_attempts(data['history'],parents)
    matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=True)
    curves={};geometry_cache={};verified=set()
    for method in METHODS:
        series=[]
        for n in PREFIXES:
            rows=[]
            for case in plan['cases']:
                parent=case['parent_id'];base=case['B'];base_energy=base['final_energy_eV']/base['num_atoms']
                selected=best(prefix_pool(data['pools'][method][parent],parent,method,n),base_energy)
                b=selected['B'];key=(base['relaxed_sha256'],b['relaxed_sha256'])
                for path,expected in ((base['relaxed_path'],key[0]),(selected['pose'],key[1])):
                    if (path,expected) not in verified:
                        if sha(path)!=expected:raise ValueError('Saved endpoint hash changed.')
                        verified.add((path,expected))
                if key not in geometry_cache:
                    first=Structure.from_file(base['relaxed_path']);second=Structure.from_file(selected['pose'])
                    geometry_cache[key]=dict(metrics=measure(first,second),author_match=bool(matcher.fit(second,first)))
                geo=geometry_cache[key];old=case['predictions']['after'];before=abs(old['crystalformer_gap']-2.5);after=abs(selected['gap']-2.5)
                valid=physical(b,base_energy) and selected['eform']<ACCEPTANCE['CF_eform_upper']
                changed=not geo['author_match'] and geo['metrics']['species_pair_distance_RMS_A']>=ACCEPTANCE['min_pair_distance_RMS_A']
                rows.append(dict(parent_id=parent,point_id=selected['id'],pose=selected['pose'],pose_sha256=key[1],physical=valid,
                    CF_error_before=before,CF_error_after=after,CF_error_gain=before-after,geometry=geo['metrics'],author_match=geo['author_match'],
                    material_changed=changed,material_CF_improved=bool(valid and changed and before-after>=ACCEPTANCE['min_CF_error_improvement_eV'])))
            costs=[stages for (m,identifier),stages in ledger.items() if m==method and attempt_number(identifier,identifier.split('-'+method+'-')[0]+'.vasp',method)<=n]
            row=dict(attempts_per_parent=n,denominator=16,CF_MAE=float(np.mean([r['CF_error_after'] for r in rows])),
                CF_MAE_delta_CI95=interval([r['CF_error_after']-r['CF_error_before'] for r in rows]),
                material_changed=sum(r['material_changed'] for r in rows),material_CF_improved=sum(r['material_CF_improved'] for r in rows),
                MatterSim_NFE=sum(event.get('NFE',0) for stages in costs for event in stages.values()),
                mechanics_seconds=sum(event.get('elapsed_seconds',0) for stages in costs for event in stages.values()),cases=rows)
            if n==ROUNDS*WIDTH:
                final=report['methods'][method];summary=final['summary']
                if abs(row['CF_MAE']-summary['crystalformer']['MAE_after'])>1e-5 or row['material_CF_improved']!=summary['material_CF_improved']:raise ValueError('Prefix16 does not reproduce final independent report.')
                if [r['point_id'] for r in rows]!=[r['point_id'] for r in final['cases']]:raise ValueError('Final selection differs.')
                if row['MatterSim_NFE']!=summary['cost']['MatterSim_NFE']:raise ValueError('Mechanics cost accounting differs.')
            series.append(row)
        curves[method]=series
    return dict(root=str(root),search_seed=data['search_seed'],parent_ids=parents,source_inventory_sha256=plan['source_inventory_sha256'],
        acceptance=plan['acceptance'],methods=curves,final_summaries={m:r['summary'] for m,r in report['methods'].items()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Batch geometry requires Slurm.')
    audits=[inspect(root) for root in args.run]
    if len({a['search_seed'] for a in audits})!=len(audits):raise ValueError('Independent runs need distinct seeds.')
    if any(a['parent_ids']!=audits[0]['parent_ids'] or a['source_inventory_sha256']!=audits[0]['source_inventory_sha256'] for a in audits):raise ValueError('Cohorts differ.')
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    result=dict(runs=audits,note='Read-only attempted-budget diagnostics. All16 parents retained; failed attempts consume N. CF ranks; ALIGNN is held out. NFE and mechanics time exclude common controls and include failed MD/quench attempts. Prefix CIs are descriptive; frozen independent-seed endpoint criteria determine confirmation.')
    (output/'scaling.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(iid='#6b7280',beam='#2563eb',soft_md='#d97706')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for i,audit in enumerate(audits):
        for method,series in audit['methods'].items():
            options=dict(color=colors[method],linestyle='-' if i==0 else '--',marker='o',markersize=4,label=f'{method}, seed {audit["search_seed"]}')
            axes[0].plot(PREFIXES,[r['CF_MAE'] for r in series],**options)
            axes[1].plot(PREFIXES,[r['material_CF_improved'] for r in series],**options)
            axes[2].plot([r['MatterSim_NFE']/16 for r in series],[r['CF_MAE'] for r in series],**options)
    axes[0].set(xlabel='Proposal attempts per parent',ylabel='Post-quench CF gap MAE (eV)')
    axes[1].set(xlabel='Proposal attempts per parent',ylabel='Materially changed + improved parents /16',ylim=(-.2,16.2))
    axes[1].axhline(4,color='#94a3b8',linestyle=':')
    axes[2].set(xlabel='MatterSim force evaluations per parent',ylabel='Post-quench CF gap MAE (eV)')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Frozen 16-parent post-quench search: attempted budget and actual mechanics cost')
    fig.savefig(output/'scaling.png',dpi=180);fig.savefig(output/'scaling.pdf');plt.close(fig)
    print(json.dumps(dict(output=str(output),seeds=[a['search_seed'] for a in audits],verified_final_endpoints=True)),flush=True)


if __name__=='__main__':main()
