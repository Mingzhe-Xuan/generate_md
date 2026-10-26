"""Reconstruct complete budgets, controls, saved poses and common-pool selection."""
from pathlib import Path
import numpy as np
from adapter import sha
from mapping import LatentGeometry
from denoised_policy import POLICY, PARENTS, TIMES, ARMS, OUTPUTS
from denoised_math import controlled_directions, validate_coverage, feasible_pose


def validate(root,plan,native,data):
    from pymatgen.core import Structure
    from denoised_search import core_hashes,validate_native,load
    from denoised_shared import best
    from geometry import Geometry
    if plan['policy']!=POLICY or tuple(plan['parent_ids'])!=PARENTS or plan['core_sha256']!=core_hashes() or native['core_sha256']!=core_hashes():
        raise ValueError('Frozen denoised scientific protocol changed.')
    if native['plan_sha256']!=sha(root/'plan.json') or sha(Path(plan['native_root'])/'native.json')!=plan['native_sha256']:
        raise ValueError('Native proof links changed.')
    validate_native(Path(plan['source']),Path(plan['native_root']))
    if plan['cases']!=load(Path(plan['source'])/'plan.json')['cases']:
        raise ValueError('Original parent-case metadata changed.')
    if (not native['passed'] or [q['parent_id'] for q in native['controls']]!=list(PARENTS)
        or any(q['CF_error']>1e-5 or not q['physical']['passed'] for q in native['controls'])):
        raise ValueError('Missing/failed16 pre-proposal controls.')
    expected=[(p,t) for p in PARENTS for t in TIMES]
    if [(q['parent_id'],q['time']) for q in native['operators']]!=expected or data['completed_parents']!=16:
        raise ValueError('Original16/operator coverage changed.')
    validate_coverage(data['attempts'])
    operators={(r['parent_id'],r['time']):r for r in native['operators']}
    for (parent,time),operator in operators.items():
        if operator['status']!='ready':
            if operator['status'] not in ('invalid_noisy_cell','invalid_decoded_seed'):
                raise ValueError('Unknown operator failure.')
            continue
        geom=LatentGeometry(**operator['latent']);index=int(Path(parent).stem.split('-')[1])
        expected_directions=controlled_directions(operator['gradient'],operator['stop_gradient'],operator['direct_gradient'],geom.project,
            np.random.default_rng([POLICY['seed'],index,int(time*1000),731]))
        for arm in ARMS:
            if not np.allclose(expected_directions[arm],operator['directions'][arm],rtol=0,atol=1e-12):
                raise ValueError('Paired/ablation/random direction changed.')
        if sha(operator['seed_input'])!=operator['seed_sha256']:
            raise ValueError('Common decoded seed changed.')
    for row in data['attempts']:
        operator=operators[row['parent_id'],row['time']]
        if operator['status']!='ready':
            if row['status']!=operator['status'] or 'B' in row or 'point' in row:
                raise ValueError('Invalid operator silently changed budget/status.')
            continue
        if 'proposal_q' in row:
            expected_q=np.zeros(len(operator['gradient'])) if row['kind']=='seed' else row['scale']*np.asarray(operator['directions'][row['arm']])
            if not np.allclose(row['proposal_q'],expected_q,rtol=0,atol=1e-12):
                raise ValueError('Actual proposal update changed.')
            reference=operator['reference'];cell=np.asarray(row['proposal_cell']);frac=np.asarray(row['proposal_frac'])
            feasible=feasible_pose(reference['numbers'],cell,frac,np.asarray(reference['cell']),np.asarray(reference['frac']))
            if feasible==(row['status']=='geometric_rejection'):
                raise ValueError('Proposal geometric guard differs.')
            if row.get('arm')=='direct':
                geom=Geometry(reference['numbers'],np.asarray(operator['clean']['clean_cell'])[0],operator['clean']['clean_frac'])
                c,f=geom.pose(row['proposal_q'])
                if not np.allclose(c,cell,atol=1e-12,rtol=0) or not np.allclose(f,frac,atol=1e-12,rtol=0):
                    raise ValueError('Direct CF comparator pose changed.')
        if 'input' in row:
            if sha(row['input'])!=row['input_sha256'] or row['B']['input_sha256']!=row['input_sha256']:
                raise ValueError('Attempt input SHA changed.')
            s=Structure.from_file(row['input'])
            if tuple(s.atomic_numbers)!=tuple(operator['reference']['numbers']) or not np.allclose(s.lattice.matrix,row['proposal_cell'],atol=1e-12,rtol=0):
                raise ValueError('Saved proposal cell/species differs.')
            delta=np.asarray(s.frac_coords)-row['proposal_frac'];delta-=np.rint(delta)
            if np.abs(delta).max()>1e-12:
                raise ValueError('Saved proposal positions differ.')
        if 'point' in row:
            p=row['point']
            if p['id']!=row['id'] or sha(p['pose'])!=p['B']['relaxed_sha256']:
                raise ValueError('Attempt terminal linkage changed.')
    if set(data['selected'])!=set(OUTPUTS) or any(len(v)!=16 for v in data['selected'].values()):
        raise ValueError('Final all16 output coverage differs.')
    for i,parent in enumerate(PARENTS):
        base=native['baseline'][i];energy=base['B']['final_energy_eV']/base['B']['num_atoms']
        case=plan['cases'][i]
        if base['id']!=parent or sha(base['pose'])!=case['B']['relaxed_sha256'] or any(base['B'].get(k)!=v for k,v in case['B'].items() if k not in ('volume_A3','volume_ratio','separated')):
            raise ValueError('Reused original baseline changed.')
        rows=[r for r in data['attempts'] if r['parent_id']==parent]
        pool=[base]+[r['point'] for r in rows if r['kind']=='seed' and 'point' in r]
        expected=dict(baseline=base,seeds=best(pool,energy))
        expected.update({a:best(pool+[r['point'] for r in rows if r.get('arm')==a and 'point' in r],energy) for a in ARMS})
        if any(data['selected'][a][i]!=q for a,q in expected.items()):
            raise ValueError('Shared-pool or matched-arm selection changed.')
    if data['NFE_native']!=native['NFE'] or data['NFE_search']!=sum(r.get('B',{}).get('NFE',0) for r in data['attempts']) or data['NFE_total']!=data['NFE_native']+data['NFE_search']:
        raise ValueError('Actual physics NFE ledger differs.')
