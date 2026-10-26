"""All-query accounting and recomputation of gradient estimates and selections."""
import numpy as np
from guided_settings import PARENTS,POLICY,FAMILIES,ARMS,SCALES,SEED,OUTPUTS
from guided_math import estimate,projected_normal_draws
from guided_common import geometry,family_noise,ray_directions,utility,best,common_seed,six_checks_pass


def attempt_key(row):
    if row['kind']=='probe':return (row['parent_id'],row['family'],'probe',row['pair'],row['side'])
    if row['kind']=='ray':return (row['parent_id'],row['family'],'ray',row['arm'],row['scale'])
    raise ValueError('Unknown guided attempt kind.')


def validate_coverage(parents,attempts):
    expected={(p,f,'probe',k,s) for p in parents for f in FAMILIES for k in range(POLICY['pairs']) for s in ('plus','minus')}
    expected|={(p,f,'ray',a,s) for p in parents for f in FAMILIES for a in ARMS for s in SCALES}
    if tuple(parents)!=PARENTS or len(attempts)!=1152 or {attempt_key(r) for r in attempts}!=expected or len({r['id'] for r in attempts})!=1152:
        raise ValueError('Missing/duplicate guided1152-attempt coverage.')


def validate_study(plan,proof,data):
    parents=plan['parent_ids'];validate_coverage(parents,data['attempts'])
    if not proof['passed'] or [r['parent_id'] for r in proof['operators']]!=parents or not all(r['passed'] for r in proof['operators']):raise ValueError('Incomplete native16 evidence.')
    if not all(six_checks_pass(r['derivative_checks']) for r in proof['operators'][:4]):raise ValueError('Incomplete native finite differences.')
    if data['completed_parents']!=16 or data['baseline']!=proof['baseline'] or [r['parent_id'] for r in data['seeds']]!=parents:raise ValueError('Guided denominator/baseline changed.')
    if set(data['selected'])!=set(OUTPUTS) or any(len(q)!=16 for q in data['selected'].values()) or any(len(data['estimates'][f])!=16 for f in FAMILIES):raise ValueError('Selected/estimate coverage changed.')
    if data['NFE_native']!=proof['NFE'] or data['NFE_total']!=data['NFE_native']+data['NFE_search'] or data['NFE_search']!=sum(r.get('B',{}).get('NFE',0) for r in data['attempts']+data['seeds']):raise ValueError('Scientific query cost ledger changed.')
    for i,(parent,case,operator,base) in enumerate(zip(parents,plan['cases'],proof['operators'],data['baseline'])):
        geom=geometry(operator);energy=base['B']['final_energy_eV']/base['B']['num_atoms'];seed=data['seeds'][i];qseed=np.array(operator['qseed']);index=int(parent.split('-')[1].split('.')[0])
        normal,scalar=projected_normal_draws(geom.project,geom.size,POLICY['pairs'],np.random.default_rng([SEED,index,330]));expected_seed,_=common_seed(geom,index,SEED)
        if not np.allclose(operator['normal'],normal,atol=1e-12) or not np.allclose(operator['scalar'],scalar,atol=1e-12) or not np.allclose(qseed,expected_seed,atol=1e-12):raise ValueError('Pre-outcome random draws changed.')
        if base['B']['relaxed_sha256']!=case['B']['relaxed_sha256'] or base['B']['final_energy_eV']!=case['B']['final_energy_eV'] or base['id']!=parent or data['selected']['baseline'][i]!=base:raise ValueError('Immutable original reference changed.')
        if not np.allclose(seed['q'],qseed,atol=1e-12) or seed['utility']!=utility(seed.get('point'),energy):raise ValueError('Shared seed control changed.')
        rows=[r for r in data['attempts'] if r['parent_id']==parent]
        for r in rows+[seed]:
            q=r.get('point')
            if r['utility']!=utility(q,energy):raise ValueError('Post-quench loss changed.')
            if q and (q['B']['input_sha256']!=r['input_sha256'] or abs(q['B']['volume_ratio']-q['B']['volume_A3']/base['B']['volume_A3'])>1e-6):raise ValueError('Input/original volume reference changed.')
        for fi,family in enumerate(FAMILIES):
            entry=data['estimates'][family][i];noise,meta=family_noise(operator,family);probes=[r for r in rows if r['family']==family and r['kind']=='probe']
            bykey={(r['pair'],r['side']):r for r in probes};plus=[bykey[k,'plus']['utility'] for k in range(POLICY['pairs'])];minus=[bykey[k,'minus']['utility'] for k in range(POLICY['pairs'])]
            g=estimate(noise,plus,minus,POLICY['sigma_RMS_A'],geom.project);directions=ray_directions(g,operator['cf_guide'],geom,np.random.default_rng([SEED,index,fi,445]))
            if entry['parent_id']!=parent or not np.allclose(entry['noise'],noise,atol=1e-12) or not np.allclose(entry['gradient'],g,atol=1e-12) or entry['plus']!=plus or entry['minus']!=minus:
                raise ValueError('Antithetic gradient/input pairs changed.')
            if any(entry['sampling'][k]!=meta[k] for k in ('alpha','effective_dimension','stationary_guide')) or not np.allclose(entry['sampling']['guide'],meta['guide'],atol=1e-12):raise ValueError('Guiding covariance changed.')
            for k in range(POLICY['pairs']):
                for side,sign in (('plus',1),('minus',-1)):
                    if not np.allclose(bykey[k,side]['q'],qseed+sign*noise[k],atol=1e-12):raise ValueError('Probe perturbation changed.')
            pool=[base]+[r['point'] for r in [seed]+probes if 'point' in r];control=best(pool,energy)
            if entry['probe_control']!=control or data['selected'][family+'-probes'][i]!=control:raise ValueError('Common probe-best fallback changed.')
            for arm in ARMS:
                if not np.allclose(entry['directions'][arm],directions[arm],atol=1e-12):raise ValueError('Heldout direction changed.')
                rays=[r for r in rows if r['family']==family and r['kind']=='ray' and r['arm']==arm]
                if any(not np.allclose(r['q'],qseed+r['scale']*directions[arm],atol=1e-12) for r in rays):raise ValueError('Heldout step changed.')
                if data['selected'][family+'-'+arm][i]!=best(pool+[r['point'] for r in rays if 'point' in r],energy):raise ValueError('Heldout common-pool selection changed.')
