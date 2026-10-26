"""Frozen source linkage and unchanged CF/MatterSim interfaces for the new method."""
from pathlib import Path
import sys
import numpy as np
DIRECTED=Path(__file__).resolve().parents[1]/'directed_tto'
sys.path.insert(0,str(DIRECTED))
from probe import load,save,sha,code_hashes,common_seed,close
from cf_gradient import GradientScorer
from geometry import Geometry
from runtime import Mechanics
from numerics import promote,score as precision_score
from directions import matched_random
from search import point
from policy import physical,best
from attractors import validate_control
from evaluation_export import export_evaluation
from validation_run import six_checks_pass
from guided_settings import PARENTS,POLICY
from guided_math import unit_or_zero,perturbations

__all__=['load','save','sha','code_hashes','common_seed','close','GradientScorer','Geometry','Mechanics','promote','precision_score',
    'matched_random','point','physical','best','validate_control','export_evaluation','six_checks_pass',
    'implementation_hashes','source_plan','validate_plan','utility','ray_directions','geometry','family_noise']

SOURCE_FILES=('plan.json','native-probe.json','direction-results.json','report.json')


def implementation_hashes():
    result=code_hashes()
    for name in ('guided_common.py','guided_settings.py','guided_math.py','guided_run.py','guided_validate.py'):result[name]=sha(Path(__file__).with_name(name))
    result['evaluation_export.py']=sha(DIRECTED/'evaluation_export.py')
    for name in ('scorer.py','search.py','policy.py','attractors.py'):result['shared_'+name]=sha(DIRECTED.parent/'basin_search'/name)
    return result


def source_plan(source):
    old=load(source/'plan.json');native=load(source/'native-probe.json');data=load(source/'direction-results.json')
    if tuple(old['parent_ids'])!=PARENTS or [c['parent_id'] for c in old['cases']]!=list(PARENTS) or not all(c['neutral'] and c['B']['status']=='converged' for c in old['cases']):raise ValueError('Original neutral pilot16 changed.')
    if not native['passed'] or native['code_sha256']!=code_hashes() or data['completed_parents']!=16:raise ValueError('Original native source incomplete/changed.')
    return dict(parent_ids=list(PARENTS),cases=old['cases'],policy=POLICY,source=str(source),source_hashes={n:sha(source/n) for n in SOURCE_FILES},
        native_math_sha256=code_hashes(),code_sha256=implementation_hashes(),role='New method development on original16; original48 results remain unchanged.')


def validate_plan(plan):
    if tuple(plan['parent_ids'])!=PARENTS or [c['parent_id'] for c in plan['cases']]!=list(PARENTS) or plan['policy']!=POLICY:raise ValueError('Frozen guided plan changed.')
    if plan['code_sha256']!=implementation_hashes() or plan['native_math_sha256']!=code_hashes():raise ValueError('Guided scientific source changed.')
    if any(sha(Path(plan['source'])/n)!=v for n,v in plan['source_hashes'].items()):raise ValueError('Original development evidence changed.')


def utility(q,energy):
    if q is None:return 1.
    if not np.isfinite([q['gap'],q['eform']]).all():raise ValueError('Nonfinite CF utility.')
    if not physical(q['B'],energy) or q['eform']>=-.5:return 1.
    error=abs(q['gap']-2.5)
    if not np.isfinite(error):raise ValueError('Nonfinite CF utility.')
    return float(error/(1+error))


def geometry(operator):
    g=operator['geometry'];return Geometry(g['numbers'],g['cell'],g['frac'])


def family_noise(operator,family):
    geom=geometry(operator);guide=operator['random_guide'] if family=='random_guided' else operator['cf_guide']
    return perturbations(guide,geom.project,operator['normal'],operator['scalar'],POLICY['sigma_RMS_A'],1. if family=='isotropic' else POLICY['guided_alpha'])


def ray_directions(gradient,cf_guide,geom,rng):
    descent=unit_or_zero(-geom.project(gradient))
    return dict(descent=descent,reverse=-descent,random=matched_random(descent,rng,geom.project),raw_cf=-np.asarray(cf_guide))
