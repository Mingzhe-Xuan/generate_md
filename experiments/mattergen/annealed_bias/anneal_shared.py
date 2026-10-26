"""Unchanged native CF/physics interfaces and isolated legacy evaluation helpers."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
DIRECTED=HERE.parent/'directed_tto'
sys.path.insert(0,str(DIRECTED))
from cf_gradient import GradientScorer
from geometry import Geometry
from numerics import promote,score as precision_score
from probe import common_seed,code_hashes,load,save,sha
from runtime import Mechanics
sys.path.insert(0,str(HERE.parent/'denoised_tto'))
from denoised_shared import point,best,physical,validate_control,interval

__all__=['GradientScorer','Geometry','promote','precision_score','common_seed','code_hashes','load','save','sha','Mechanics','point','best','physical','validate_control','interval','implementation_hashes']


def implementation_hashes():
    result=code_hashes()
    for name in ('anneal_policy.py','anneal_math.py','anneal_fire.py','anneal_shared.py','anneal_objective.py','anneal_native.py'):
        result['annealed/'+name]=sha(HERE/name)
    for name in ('policy.py','search.py','scorer.py','attractors.py'):
        result['basin/'+name]=sha(HERE.parent/'basin_search'/name)
    result['isolated_helpers']=sha(HERE.parent/'denoised_tto/denoised_shared.py')
    return result
