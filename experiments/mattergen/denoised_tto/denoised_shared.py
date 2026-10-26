"""Isolated access to unchanged legacy helpers with colliding module names."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'basin_search'


def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


basin_policy=_load(BASE/'policy.py','denoised_legacy_basin_policy')
previous_policy=sys.modules.get('policy')
previous_path=list(sys.path)
try:
    sys.modules['policy']=basin_policy
    sys.path.insert(0,str(BASE))
    basin_search=_load(BASE/'search.py','denoised_legacy_basin_search')
    controls=_load(BASE/'attractors.py','denoised_legacy_basin_controls')
finally:
    sys.path[:]=previous_path
    if previous_policy is None:
        sys.modules.pop('policy',None)
    else:
        sys.modules['policy']=previous_policy

directed_settings=_load(HERE.parent/'directed_tto/settings.py','denoised_legacy_directed_settings')
point=basin_search.point
best=basin_policy.best
physical=basin_policy.physical
validate_control=controls.validate_control
interval=directed_settings.interval
