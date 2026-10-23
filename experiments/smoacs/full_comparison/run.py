"""Run a full128-template GPU arm from the identical original initialization."""
import importlib.util
import os
from pathlib import Path
import sys

SHARED=Path(__file__).resolve().parents[1]/'adam_cycles'
sys.path.insert(0,str(SHARED))
spec=importlib.util.spec_from_file_location('full_shared_adam',SHARED/'run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)


if __name__=='__main__':
    arm=os.environ['SMOACS_FULL_ARM']
    if arm not in ('single','alternating','restart200'):raise ValueError('Unknown comparison arm.')
    steps,cycles=(200,3) if arm=='alternating' else (600,1)
    schedule='restart200' if arm=='restart200' else 'per_phase'
    shared.block.legacy.main(driver=lambda context:shared.block.run_blocks(context,
        driver=lambda c,h,p:shared.run_cycles(c,h,p,steps=steps,cycles=cycles,schedule=schedule)))
