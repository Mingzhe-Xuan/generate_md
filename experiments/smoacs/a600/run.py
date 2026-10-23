"""Run one600-update GPU Adam phase followed by MatterSim relaxation."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]/'adam_cycles';sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('shared_adam_cycles',ROOT/'run.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)


if __name__=='__main__':
    schedule=os.environ.get('SMOACS_A600_SCHEDULE','per_phase')
    shared.block.legacy.main(driver=lambda context:shared.block.run_blocks(context,
        driver=lambda c,h,p:shared.run_cycles(c,h,p,steps=600,cycles=1,schedule=schedule)))
