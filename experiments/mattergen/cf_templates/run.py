"""Four MatterGen initial templates through existing GPU A600/B controller."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

from context import create, SMOACS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit native optimization through Slurm.')
    args.pilot, args.max_rounds, args.workers, args.resume = True, 1, 2, False
    output = args.output.resolve()
    if not output.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Use the task results/mattergen directory.')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SMOACS/'adam_cycles'))
    spec = importlib.util.spec_from_file_location('mattergen_shared_adam', SMOACS/'adam_cycles/run.py')
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    context = create(args.source.resolve(), output, args)
    shared.block.run_blocks(context, driver=lambda c, h, p: shared.run_cycles(c, h, p, steps=600, cycles=1))


if __name__ == '__main__':
    main()
