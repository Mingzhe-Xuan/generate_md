"""MatterGen initial templates through the GPU Adam200/600 then B controller."""
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
    parser.add_argument('--cohort', choices=('pilot', 'neutral', 'unmasked'), default='pilot')
    parser.add_argument('--microbatch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, choices=(200, 600), default=600)
    parser.add_argument('--fixed-species', action='store_true')
    parser.add_argument('--relax-every-step', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit native optimization through Slurm.')
    if args.microbatch_size < 2:
        parser.error('Use at least two structures per microbatch for partition checks.')
    if args.relax_every_step and (not args.fixed_species or args.cohort!='pilot' or args.steps!=200):
        parser.error('Per-step relaxation is a fixed-species four-template200-update pilot.')
    args.pilot, args.max_rounds, args.workers, args.resume = args.cohort == 'pilot', 1, 2, False
    if args.relax_every_step:
        args.max_rounds=200
    output = args.output.resolve()
    if not output.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Use the task results/mattergen directory.')
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(SMOACS/'adam_cycles'))
    spec = importlib.util.spec_from_file_location('mattergen_shared_adam', SMOACS/'adam_cycles/run.py')
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    context = create(args.source.resolve(), output, args)
    shared.block.run_blocks(context, driver=lambda c, h, p: shared.run_cycles(c, h, p,
        steps=1 if args.relax_every_step else args.steps, cycles=200 if args.relax_every_step else 1,
        schedule='global' if args.relax_every_step else 'per_phase'))
    import json
    import torch
    if args.fixed_species:
        original = context['data']['fixed_atomic_distribution'].argmax(1)+1
        if not torch.equal(context['targets'][3], context['data']['fixed_atomic_distribution']):
            raise ValueError('Fixed elemental interface tensor changed during optimization.')
        data=json.loads((output/'results.json').read_text())
        for cycle in data['cycles']:
            for row, atoms in zip(cycle['candidates'],context['slices']):
                if row['A']['atomic_numbers']!=original[atoms].tolist():
                    raise ValueError('An intermediate A endpoint changed original elements.')
        rows = data['cycles'][-1]['candidates']
        cases = []
        for row, atoms in zip(rows, context['slices']):
            expected = original[atoms].tolist()
            if row['A']['atomic_numbers'] != expected:
                raise ValueError('Optimized endpoint changed original per-site elements.')
            cases.append(dict(candidate_id=row['candidate_id'], original_atomic_numbers=expected,
                              final_atomic_numbers=row['A']['atomic_numbers'], A_sha256=row['A']['sha256']))
        (output/'fixed-species-audit.json').write_text(json.dumps(dict(passed=True, cases=cases, cycles_checked=len(data['cycles']),
            note='Exact onehot is passed to every model forward; elemental and oxidation gradients are zero.'), indent=2)+'\n')
    (output/'torch-memory.json').write_text(json.dumps(dict(
        max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved(),
        allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
        pid=os.getpid(), cohort=args.cohort), indent=2)+'\n')


if __name__ == '__main__':
    main()
