"""MatterGen -> Adam200 with F2 + lambda F1 -> unchanged MatterSim relaxation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from client import ForceClient
from probe import code_hashes,run as native_probe
from provider import MixedGradient

MATTERGEN=Path(__file__).resolve().parents[1]
SMOACS=MATTERGEN.parent/'smoacs'
sys.path[:0]=[str(MATTERGEN/'cf_templates'),str(SMOACS/'adam_cycles'),str(SMOACS/'block_alternating')]
from context import create


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weight',type=float,choices=(.1,.2,.3),required=True)
    parser.add_argument('--cohort',choices=('pilot','neutral','unmasked'),default='pilot')
    parser.add_argument('--physics-probe',type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    if args.cohort!='pilot' and not args.physics_probe:
        parser.error('Full cohorts require a valid native physics gradient probe.')
    output=args.output.resolve()
    if not output.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Use current task results.')
    args.fixed_species=True
    args.pilot=args.cohort=='pilot'
    args.max_rounds=1
    args.microbatch_size=4
    args.workers=2
    args.resume=False
    output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('force_mixed_adam',SMOACS/'adam_cycles/run.py')
    shared=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    context=create(args.source.resolve(),output,args)
    client=ForceClient(output,workers=2)
    try:
        if args.physics_probe:
            proof=json.loads(args.physics_probe.read_text())
            if not proof['passed'] or proof['code_sha256']!=code_hashes() or proof['checkpoint_sha256']!=client.ready[0]['checkpoint_sha256']:
                raise ValueError('Native physical-gradient probe does not match current code/model.')
            (output/'physics-gradient-probe.json').write_text(json.dumps(dict(proof,source_report=str(args.physics_probe.resolve())),indent=2)+'\n')
        else:
            native_probe(context,client)
        provider=MixedGradient(context,client,args.weight)
        context.update(gradient_transform=provider,after_A=provider.finish)
        context['protocol']['force_mixing']=dict(weight=args.weight,
            objective='MatterSim total energy + lambda*(relu(abs(CFgap-2.5)-0.04)+CF_Ef)',
            descent_direction='F2 + lambda F1; F1 is negative CF loss gradient, F2 is negative total-energy gradient',
            normalization='No per-atom/RMS/force normalization; both terms receive identical original cohort mean before Adam.',
            cell_gradient='At fixed fractional coordinates: V*H^{-T}*ASE_stress; dE/ds=-forces*H^T.',
            optimizer='Mixed gradients are input to unchanged Adam200 cosine; not a separate direct Euler force step.',
            physical_constraints_during_A=False,final_B_profile_unchanged=True,
            force_checkpoint_sha256=client.ready[0]['checkpoint_sha256'],
            property_gradient='Full original bandgap+formation-energy objective, not bandgap alone.',
            A_trace_loss='CF property loss; force-trace separately records total energy and mixed loss.')
        shared.block.run_blocks(context,driver=lambda c,h,p:shared.run_cycles(c,h,p,steps=200,cycles=1))
        original=context['data']['fixed_atomic_distribution']
        if not torch.equal(context['targets'][3],original):
            raise ValueError('Fixed species changed.')
        data=json.loads((output/'results.json').read_text())
        for row,sl in zip(data['cycles'][0]['candidates'],context['slices']):
            if row['A']['atomic_numbers']!=(original[sl].argmax(1)+1).tolist():
                raise ValueError('Ordered elements changed at endpoint.')
        (output/'fixed-species-audit.json').write_text(json.dumps(dict(passed=True,denominator=len(context['ids'])))+'\n')
        (output/'torch-memory.json').write_text(json.dumps(dict(max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(),pid=os.getpid(),cohort=args.cohort))+'\n')
    finally:
        client.close()


if __name__=='__main__':
    main()
