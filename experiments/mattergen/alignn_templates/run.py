"""Fixed-composition MatterGen -> ALIGNN Adam200 -> identical MatterSim B."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from model import Predictor,load_models
from probe import run as gradient_probe,source_hashes

ROOT=Path(__file__).resolve().parents[1]
SMOACS=ROOT.parent/'smoacs'
sys.path[:0]=[str(ROOT/'cf_templates'),str(SMOACS/'adam_cycles'),str(SMOACS/'block_alternating')]
from context import create
from microbatch import Microbatch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cohort',choices=('pilot','neutral','unmasked'),default='pilot')
    parser.add_argument('--probe-report',type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run optimization and probes through Slurm.')
    output=args.output.resolve()
    if not output.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Use task results.')
    if args.cohort!='pilot' and not args.probe_report:
        parser.error('Full runs require the successful pilot gradient probe.')
    args.fixed_species=True
    args.pilot=args.cohort=='pilot'
    args.microbatch_size=4
    args.max_rounds=1
    args.workers=2
    args.resume=False
    output.mkdir(parents=True,exist_ok=False)
    context=create(args.source.resolve(),output,args)
    original_cf_models=(context['gap_model'],context['ef_model'])
    cf_hard_predict=context['hard_predict']
    context['template_initialization']['optimization_predictor']='ALIGNN'
    (output/'crystalformer-initialization-control.json').write_text(json.dumps(context['template_initialization'],indent=2)+'\n')
    models,metadata=load_models(output)
    predictor=Predictor(models)
    context.update(gap_model=models[0],ef_model=models[1],optimization_predictor='ALIGNN',
        backend='Frozen ALIGNN; signed periodic geometry autograd; kNN rebuilt each forward')
    context['author'].forward_propagation_Crystalformer=predictor.forward

    def forward(temperature):
        micro=Microbatch(context,size=4)
        return micro.forward(temperature) if len(context['ids'])>4 else micro.evaluate(0,len(context['ids']),temperature)

    context['forward']=forward
    if args.cohort=='pilot':
        probe=gradient_probe(context,predictor)
    else:
        probe=json.loads(args.probe_report.read_text())
        if not probe['passed'] or probe['adapter_source_sha256']!=source_hashes() or not probe['fixed_species']:
            raise ValueError('Pilot geometry probe is missing or from a different adapter.')
        (output/'gradient-probe.json').write_text(json.dumps(dict(probe,source_report=str(args.probe_report.resolve())),indent=2)+'\n')
    metadata.update(gradient_probe=probe,optimization_predictor='ALIGNN',evaluation_reference='CF score of exported geometry; never substituted for ALIGNN optimization loss.')
    # CF remains a frozen observer for the established export/evaluation controls.
    for model in original_cf_models:
        model.cuda()
    property_points=[]

    def hard_predict(out):
        gaps=[]
        eforms=[]
        with torch.no_grad():
            for start in range(0,len(context['ids']),4):
                stop=min(start+4,len(context['ids']))
                atoms=slice(context['slices'][start].start,context['slices'][stop-1].stop)
                g,e=predictor.predict(out['lattice_vectors'][start:stop],out['normed_batch_dir_coords'][atoms],
                    out['normalized_dist'][atoms],context['size'][start:stop],native_control=len(predictor.native_controls)<3)
                gaps.extend(g.tolist())
                eforms.extend(e.tolist())
        property_points.append(dict(gap=gaps,eform=eforms))
        return cf_hard_predict(out)

    context['hard_predict']=hard_predict
    protocol=dict(context['protocol'],a_dtype='float32',precision_control_error=None,
        gaussian_derivative='Not used: ALIGNN geometry ordinary autograd',adapter_control_error=max(max(c['max_errors']) for c in probe['native_controls']),
        adapter_control_calls=len(probe['native_controls']),optimization_predictor='ALIGNN',predictor_metadata=metadata)
    spec=importlib.util.spec_from_file_location('alignn_shared_adam',SMOACS/'adam_cycles/run.py')
    shared=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.run_cycles(context,hard_predict,protocol,steps=200,cycles=1)
    original=context['data']['fixed_atomic_distribution']
    if not torch.equal(context['targets'][3],original):
        raise ValueError('Immutable elemental tensor changed.')
    data=json.loads((output/'results.json').read_text())
    for row,atoms in zip(data['cycles'][0]['candidates'],context['slices']):
        if row['A']['atomic_numbers']!=(original[atoms].argmax(1)+1).tolist():
            raise ValueError('ALIGNN endpoint changed ordered species.')
    (output/'fixed-species-audit.json').write_text(json.dumps(dict(passed=True,denominator=len(context['ids'])))+'\n')
    (output/'alignn-adapter-audit.json').write_text(json.dumps(dict(passed=True,graph_builds=predictor.graph_builds,
        forward_calls=predictor.forward_calls,native_controls=predictor.native_controls,property_points=property_points,
        metadata=metadata),indent=2)+'\n')
    (output/'torch-memory.json').write_text(json.dumps(dict(max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved(),pid=os.getpid(),cohort=args.cohort))+'\n')
    gc.collect()


if __name__=='__main__':
    main()
