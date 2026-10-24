"""Native-forward and fixed-connectivity finite-difference checks before optimization."""
import hashlib
import json
from pathlib import Path

import torch


def source_hashes():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('geometry.py','model.py','probe.py')}


def run(context,predictor):
    from gpu import move_context
    if len(context['ids'])!=4:
        raise ValueError('Run the finite-difference probe on the original four-parent pilot.')
    original=[p.detach().clone() for p in context['targets']]
    move_context(context,'cpu',torch.float64)
    with torch.no_grad():
        _,_,_,out=context['forward'](.01)
        predictor.predict(out['lattice_vectors'],out['normed_batch_dir_coords'],out['normalized_dist'],context['size'],native_control=True)
        context['targets'][0].add_(.0017)
        context['targets'][1].add_(.0013)
        for target,low,high in ((context['targets'][0],context['lower']/context['base'],context['upper']/context['base']),
                                (context['targets'][1],(context['angle_lower']-60)/60,(context['angle_upper']-60)/60)):
            target.copy_(torch.clamp(target,min=low+.001,max=high-.001))
        context['targets'][2].add_(.0013)
        _,_,_,out=context['forward'](.01)
        predictor.predict(out['lattice_vectors'],out['normed_batch_dir_coords'],out['normalized_dist'],context['size'],native_control=True)
        predictor.frozen_graphs=predictor.graphs(out['lattice_vectors'],out['normed_batch_dir_coords'],out['normalized_dist'],context['size'])
    records=[]
    generator=torch.Generator().manual_seed(1407)
    for block in range(3):
        parameter=context['targets'][block]
        direction=torch.randn(parameter.shape,generator=generator,dtype=torch.float64)
        direction/=direction.norm()
        saved=parameter.detach().clone()
        for prop,name in ((0,'gap'),(1,'eform')):
            result=context['forward'](.01)[prop].sum()
            gradient=torch.autograd.grad(result,parameter)[0]
            autograd=float((gradient*direction).sum())
            eps=1e-5
            values=[]
            with torch.no_grad():
                for sign in (1,-1):
                    parameter.copy_(saved+sign*eps*direction)
                    values.append(float(context['forward'](.01)[prop].sum()))
                parameter.copy_(saved)
            finite=(values[0]-values[1])/(2*eps)
            error=abs(finite-autograd)
            tolerance=1e-5+1e-3*max(abs(finite),abs(autograd))
            records.append(dict(block=block,property=name,autograd=autograd,finite_difference=finite,
                absolute_error=error,tolerance=tolerance,passed=error<=tolerance))
    predictor.frozen_graphs=None
    move_context(context,'cpu',torch.float32)
    with torch.no_grad():
        for parameter,value in zip(context['targets'],original):
            parameter.copy_(value)
    report=dict(passed=all(r['passed'] for r in records),finite_differences=records,native_controls=predictor.native_controls,
        adapter_source_sha256=source_hashes(),dtype='float64',topology='Fixed during finite differences only; rebuilt every optimization forward.',
        fixed_species=True)
    (context['output']/'gradient-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise ValueError('ALIGNN geometry finite differences failed; do not start the experiment.')
    return report
