"""Independent directional finite differences for the actual frozen predictors."""
import json
import torch
from gaussian import ExactGaussianAverage


def directional_probe(evaluate,parameter,direction):
    original=parameter.detach().clone()
    gradient=torch.autograd.grad(evaluate(),parameter)[0]
    analytic=float((gradient*direction).sum());rows=[]
    try:
        for step in (1e-5,1e-6,1e-7):
            with torch.no_grad():
                parameter.copy_(original+step*direction);plus=float(evaluate())
                parameter.copy_(original-step*direction);minus=float(evaluate())
            numeric=(plus-minus)/(2*step)
            rows.append(dict(step=step,analytic=analytic,numeric=numeric,
                absolute_error=abs(analytic-numeric),relative_error=abs(analytic-numeric)/max(1.,abs(analytic),abs(numeric))))
    finally:
        with torch.no_grad():parameter.copy_(original)
    return rows


def run(context,attention,legacy,output):
    forward=context['forward'];targets=context['targets'];base=context['base']
    generator=torch.Generator().manual_seed(42);rows=[]
    bounds=[(context['lower']/base,context['upper']/base),
            ((context['angle_lower']-60)/60,(context['angle_upper']-60)/60)]
    def evaluate():return forward(.01)[2].sum()
    for index,name in enumerate(('lengths','angles','coordinates','elements','oxidation')):
        parameter=targets[index];direction=torch.randn(parameter.shape,dtype=parameter.dtype,generator=generator)
        if index<2:
            lo,hi=bounds[index];direction*=((parameter>lo+1e-4)&(parameter<hi-1e-4))
        if index>=3:direction*=parameter>1e-4
        direction/=max(float(direction.abs().max()),1e-12)
        record=dict(variable=name)
        for label,operator in [('legacy',legacy),('corrected',ExactGaussianAverage)]:
            attention.AverageGaussiansBySoftmaxXYFunc_=operator
            record[label]=directional_probe(evaluate,parameter,direction)
        rows.append(record)
    attention.AverageGaussiansBySoftmaxXYFunc_=ExactGaussianAverage
    report=dict(role='original_four_template_directional_gradient_check',groups=rows,
                corrected_all_pass=all(r['corrected'][-1]['relative_error']<=1e-3 for r in rows),
                note='Central differences at fixed original inputs; no optimization or B relaxation.')
    (output/'gradient-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['corrected_all_pass']:raise ValueError('Full-model corrected directional gradient check failed.')
