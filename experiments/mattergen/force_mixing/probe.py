"""Native MatterSim finite differences for coordinates and variable-cell pullback."""
import hashlib
import json
from pathlib import Path

import torch

from geometry import energy_surrogate,geometry


def code_hashes():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('geometry.py','protocol.py','provider.py','worker.py','probe.py')}


def run(context,client):
    if len(context['ids'])!=4:
        raise ValueError('Native gradient validation uses the original four-case pilot.')
    c={k:(v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k,v in context.items()}
    p=[v.detach().double().clone().requires_grad_(True) for v in context['targets']]
    generator=torch.Generator().manual_seed(8603)
    with torch.no_grad():
        p[0].add_(.01)
        p[1].add_(.0015)
        p[2].add_(torch.randn(p[2].shape,generator=generator,dtype=torch.float64)*.002)
        p[0].copy_(torch.clamp(p[0],min=c['lower']/c['base']+.005,max=c['upper']/c['base']-.005))
        p[1].copy_(torch.clamp(p[1],min=(c['angle_lower']-60)/60+.005,max=(c['angle_upper']-60)/60-.005))
    c['targets']=p
    cells,frac=geometry(c)
    numbers=context['data']['fixed_atomic_distribution'].argmax(1)+1

    def evaluate():
        h,s=geometry(c)
        return client.evaluate(c['ids'],h,s,numbers,c['slices'])

    rows=evaluate()
    energy=torch.tensor([rows[k]['energy_eV'] for k in c['ids']],dtype=torch.float64)
    force=torch.tensor([v for k in c['ids'] for v in rows[k]['forces_eV_A']],dtype=torch.float64)
    stress=torch.tensor([rows[k]['stress_eV_A3'] for k in c['ids']],dtype=torch.float64)
    gradients=torch.autograd.grad(energy_surrogate(cells,frac,c['batch'],energy,force,stress).sum(),p[:3])
    records=[]
    for block in range(3):
        direction=torch.randn(p[block].shape,generator=generator,dtype=torch.float64)
        direction/=direction.norm()
        before=p[block].detach().clone()
        predicted=float((gradients[block]*direction).sum())
        for eps in (.001,.003):
            values=[]
            with torch.no_grad():
                for sign in (1,-1):
                    p[block].copy_(before+sign*eps*direction)
                    result=evaluate()
                    values.append(sum(result[k]['energy_eV'] for k in c['ids']))
                p[block].copy_(before)
            finite=(values[0]-values[1])/(2*eps)
            tolerance=.05+.03*max(abs(finite),abs(predicted))
            records.append(dict(block=block,epsilon=eps,analytic=predicted,finite_difference=finite,
                absolute_error=abs(predicted-finite),tolerance=tolerance,passed=abs(predicted-finite)<=tolerance))
    report=dict(passed=all(r['passed'] for r in records),checks=records,code_sha256=code_hashes(),
        checkpoint_sha256=client.ready[0]['checkpoint_sha256'],dtype_geometry='float64',
        force_model='native MatterSim5M GPU float32',sign='F2=-grad total energy; ASE tensile-positive stress',
        note='Two central-difference scales; fixed predeclared abs0.05+rel0.03 tolerances account for native F32 energy noise.')
    (context['output']/'physics-gradient-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:
        raise ValueError('Native MatterSim force/stress pullback finite differences failed.')
    return report
