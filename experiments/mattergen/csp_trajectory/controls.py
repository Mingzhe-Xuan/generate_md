"""Two-scale energy/force checks at four prespecified last predictor inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'forward_inverse'))
from physics_client import Physics


def main(root):
    source=json.loads((root/'complete.json').read_text())
    physics=Physics(root/'force-controls');checks=[]
    try:
        for i in (0,7,15,23):
            row=next(r for r in source['states'] if r['trajectory']==i and r['stage']=='predictor' and r['step']==999)
            if row['status'] not in ('compared','stationary'):
                checks.append(dict(trajectory=i,status='not_evaluable',reasons=row['reasons']));continue
            path=next(p for p in source['paths'] if p['trajectory']==i)
            with np.load(root/path['path']) as d:
                numbers=d['numbers'];cell=d['cell'][-1];cart=d['frac'][-1]@cell
            direction=np.arange(cart.size,dtype=float).reshape(cart.shape)**2
            direction-=direction.mean(0);direction/=np.linalg.norm(direction)
            base=physics.request([dict(id=f'base:{i}',action='point',numbers=numbers.tolist(),cell=cell.tolist(),frac=(cart@np.linalg.inv(cell)).tolist())])[0]
            if base['status']!='completed':raise ValueError('Force control base failed.')
            expected=-float(np.sum(np.asarray(base['forces_eV_A'])*direction))
            for eps in (.002,.001):
                endpoints=physics.request([dict(id=f'{i}:{eps}:{sign}',action='point',numbers=numbers.tolist(),
                    cell=cell.tolist(),frac=((cart+sign*eps*direction)@np.linalg.inv(cell)).tolist()) for sign in (1,-1)])
                if any(p['status']!='completed' for p in endpoints):raise ValueError('Force FD endpoint failed.')
                fd=(endpoints[0]['energy_eV']-endpoints[1]['energy_eV'])/(2*eps)
                passed=abs(fd-expected)<=.005+.02*max(abs(fd),abs(expected))
                checks.append(dict(trajectory=i,epsilon=eps,FD=fd,expected=expected,passed=passed))
                if not passed:raise ValueError('Native force FD check failed: '+repr(checks[-1]))
        (root/'force-controls.json').write_text(json.dumps(dict(checks=checks,NFE=physics.NFE,requests=physics.requests,
            physics=physics.metadata,passed=all(c.get('passed',True) for c in checks)),indent=2)+'\n')
    finally:physics.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(args.run.resolve())
