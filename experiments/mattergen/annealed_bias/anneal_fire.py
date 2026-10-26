"""FIRE in optimization coordinates with bounded, explicitly rejected updates."""
import numpy as np
from anneal_policy import FIRE


def minimize(q0,objective,project,feasible,settings=None):
    cfg=dict(FIRE,**(settings or {}))
    q=project(np.asarray(q0,dtype=float).copy())
    if not feasible(q):
        raise ValueError('FIRE initial point is infeasible.')
    velocity=np.zeros_like(q);dt=cfg['dt'];alpha=cfg['alpha'];positive=0;started=False
    records=[];rejected=0;accepted=0
    status='step_limit'
    for iteration in range(cfg['steps']+1):
        current=objective(q)
        gradient=np.asarray(current['gradient']);force=-gradient
        if gradient.shape!=q.shape or not np.isfinite(gradient).all() or not np.isfinite(current['energy']):
            raise ValueError('Invalid FIRE objective/gradient.')
        maximum=float(np.abs(gradient).max())
        records.append(dict(iteration=iteration,energy=float(current['energy']),gradient_inf=maximum,dt=float(dt),alpha=float(alpha)))
        if maximum<=cfg['fmax']:
            status='converged';break
        if iteration==cfg['steps']:
            break
        if started:
            power=float(velocity@force)
            if power>0:
                velocity=(1-alpha)*velocity+alpha*force*np.linalg.norm(velocity)/max(np.linalg.norm(force),1e-30)
                if positive>cfg['Nmin']:
                    dt=min(dt*cfg['finc'],cfg['dtmax']);alpha*=cfg['fa']
                positive+=1
            else:
                velocity[:]=0.;alpha=cfg['alpha'];dt*=cfg['fdec'];positive=0
        else:
            started=True
        if dt<cfg['minimum_dt']:
            status='stalled_dt';break
        velocity+=dt*force
        step=dt*velocity
        norm=np.linalg.norm(step)
        if norm>cfg['maxstep']:
            step*=cfg['maxstep']/norm
        candidate=None
        for backtrack in range(cfg['backtracks']+1):
            trial=project(q+step*(.5**backtrack))
            if np.linalg.norm(trial-q)<=1e-14:
                continue
            if feasible(trial):
                candidate=trial
                if backtrack:velocity*=.5**backtrack
                break
            rejected+=1
        if candidate is None:
            status='stalled_boundary';break
        q=candidate;accepted+=1
    return dict(q=q,status=status,converged=status=='converged',updates=accepted,rejected_trials=rejected,trace=records,
        final_energy=float(records[-1]['energy']),final_gradient_inf=records[-1]['gradient_inf'])
