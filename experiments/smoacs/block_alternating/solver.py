"""Bounded minimization with explicit stationarity and geometry diagnostics."""
import numpy as np
from scipy.optimize import linprog,minimize


def projected_gradient(x, gradient, lower, upper):
    x,g,lo,hi=map(np.asarray,(x,gradient,lower,upper))
    return np.where(((x<=lo+1e-8)&(g>0))|((x>=hi-1e-8)&(g<0)),0.,g)


def hinge_subgradient(x,gap,gap_gradient,energy_gradient,lower,upper):
    if abs(abs(gap-2.5)-.04)>1e-7:raise ValueError('Not at the band-gap hinge.')
    x,d,e,lo,hi=map(np.asarray,(x,gap_gradient,energy_gradient,lower,upper))
    use_upper=x>lo+1e-8;use_lower=x<hi-1e-8
    matrix=np.concatenate([np.column_stack((d[use_upper],-np.ones(use_upper.sum()))),
                           np.column_stack((-d[use_lower],-np.ones(use_lower.sum())))])
    rhs=np.concatenate([-e[use_upper],e[use_lower]])
    interval=(-1.,0.) if gap<2.5 else (0.,1.)
    result=linprog([0.,1.],A_ub=matrix,b_ub=rhs,bounds=[interval,(0.,None)],method='highs')
    if not result.success:raise ValueError('Hinge stationarity LP failed: '+result.message)
    coefficient=float(result.x[0])
    return e+coefficient*d,coefficient


def solve(fun,x,lower,upper,maxiter=1200,maxfun=3000):
    x=np.clip(np.asarray(x,dtype=float),lower,upper)
    calls=0;iterations=0;messages=[];best=None
    class Budget(Exception):pass
    def evaluate(point):
        nonlocal calls,best
        if calls>=maxfun:raise Budget()
        value,gradient=fun(point);calls+=1
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise ValueError('Nonfinite objective/gradient.')
        residual=float(np.max(np.abs(projected_gradient(point,gradient,lower,upper))))
        if best is None or value<best[0]-1e-13 or (abs(value-best[0])<=1e-13 and residual<best[3]):
            best=(value,np.array(point,copy=True),np.array(gradient,copy=True),residual)
        return value,gradient
    try:
        evaluate(x)
        for attempt in range(16):
            if best[3]<=1e-3 or iterations>=maxiter or calls>=maxfun:break
            result=minimize(evaluate,x,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),
                options=dict(maxiter=maxiter-iterations,maxfun=maxfun-calls,maxls=40,gtol=1e-3,ftol=0.,maxcor=15))
            iterations+=int(result.nit);messages.append(str(result.message))
            evaluate(result.x)
            value,x,gradient,residual=best
            if residual<=1e-3 or iterations>=maxiter:break
            direction=-projected_gradient(x,gradient,lower,upper)/max(residual,1e-12)
            improved=False
            for step in (.01,.003,.001,.0003,.0001,1e-5,1e-6,1e-7,1e-8):
                trial=np.clip(x+step*direction,lower,upper)
                trial_value,_=evaluate(trial)
                improved=improved or trial_value<value-1e-12
            if not improved:
                messages.append('No measured projected-gradient descent in fallback grid.');break
            x=best[1].copy();iterations+=1
            messages.append('Restart after measured descent; optimizer stop was not stationarity.')
        else:messages.append('Restart safety limit reached, not convergence.')
    except Budget:messages.append('Shared evaluation budget reached, not convergence.')
    value,x,gradient,residual=best
    return x,dict(converged=bool(np.isfinite(value) and residual<=1e-3),
        projected_gradient=residual,raw_gradient=float(np.max(np.abs(gradient))),loss=float(value),
        iterations=iterations,evaluations=calls,optimizer_message=messages)


def gram_check(cell,reconstructed):
    cell,reconstructed=map(np.asarray,(cell,reconstructed))
    original=cell@cell.T;new=reconstructed@reconstructed.T
    difference=np.abs(new-original)
    return dict(passed=bool(np.allclose(new,original,atol=1e-5,rtol=2e-6)),
                absolute_error=float(difference.max()),
                relative_error=float(difference.max()/max(np.abs(original).max(),1e-12)))
