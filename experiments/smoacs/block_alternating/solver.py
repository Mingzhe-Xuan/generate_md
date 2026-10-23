"""Bounded minimization with explicit stationarity and geometry diagnostics."""
import numpy as np
from scipy.optimize import minimize


def projected_gradient(x, gradient, lower, upper):
    x,g,lo,hi=map(np.asarray,(x,gradient,lower,upper))
    return np.where(((x<=lo+1e-8)&(g>0))|((x>=hi-1e-8)&(g<0)),0.,g)


def solve(fun,x,lower,upper,maxiter=1200,maxfun=3000):
    x=np.clip(np.asarray(x,dtype=float),lower,upper)
    result=minimize(fun,x,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),
                    options=dict(maxiter=maxiter,maxfun=maxfun,maxls=40,gtol=1e-3,ftol=0.,maxcor=15))
    value,gradient=fun(result.x)
    residual=float(np.max(np.abs(projected_gradient(result.x,gradient,lower,upper))))
    return result.x,dict(converged=bool(np.isfinite(value) and residual<=1e-3),
        projected_gradient=residual,raw_gradient=float(np.max(np.abs(gradient))),loss=float(value),
        iterations=int(result.nit),evaluations=int(result.nfev),optimizer_message=str(result.message))


def gram_check(cell,reconstructed):
    cell,reconstructed=map(np.asarray,(cell,reconstructed))
    original=cell@cell.T;new=reconstructed@reconstructed.T
    difference=np.abs(new-original)
    return dict(passed=bool(np.allclose(new,original,atol=1e-5,rtol=2e-6)),
                absolute_error=float(difference.max()),
                relative_error=float(difference.max()/max(np.abs(original).max(),1e-12)))
