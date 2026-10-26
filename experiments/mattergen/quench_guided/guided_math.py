"""Trace-one guided covariance and antithetic post-quench gradient estimates."""
import numpy as np


def projected_normal_draws(project,size,pairs,rng):
    return np.stack([project(z) for z in rng.normal(size=(pairs,size))]),rng.normal(size=pairs)


def perturbations(guide,project,normal,scalar,sigma,alpha):
    guide=np.asarray(guide,dtype=float);normal=np.asarray(normal,dtype=float);scalar=np.asarray(scalar,dtype=float)
    size=len(guide);effective=size-3
    if effective<=0 or normal.shape!=(len(scalar),size) or not 0<=alpha<=1 or sigma<=0:raise ValueError('Invalid guided sampling dimensions/scale.')
    if not np.isfinite(np.r_[guide,normal.ravel(),scalar]).all():raise ValueError('Nonfinite guided sampling inputs.')
    guide=project(guide);norm=np.linalg.norm(guide)
    if norm<1e-10:alpha=1.;guide=np.zeros_like(guide)
    else:guide=guide/norm
    full=np.stack([project(z) for z in normal])
    noise=sigma*(np.sqrt(alpha/effective)*full+np.sqrt(1-alpha)*scalar[:,None]*guide[None,:])
    return noise,dict(alpha=alpha,effective_dimension=effective,guide=guide.tolist(),stationary_guide=bool(norm<1e-10))


def estimate(noise,plus,minus,sigma,project):
    noise=np.asarray(noise,dtype=float);plus=np.asarray(plus,dtype=float);minus=np.asarray(minus,dtype=float)
    if noise.ndim!=2 or noise.shape[0]!=len(plus) or plus.shape!=minus.shape or not len(plus) or sigma<=0:raise ValueError('Incomplete antithetic pairs.')
    if not np.isfinite(np.r_[noise.ravel(),plus,minus]).all() or np.any(plus<0) or np.any(plus>1) or np.any(minus<0) or np.any(minus>1):raise ValueError('Invalid bounded post-quench utility.')
    return project(noise.T@(plus-minus)/(2*len(plus)*sigma**2))


def unit_or_zero(vector):
    vector=np.asarray(vector,dtype=float);norm=np.linalg.norm(vector)
    if not np.isfinite(norm):raise ValueError('Nonfinite estimated direction.')
    return vector/norm if norm>=1e-10 else np.zeros_like(vector)
