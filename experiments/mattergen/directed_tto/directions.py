"""True-gradient directions, matched controls and finite-HVP unrolled reverse pass."""
import numpy as np


def normalize(v):
    length=np.linalg.norm(v)
    if not np.isfinite(length) or length<1e-10:raise ValueError('Stationary or nonfinite gradient.')
    return np.asarray(v)/length


def hessian(gradient,q,epsilon=.002):
    q=np.asarray(q);offsets=np.eye(len(q))*epsilon
    values=gradient(np.vstack([q+offsets,q-offsets]))
    H=((values[:len(q)]-values[len(q):])/(2*epsilon)).T
    asymmetry=float(np.linalg.norm(H-H.T)/max(np.linalg.norm(H),1e-12))
    return (H+H.T)/2,asymmetry


def soft_descent(g,H,project):
    values,vectors=np.linalg.eigh(H);ridge=max(.1,.01*np.max(np.abs(values)))
    result=project(-(vectors@((vectors.T@g)/np.maximum(values,ridge))))
    if np.dot(g,result)>=0:raise ValueError('Soft preconditioner lost CF descent.')
    return normalize(result),dict(ridge=ridge,eigenvalues=values.tolist())


def matched_random(direction,rng,project):
    out=project(rng.normal(size=len(direction)))
    for block in (slice(None,-6),slice(-6,None)):
        target=np.linalg.norm(direction[block]);length=np.linalg.norm(out[block])
        out[block]*=target/max(length,1e-30)
    return out


def forward(q,gradient,eta,steps,project):
    states=[np.asarray(q).copy()]
    for _ in range(steps):states.append(states[-1]-eta*project(gradient(states[-1][None])[0]))
    return states


def unrolled(q,gradient,property_gradient,eta,steps,project,epsilon=.002):
    states=forward(q,gradient,eta,steps,project);v=project(property_gradient(states[-1]))
    for state in states[:-1][::-1]:
        norm=np.linalg.norm(v)
        if norm<1e-14:break
        offset=epsilon*v/norm;plus,minus=gradient(np.array([state+offset,state-offset]))
        v=project(v-eta*project((plus-minus)/(2*epsilon)*norm))
    return v,states
