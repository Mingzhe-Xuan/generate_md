"""Bounded independent or correlated structural moves and soft-mode directions."""
import numpy as np


def soft_directions(hessian,masses):
    masses=np.asarray(masses,dtype=float);inv=np.repeat(1/np.sqrt(masses),3)
    D=np.asarray(hessian)*inv[:,None]*inv[None,:];D=(D+D.T)/2;n=len(masses)
    Q=np.zeros((3*n,3))
    for axis in range(3):Q[axis::3,axis]=np.sqrt(masses)/np.sqrt(masses.sum())
    P=np.eye(3*n)-Q@Q.T;values,modes=np.linalg.eigh(P@D@P)
    selected=[i for i,v in enumerate(values) if abs(v)>1e-5 and np.linalg.norm(Q.T@modes[:,i])<.1][:6]
    if not selected:raise ValueError('No nontranslational curvature directions.')
    return values[selected],(modes[:,selected]*inv[:,None]).T.reshape(len(selected),n,3)


def mutate(atoms,reference,rng,amplitude,collective=False,strain=.0):
    trial=atoms.copy();trial.set_constraint(None)
    for _ in range(100):
        noise=rng.normal(size=(len(atoms),3))
        if collective:
            k=rng.integers(-1,2,size=3)
            if not np.any(k):k[0]=1
            phase=2*np.pi*(atoms.get_scaled_positions()@k)+rng.uniform(0,2*np.pi)
            noise=.8*np.sin(phase)[:,None]*rng.normal(size=3)[None,:]+.2*noise
        noise-=noise.mean(0);norm=np.sqrt(np.mean(np.sum(noise**2,axis=1)))
        if norm<1e-10:continue
        noise*=amplitude/norm
        if np.linalg.norm(noise,axis=1).max()>2*amplitude:continue
        eps=rng.normal(size=(3,3));eps=(eps+eps.T)/2;eps*=strain/max(np.linalg.norm(eps,2),1e-12)
        cell=atoms.cell.array@(np.eye(3)+eps);relative=np.linalg.solve(reference.cell.array,cell)
        singular=np.linalg.svd(relative,compute_uv=False);volume=np.linalg.det(relative)
        if min(singular)<.65 or max(singular)>1.4 or not .65<=volume<=1.4:continue
        trial.set_cell(cell,scale_atoms=True);trial.positions=atoms.get_scaled_positions()@cell+noise
        matrix=trial.get_all_distances(mic=True);np.fill_diagonal(matrix,np.inf)
        if not np.isfinite(trial.positions).all() or matrix.min()<.7:continue
        return trial
    raise ValueError('No collision-free bounded proposal in100 draws.')
