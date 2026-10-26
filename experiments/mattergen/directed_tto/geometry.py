"""RMS-scaled Cartesian offsets and orthonormal symmetric cell-strain variables."""
import numpy as np


def strain_basis():
    basis=np.zeros((6,3,3))
    for i in range(3):basis[i,i,i]=1
    for i,(a,b) in enumerate(((1,2),(0,2),(0,1)),3):basis[i,a,b]=basis[i,b,a]=1/np.sqrt(2)
    return basis


class Geometry:
    def __init__(self,numbers,cell,frac):
        self.numbers=np.array(numbers,dtype=int,copy=True);self.cell=np.array(cell,dtype=float,copy=True);self.frac=np.array(frac,dtype=float,copy=True)
        self.n=len(self.numbers);self.size=3*self.n+6;self.scale=float(np.linalg.det(self.cell)**(1/3));self.basis=strain_basis()
        if self.n<2 or self.frac.shape!=(self.n,3) or not np.isfinite(self.scale) or self.scale<=0:raise ValueError('Invalid reference pose.')

    def project(self,q):
        q=np.asarray(q,dtype=float).copy();atomic=q[:-6].reshape(self.n,3);atomic-=atomic.mean(0)
        return q

    def pose(self,q):
        q=np.asarray(q,dtype=float);strain=np.einsum('a,aij->ij',q[-6:]/self.scale,self.basis)
        cell=self.cell@(np.eye(3)+strain);cart=self.frac@cell+np.sqrt(self.n)*q[:-6].reshape(self.n,3)
        if np.linalg.det(cell)<=1e-6 or not np.isfinite(cart).all():raise ValueError('Invalid trial cell/positions.')
        return cell,cart@np.linalg.inv(cell)

    def torch_pose(self,q):
        import torch
        old=torch.as_tensor(self.cell,dtype=q.dtype,device=q.device);frac=torch.as_tensor(self.frac,dtype=q.dtype,device=q.device)
        basis=torch.as_tensor(self.basis,dtype=q.dtype,device=q.device)
        strain=torch.einsum('a,aij->ij',q[-6:]/self.scale,basis)
        cell=old@(torch.eye(3,dtype=q.dtype,device=q.device)+strain)
        cart=frac@cell+np.sqrt(self.n)*q[:-6].reshape(self.n,3)
        return cell,cart

    def energy_gradient(self,q,force,stress):
        cell,frac=self.pose(q);force=np.asarray(force);stress=np.asarray(stress)
        # ASE stress differentiates affine motion of current fractions. Offsets
        # here stay Cartesian, so subtract their affine contribution explicitly.
        cell_gradient=np.linalg.det(cell)*np.linalg.inv(cell).T@stress+(self.frac-frac).T@(-force)
        strain_gradient=np.einsum('ij,aij->a',cell_gradient,np.einsum('ij,ajk->aik',self.cell,self.basis))/self.scale
        return np.r_[-np.sqrt(self.n)*force.reshape(-1),strain_gradient]

    def feasible(self,q):
        from ase import Atoms
        from ase.neighborlist import neighbor_list
        q=np.asarray(q);strain=np.einsum('a,aij->ij',q[-6:]/self.scale,self.basis)
        if np.linalg.norm(strain,2)>.15 or np.linalg.norm(np.sqrt(self.n)*q[:-6].reshape(self.n,3),axis=1).max()>2.:return False
        try:cell,frac=self.pose(q)
        except ValueError:return False
        ratio=np.linalg.det(cell)/np.linalg.det(self.cell)
        return bool(.65<=ratio<=1.4 and not len(neighbor_list('d',Atoms(numbers=self.numbers,cell=cell,scaled_positions=frac,pbc=True),.7-1e-9)))
