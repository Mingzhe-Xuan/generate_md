"""Exact input derivatives for the CPU Gaussian positional encoding."""
import math
import torch


class ExactGaussianAverage:
    @staticmethod
    def apply(m,K,x,y,dim,W=None,width_scale=1.,basis_func='',keepdim=False):
        if basis_func not in ('',None,'cgcnn') or dim not in (-2,2):
            raise ValueError('Only the audited Gaussian encoding configuration is supported.')
        positive=x>0
        distance=torch.where(positive,torch.sqrt(torch.where(positive,x,torch.ones_like(x))),torch.zeros_like(x))
        centers=torch.arange(1,K+1,dtype=x.dtype,device=x.device)[None,None,None]*(m/K)
        basis=torch.exp(-((centers-distance)/(math.sqrt(2)*width_scale*m/K))**2)
        weights=torch.softmax(x*y,dim=dim)
        result=weights.transpose(-2,-1)@basis
        if W is not None:result=result@W.transpose(-2,-1)
        return result if keepdim else result.squeeze(2)
