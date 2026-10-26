"""Original CF CUDA predictions and input gradients in the declared physical metric."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'basin_search'))
from scorer import Scorer


class GradientScorer(Scorer):
    def __init__(self):
        super().__init__();self.metadata['CF_backward_calls']=0

    def evaluate(self,geometry,q,backward=False):
        torch=self.torch
        with torch.enable_grad() if backward else torch.no_grad():
            variable=torch.tensor(np.asarray(q),dtype=torch.float32,device='cuda',requires_grad=backward)
            cell,cart=geometry.torch_pose(variable)
            atomic=torch.nn.functional.one_hot(torch.tensor(geometry.numbers,device='cuda')-1,num_classes=98).float()
            gap,ef=self.author.Crystalformer_prediction(*self.models,atomic,cart,torch.zeros(geometry.n,dtype=torch.long,device='cuda'),cell[None],torch.tensor([geometry.n],device='cuda'))
            loss=.5*(gap.reshape(-1)[0]-2.5)**2
            if not all(v.is_cuda and torch.isfinite(v).all() for v in (gap,ef,loss)):raise ValueError('Nonfinite/CPU CF result.')
            gradient=None
            if backward:
                g=torch.autograd.grad(loss,variable)[0]
                if not g.is_cuda or not torch.isfinite(g).all():raise ValueError('Invalid CF input gradient.')
                gradient=g.detach().cpu().double().numpy();self.metadata['CF_backward_calls']+=1
        self.metadata['CF_calls']+=1
        return dict(gap=float(gap.detach().reshape(-1)[0]),eform=float(ef.detach().reshape(-1)[0]),loss=float(loss.detach()),gradient=gradient)
