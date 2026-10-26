"""Original CF CUDA predictions and input gradients in the declared physical metric."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import importlib.util
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'basin_search'))
from scorer import Scorer


class GradientScorer(Scorer):
    def __init__(self):
        super().__init__();self.metadata['CF_backward_calls']=0
        path=Path(__file__).resolve().parents[2]/'smoacs/block_alternating/gaussian.py'
        spec=importlib.util.spec_from_file_location('directed_exact_gaussian',path)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        self.gaussian=adapter.ExactGaussianAverage;self.attention={}
        for model in self.models:
            for layer in model.modules():
                name=layer.__class__.__module__
                if name.endswith('indexed_lattice_multi_head_attention'):
                    module=sys.modules[name];self.attention[name]=(module,module.AverageGaussiansBySoftmaxXYFunc_)
        if not self.attention:raise RuntimeError('No actual Torch CF attention module found for input-gradient adapter.')
        self.metadata['input_gradient_adapter']=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),attention_modules=list(self.attention))

    @contextmanager
    def operators(self,corrected):
        for module,legacy in self.attention.values():module.AverageGaussiansBySoftmaxXYFunc_=self.gaussian if corrected else legacy
        try:yield
        finally:
            for module,legacy in self.attention.values():module.AverageGaussiansBySoftmaxXYFunc_=legacy

    def predict(self,path):
        with self.operators(False):return super().predict(path)

    def evaluate(self,geometry,q,backward=False):
        torch=self.torch
        with self.operators(backward), torch.enable_grad() if backward else torch.no_grad():
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
                if self.gaussian.calls<=0:raise RuntimeError('Gaussian input-gradient adapter was not executed.')
                self.metadata['input_gradient_adapter']['actual_calls']=self.gaussian.calls
        self.metadata['CF_calls']+=1
        return dict(gap=float(gap.detach().reshape(-1)[0]),eform=float(ef.detach().reshape(-1)[0]),loss=float(loss.detach()),gradient=gradient)
