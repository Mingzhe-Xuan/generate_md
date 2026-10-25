"""Frozen original CUDA CF inference with explicit source and checkpoint checks."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import types


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Scorer:
    def __init__(self):
        import torch
        self.torch=torch;smoacs=Path(__file__).resolve().parents[2]/'smoacs';sys.path.insert(0,str(smoacs))
        from run_section21 import SOURCE_MEMBER,SOURCE_SHA256
        assets=Path('results/smoacs/assets-199').resolve();archive=Path('results/smoacs/downloads/reproducibility.tar.gz')
        expected=json.loads((smoacs/'assets.json').read_text())['reproducibility.tar.gz']['sha256']
        if sha(archive)!=expected:raise ValueError('Author archive changed.')
        checkpoints=json.loads((smoacs/'checkpoints.json').read_text())
        for name,item in checkpoints.items():
            if sha(assets/'models/crystalformer'/name)!=item['sha256']:raise ValueError('CF checkpoint changed.')
        with tarfile.open(archive) as handle:source=handle.extractfile(SOURCE_MEMBER).read()
        if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:raise ValueError('Frozen author source changed.')
        os.environ.update(SMOACS_ROOT=str(assets),MPLBACKEND='Agg')
        native=Path('third_party/SMOACS').resolve();sys.path[:0]=[str(native),str(native/'src')]
        self.author=types.ModuleType('basin_search_frozen_author');self.author.__file__=str(assets/SOURCE_MEMBER);sys.modules[self.author.__name__]=self.author
        exec(compile(source,self.author.__file__,'exec',dont_inherit=True),self.author.__dict__)
        if not torch.cuda.is_available():raise RuntimeError('CF must run on GPU.')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.models=self.author.Load_Pretrained_Crystalformers(device='cpu')
        for model in self.models:
            model.to('cuda').eval()
            for p in model.parameters():p.requires_grad_(False)
            if not all(p.is_cuda for p in model.parameters()):raise ValueError('CF CPU fallback.')
        self.metadata=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,source_sha256=SOURCE_SHA256,checkpoints=checkpoints,CF_calls=0)

    def predict(self,path):
        from pymatgen.core import Structure
        s=Structure.from_file(path);torch=self.torch;n=len(s)
        atomic=torch.nn.functional.one_hot(torch.tensor(s.atomic_numbers,device='cuda')-1,num_classes=98).float()
        cell=torch.tensor(s.lattice.matrix,dtype=torch.float32,device='cuda');coords=torch.tensor(s.frac_coords,dtype=torch.float32,device='cuda')@cell
        with torch.no_grad():gap,ef=self.author.Crystalformer_prediction(*self.models,atomic,coords,torch.zeros(n,dtype=torch.long,device='cuda'),cell[None],torch.tensor([n],device='cuda'))
        if not gap.is_cuda or not ef.is_cuda:raise ValueError('CF output not on CUDA.')
        values=float(gap.reshape(-1)[0]),float(ef.reshape(-1)[0])
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite CF score.')
        self.metadata['CF_calls']+=1
        return values
