"""Export every generated crystal to the existing immutable property/SUN interface."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import types

from fetch import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm for batch export and predictor inference.')
    import numpy as np
    import torch
    from pymatgen.core import Structure
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for property inference.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(__file__).resolve().parents[1]/'smoacs';sys.path.insert(0,str(root))
    from run_section21 import SOURCE_MEMBER,SOURCE_SHA256
    archive=Path('results/smoacs/downloads/reproducibility.tar.gz');assets=Path('results/smoacs/assets-199').resolve()
    if digest(archive)!=json.loads((root/'assets.json').read_text())['reproducibility.tar.gz']['sha256']:raise ValueError('Author archive changed.')
    for name,record in json.loads((root/'checkpoints.json').read_text()).items():
        if digest(assets/'models/crystalformer'/name)!=record['sha256']:raise ValueError('Frozen Crystalformer checkpoint changed.')
    with tarfile.open(archive) as handle:original=handle.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA256:raise ValueError('Author source changed.')
    source=Path('third_party/SMOACS').resolve();sys.path[:0]=[str(source),str(source/'src')]
    os.environ.update(SMOACS_ROOT=str(assets),SMOACS_DEVICE='cpu',MPLBACKEND='Agg')
    author=types.ModuleType('mattergen_frozen_task_predictors');author.__file__=str(assets/SOURCE_MEMBER);sys.modules[author.__name__]=author
    exec(compile(original,author.__file__,'exec',dont_inherit=True),author.__dict__)
    models=author.Load_Pretrained_Crystalformers(device='cpu')
    data=json.loads((args.run/'generated/structures.json').read_text());completion=json.loads((args.run/'generated/complete.json').read_text())
    if not completion['completed'] or len(data)!=completion['count']:raise ValueError('Incomplete generated cohort.')
    folder=args.run/'structures';folder.mkdir();structures=[];paths=[]
    for i,value in enumerate(data):
        structure=Structure.from_dict(value)
        if not np.isfinite(structure.lattice.matrix).all() or structure.volume<=1e-6 or not np.isfinite(structure.frac_coords).all():
            raise ValueError('Invalid generated geometry at index '+str(i)+'; cohort retained, evaluation blocked.')
        if not structure.is_ordered or not all(1<=z<=98 for z in structure.atomic_numbers):raise ValueError('Unsupported generated species.')
        path=folder/f'mattergen-{i:04d}.vasp';structure.to(filename=str(path),fmt='poscar')
        paths.append(path);structures.append(Structure.from_file(path))
    def predict(structure,device):
        count=len(structure);cell=torch.tensor(structure.lattice.matrix,dtype=torch.float32,device=device)
        onehot=torch.nn.functional.one_hot(torch.tensor(structure.atomic_numbers,device=device)-1,98).float()
        xyz=torch.tensor(structure.frac_coords,dtype=torch.float32,device=device)@cell
        with torch.no_grad():gap,ef=author.Crystalformer_prediction(*models,onehot,xyz,
            torch.zeros(count,dtype=torch.long,device=device),cell[None],torch.tensor([count],device=device))
        if not all(torch.isfinite(t).all() for t in (gap,ef)):raise ValueError('Nonfinite frozen property prediction.')
        return float(gap.reshape(-1)[0]),float(ef.reshape(-1)[0])
    cpu=[predict(s,'cpu') for s in structures[:2]]
    for model in models:
        model.to('cuda').eval()
        if not all(parameter.is_cuda for parameter in model.parameters()):raise RuntimeError('Predictor CPU fallback.')
    gpu=[predict(s,'cuda') for s in structures];control=max(abs(a-b) for row,reference in zip(gpu,cpu) for a,b in zip(row,reference))
    if control>1e-3:raise ValueError('Frozen CPU/GPU export control failed.')
    candidates=[];csv_rows=[];neutral_cache={}
    for structure,path,(gap,ef) in zip(structures,paths,gpu):
        counts=Counter(s.specie.symbol for s in structure);key=tuple(sorted(counts.items()))
        if key not in neutral_cache:
            neutral_cache[key]=bool(author.elec_neutral_check_SUPER_COMMON(0,len(data),[e for e,c in key],[[c] for e,c in key],True)[0])
        neutral=neutral_cache[key];separated=not any(n.nn_distance<.5 for group in structure.get_all_neighbors(.5) for n in group)
        candidates.append(dict(candidate_id=path.name,source=str(path.resolve()),sha256=digest(path),num_atoms=len(structure),
            formula=structure.composition.formula,elements=sorted(counts),original_gap=gap,original_eform=ef,original_valid=str(neutral and separated)))
        csv_rows.append({'original_fnames':path.name,'bandgap_onehot':gap,'ef_onehot':ef,'is_neutral_cmn':str(neutral),
            'minbond_less_than_0.5':str(separated),'valid_structure':str(neutral and separated)})
    csv_path=args.run/'result.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    inventory=dict(role='MatterGen_dft_band_gap_2.5_de_novo',denominator=len(data),candidates=candidates,
        source_csv_sha256=digest(csv_path),generation_manifest_sha256=digest(args.run/'generated/complete.json'),
        cpu_gpu_control_error=control,author_source_sha256=SOURCE_SHA256)
    (args.run/'inventory-249').mkdir();(args.run/'inventory-249/inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(exported=len(data),cpu_gpu_control_error=control)),flush=True)


if __name__=='__main__':main()
