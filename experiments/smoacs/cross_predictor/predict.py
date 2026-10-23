"""Frozen ALIGNN inference on immutable Crystalformer-generated structures."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from metrics import classify,index_rows

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'mp_gappbe_alignnn':'8e9309fcb435a88481ccbc945d16cb1343f9ac24d8198607f753ce59b687242e',
          'mp_e_form_alignnn':'848fb8f5844445ab2c751a7a7790ad69d7d3c6c385eb645758c48049e2bd8e71'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--predictor',type=Path,required=True)
    parser.add_argument('--relaxation',type=Path,required=True)
    parser.add_argument('--author-csv',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Submit inference through Slurm.')
    import torch
    import numpy as np
    from pymatgen.core import Structure
    from jarvis.core.atoms import Atoms
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback.')
    source=Path('third_party/SMOACS').resolve()
    sys.path[:0]=[str(source),str(ROOT)]
    from src.models import alignn_loader
    from third_party.alignn_minimal.models.alignn import ALIGNN
    from third_party.alignn_minimal.graphs import Graph
    from alignn_checkpoint import TrustedCheckpointTorch
    inventory=json.loads(args.inventory.read_text())
    candidates=inventory['candidates'];ids=[c['candidate_id'] for c in candidates]
    if inventory['denominator']!=len(ids):raise ValueError('Inventory denominator mismatch.')
    with (args.predictor/'predictions.csv').open(encoding='utf8') as stream:
        cf=index_rows(list(csv.DictReader(stream)),ids)
    control=json.loads((args.predictor/'summary.json').read_text())
    if control['control_passed'] is not True or control['candidate_count']!=len(ids):
        raise ValueError('Original Crystalformer control did not pass.')
    if digest(args.author_csv)!=inventory['source_csv_sha256']:raise ValueError('Original CSV changed.')
    with args.author_csv.open(encoding='utf8') as stream:
        originals=index_rows([dict(r,candidate_id=r['original_fnames']) for r in csv.DictReader(stream)],ids)
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output.resolve();weights=args.weights.resolve()
    digests=[];configs={}
    for name,expected in EXPECTED.items():
        path=weights/(name+'.zip')
        if digest(path)!=expected:raise ValueError('Unverified ALIGNN ZIP: '+name)
        with zipfile.ZipFile(path) as archive:
            config_name=[n for n in archive.namelist() if 'config.json' in n][-1]
            configs[name]=json.loads(archive.read(config_name))
            for member in archive.namelist():
                if ('checkpoint_' in member and 'pt' in member) or 'best_model.pt' in member:
                    digests.append(hashlib.sha256(archive.read(member)).hexdigest())
    options=dict(neighbor_strategy='k-nearest',cutoff=8.,max_neighbors=12,atom_features='cgcnn',
                 compute_line_graph=True,use_canonize=True,cutoff_extra=3.)
    for config in configs.values():
        for key in ('neighbor_strategy','cutoff','max_neighbors','atom_features'):
            if key in config and config[key]!=options[key]:
                raise ValueError('Checkpoint graph configuration differs: '+key)
    tmp=output/'loader-tmp';tmp.mkdir()
    cwd=Path.cwd();old=alignn_loader.torch
    alignn_loader.torch=TrustedCheckpointTorch(torch,digests)
    try:
        os.chdir(tmp)
        models=alignn_loader.Load_Pretrained_ALIGNN('bg_and_eform',device='cuda',weights_dir=str(weights))
    finally:
        os.chdir(cwd);alignn_loader.torch=old
    natives=[]
    for model in models:
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        if not all(p.is_cuda for p in model.parameters()):raise RuntimeError('ALIGNN must be on CUDA.')
        native=ALIGNN(model.config).to('cuda').eval()
        native.load_state_dict(model.state_dict(),strict=True)
        for p in native.parameters():p.requires_grad_(False)
        natives.append(native)
    metadata=dict(job_id=os.environ['SLURM_JOB_ID'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        submodule_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        packages={name:version(name) for name in ('torch','dgl','numpy','pymatgen','jarvis-tools')},
        device=torch.cuda.get_device_name(),checkpoint_zip_sha256=EXPECTED,checkpoint_payload_sha256=digests,
        checkpoint_graph_configurations={k:{f:v.get(f) for f in options if f in v} for k,v in configs.items()},
        graph_options=options,inventory_sha256=digest(args.inventory),crystalformer_csv_sha256=digest(args.predictor/'predictions.csv'),
        author_csv_sha256=digest(args.author_csv),crystalformer_control=control,controls=[],denominator=len(ids),
        interpretation='Different frozen architectures trained on MEGNet; agreement is not ground-truth/DFT accuracy. No model optimization.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]
    for candidate in candidates:
        identifier=candidate['candidate_id'];prediction=cf[identifier]
        if digest(candidate['source'])!=candidate['sha256']:raise ValueError('Original structure changed.')
        for prop,key in (('gap','original_gap'),('eform','original_eform')):
            if abs(float(prediction[prop+'_before'])-candidate[key])>control['control_tolerance']:
                raise ValueError('Crystalformer control differs from inventory.')
        neutral=originals[identifier]['is_neutral_cmn']
        if neutral not in ('True','False'):raise ValueError('Unknown composition neutrality.')
        for stage in ('before','after'):
            row=dict(candidate_id=identifier,stage=stage,status='pending',input_sha256=candidate['sha256'],
                crystalformer_gap=None,crystalformer_eform=None,alignn_gap=None,alignn_eform=None,
                crystalformer_flags=classify(None,None,False),alignn_flags=classify(None,None,False))
            rows.append(row)
            path=Path(candidate['source'])
            if stage=='after':
                folder=args.relaxation/Path(identifier).stem
                record=json.loads((folder/'result.json').read_text())
                if record['input_sha256']!=candidate['sha256']:raise ValueError('Relaxation input mismatch.')
                if record['status']!='converged':
                    row.update(status='not_eligible_relaxation',relaxation_status=record['status'])
                    continue
                path=folder/'relaxed.vasp'
                if digest(path)!=record['relaxed_sha256']:raise ValueError('Relaxed structure changed.')
                if prediction['relaxation_status']!='converged':raise ValueError('Crystalformer relaxed prediction mismatch.')
            row['geometry_sha256']=digest(path)
            row['crystalformer_gap']=float(prediction['gap_'+stage])
            row['crystalformer_eform']=float(prediction['eform_'+stage])
            structure=Structure.from_file(path)
            valid=neutral=='True' and not any(n.nn_distance<.5 for neighbors in structure.get_all_neighbors(.5) for n in neighbors)
            row['structure_valid']=valid
            row['crystalformer_flags']=classify(row['crystalformer_gap'],row['crystalformer_eform'],valid)
            try:
                atoms=Atoms(lattice_mat=structure.lattice.matrix,coords=structure.frac_coords,
                            elements=[s.specie.symbol for s in structure],cartesian=False)
                g,lg=Graph.atom_dgl_multigraph(atoms,**options)
                if g.num_nodes()!=len(structure):raise ValueError('Graph dropped atoms.')
                g,lg=g.to('cuda'),lg.to('cuda')
                features=g.ndata['atom_features'];lengths=torch.norm(g.edata['r'],dim=1);angles=lg.edata['h']
                if not all(v.is_cuda and torch.isfinite(v).all() for v in (features,lengths,angles)):
                    raise ValueError('Invalid graph features.')
                with torch.inference_mode():
                    raw=[model((g,lg),features,lengths,angles) for model in models]
                if not all(v.is_cuda and torch.isfinite(v).all() for v in raw):raise ValueError('Invalid ALIGNN output.')
            except Exception as error:
                row.update(status='prediction_failed',error=repr(error))
                print(json.dumps(dict(candidate_id=identifier,stage=stage,error=repr(error))),flush=True)
                continue
            if len(metadata['controls'])<2:
                lattice=torch.tensor(structure.lattice.matrix,dtype=features.dtype,device='cuda')[None]
                with torch.inference_mode():native_outputs=[m((g.local_var(),lg.local_var(),lattice)) for m in natives]
                errors=[float(torch.max(torch.abs(a-b))) for a,b in zip(raw,native_outputs)]
                if not all(np.isfinite(e) and e<=1e-5 for e in errors):
                    raise ValueError('Native ALIGNN and explicit-feature implementation disagree: '+repr(errors))
                metadata['controls'].append(dict(candidate_id=identifier,stage=stage,max_errors=errors))
                (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            gap_raw=float(raw[0].reshape(-1)[0]);energy=float(raw[1].reshape(-1)[0])
            row.update(status='completed',alignn_raw_gap=gap_raw,alignn_gap=max(0.,gap_raw),alignn_eform=energy,
                       alignn_flags=classify(max(0.,gap_raw),energy,valid),graph_nodes=g.num_nodes(),graph_edges=g.num_edges())
            (output/'predictions.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(candidate_id=identifier,stage=stage,alignn_gap=row['alignn_gap'],alignn_eform=energy)),flush=True)
    if len(metadata['controls'])<2:raise ValueError('Insufficient successful native implementation controls.')
    (output/'predictions.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('CROSS_PREDICTOR_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
