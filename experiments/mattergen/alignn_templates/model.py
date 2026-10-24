"""Pinned ALIGNN predictors with freshly rebuilt graphs and geometry autograd."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import torch

from geometry import edge_features,recover_images

SMOACS=Path(__file__).resolve().parents[2]/'smoacs'
OPTIONS=dict(neighbor_strategy='k-nearest',cutoff=8.,max_neighbors=12,atom_features='cgcnn',
    compute_line_graph=True,use_canonize=True,cutoff_extra=3.)
EXPECTED={'mp_gappbe_alignnn':'8e9309fcb435a88481ccbc945d16cb1343f9ac24d8198607f753ce59b687242e',
    'mp_e_form_alignnn':'848fb8f5844445ab2c751a7a7790ad69d7d3c6c385eb645758c48049e2bd8e71'}


def load_models(output):
    sys.path[:0]=[str(Path('third_party/SMOACS').resolve()),str(SMOACS)]
    from src.models import alignn_loader
    from alignn_checkpoint import TrustedCheckpointTorch
    weights=Path('results/smoacs/alignn-weights').resolve()
    digests=[]
    for name,expected in EXPECTED.items():
        path=weights/(name+'.zip')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Unverified ALIGNN checkpoint ZIP.')
        with zipfile.ZipFile(path) as archive:
            config=json.loads(archive.read([n for n in archive.namelist() if 'config.json' in n][-1]))
            for key in ('neighbor_strategy','cutoff','max_neighbors','atom_features'):
                if key in config and config[key]!=OPTIONS[key]:
                    raise ValueError('Graph options differ from checkpoint.')
            for name in archive.namelist():
                if ('checkpoint_' in name and 'pt' in name) or 'best_model.pt' in name:
                    digests.append(hashlib.sha256(archive.read(name)).hexdigest())
    tmp=output/'loader-tmp'
    tmp.mkdir()
    cwd=Path.cwd()
    old=alignn_loader.torch
    try:
        alignn_loader.torch=TrustedCheckpointTorch(torch,digests)
        os.chdir(tmp)
        models=alignn_loader.Load_Pretrained_ALIGNN('bg_and_eform',device='cpu',weights_dir=str(weights))
    finally:
        os.chdir(cwd)
        alignn_loader.torch=old
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return models,dict(checkpoint_zip_sha256=EXPECTED,checkpoint_payload_sha256=digests,graph_options=OPTIONS)


class Predictor:
    def __init__(self,models):
        self.models=models
        self.graph_builds=0
        self.forward_calls=0
        self.native_controls=[]
        self.frozen_graphs=None

    def graphs(self,cells,frac,onehot,sizes):
        from jarvis.core.atoms import Atoms
        from pymatgen.core import Element
        from third_party.alignn_minimal.graphs import Graph
        graphs=[]
        offset=0
        for cell,count in zip(cells,sizes.tolist()):
            coordinates=frac[offset:offset+count]
            elements=[Element.from_Z(z).symbol for z in (onehot[offset:offset+count].argmax(1)+1).tolist()]
            atoms=Atoms(lattice_mat=cell.detach().cpu().numpy(),coords=coordinates.detach().cpu().numpy(),elements=elements,cartesian=False)
            g,lg=Graph.atom_dgl_multigraph(atoms,**OPTIONS)
            if g.num_nodes()!=count:
                raise ValueError('Graph changed the ordered atoms.')
            # Upstream reverse edges reuse the forward image metadata; reconstruct signs from r.
            images=recover_images(g.edata['r'].double(),cell.detach().cpu().double(),
                coordinates.detach().cpu().double(),*g.edges(order='eid'))
            graphs.append((g,lg,images))
            self.graph_builds+=1
            offset+=count
        return graphs

    def predict(self,cells,frac,onehot,sizes,graphs=None,native_control=False):
        import dgl
        if graphs is None:
            graphs=self.frozen_graphs
        graphs=self.graphs(cells,frac,onehot,sizes) if graphs is None else graphs
        device,dtype=cells.device,cells.dtype
        features=[]
        lengths=[]
        cosines=[]
        gs=[]
        lgs=[]
        offset=0
        for cell,count,(g,lg,images) in zip(cells,sizes.tolist(),graphs):
            g,lg=g.to(device),lg.to(device)
            r,h=edge_features(cell,frac[offset:offset+count],*g.edges(order='eid'),images.to(device=device,dtype=dtype),*lg.edges(order='eid'))
            features.append(g.ndata['atom_features'].to(dtype))
            lengths.append(r)
            cosines.append(h)
            gs.append(g)
            lgs.append(lg)
            offset+=count
        g,lg=dgl.batch(gs),dgl.batch(lgs)
        x,r,h=[torch.cat(v) for v in (features,lengths,cosines)]
        values=[model((g,lg),x,r,h).reshape(-1) for model in self.models]
        if not all(torch.isfinite(v).all() for v in values):
            raise ValueError('Nonfinite ALIGNN prediction.')
        self.forward_calls+=1
        if native_control:
            from third_party.alignn_minimal.models.alignn import ALIGNN
            errors=[]
            with torch.no_grad():
                for model,value in zip(self.models,values):
                    native=ALIGNN(model.config).to(device=device,dtype=dtype).eval()
                    native.load_state_dict(model.state_dict(),strict=True)
                    ng,nlg=g.local_var(),lg.local_var()
                    ng.ndata['atom_features']=ng.ndata['atom_features'].to(dtype)
                    ng.edata['r']=ng.edata['r'].to(dtype)
                    nlg.edata['h']=nlg.edata['h'].to(dtype)
                    reference=native((ng,nlg,cells))
                    errors.append(float((reference.reshape(-1)-value).abs().max()))
            if max(errors)>1e-5:
                raise ValueError('Native ALIGNN predictor equivalence failed: '+str(errors))
            self.native_controls.append(dict(max_errors=errors,crystals=len(cells),device=str(device)))
        return values[0].clamp_min(0),values[1]

    def forward(self,*,optimization_targets,fixed_inputs,scalers,bandgap_model,e_form_model,
                abc_range,angle_range,**unused):
        from src.utils.coord import compute_lattice_vectors
        lengths,angles,coords,elements,_=optimization_targets
        onehot=fixed_inputs['fixed_atomic_distribution']
        if not torch.equal(elements,onehot):
            raise ValueError('ALIGNN optimization changed immutable elements.')
        if bandgap_model is not self.models[0] or e_form_model is not self.models[1]:
            raise ValueError('Optimization model handles do not match the ALIGNN adapter.')
        abc=torch.clamp(scalers['abc_scaler'].rescale(lengths),min=abc_range[0],max=abc_range[1])
        angle=torch.clamp(scalers['angle_scaler'].rescale(angles),min=angle_range[0],max=angle_range[1])
        cells=compute_lattice_vectors(abc,angle)
        frac=torch.remainder(coords,1.)
        sizes=fixed_inputs['size']
        valid=torch.isfinite(cells).all(dim=(1,2)) & (torch.linalg.det(cells).abs()>1e-6)
        if bool(valid.all()):
            gap,ef=self.predict(cells,frac,onehot,sizes)
        else:
            if torch.is_grad_enabled():
                raise ValueError('Invalid geometry before an active ALIGNN gradient update.')
            # Return invalid values only for the rejected proposal; the shared driver rolls it back.
            gap=torch.full((len(sizes),),float('nan'),dtype=cells.dtype,device=cells.device)
            ef=gap.clone()
            atom_mask=torch.repeat_interleave(valid,sizes)
            if valid.any():
                g,e=self.predict(cells[valid],frac[atom_mask],onehot[atom_mask],sizes[valid])
                gap[valid],ef[valid]=g,e
        pos=torch.bmm(frac[:,None,:],cells.repeat_interleave(sizes,dim=0)).squeeze(1)
        return gap,ef,dict(lattice_vectors=cells,normed_batch_dir_coords=frac,normalized_dist=onehot,pos=pos,
            bandgap_pred=gap,ef_pred=ef,batch_abc=abc,batch_angle=angle)
