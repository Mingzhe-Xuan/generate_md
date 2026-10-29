"""Pinned MP20 readout-label splits and property-blind hash selection."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from prepare_anchors import symmetric_cell
from csp_model import sha


POLICY=dict(seed=20261056,counts=dict(train=256,val=64,test=128),times=[.001,.05,.15,.30],
    properties=['gap_eV','eform_eV_atom'],label_columns=['band_gap','formation_energy_per_atom'],
    noise='Native marginal fractional Gaussian, remove only uniform translation, fixed clean cell/species.',
    label_exposure='Readout labels held out by author split; CSP/ALIGNN checkpoint structure exposure unknown.',
    eligibility='Ordered2..20atoms,Z<=94,finitecell/coordinates/labels; no property ranking or favorable replacement.')


def key(split,material_id):
    return hashlib.sha256(f"{POLICY['seed']}|{split}|{material_id}".encode()).hexdigest()


def ordered_rows(rows,split):
    return sorted(enumerate(rows),key=lambda p:key(split,p[1]['material_id']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    from pymatgen.core import Structure
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=False)
    dataset=Path('results/mattergen/csp-benchmarks-20261029')
    source=json.loads(Path('experiments/mattergen/csp_benchmarks/sources.json').read_text())
    selection=[];examined=[];split_metadata={};ids={}
    for split,count in POLICY['counts'].items():
        path=dataset/'mp20'/f'{split}.csv';raw=path.read_bytes()
        spec=next(a for a in source['assets'] if a['path']==f'mp20/{split}.csv')
        blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        if len(raw)!=spec['size'] or blob!=spec['checksum']:raise ValueError('Source CSV changed.')
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        ids[split]={r['material_id'] for r in rows}
        if len(ids[split])!=len(rows):raise ValueError('Duplicate material IDs.')
        ordered=ordered_rows(rows,split)
        chosen=0
        for row_index,row in ordered:
            reason=None
            try:
                structure=Structure.from_str(row['cif'],fmt='cif',primitive=False)
                numbers=np.asarray(structure.atomic_numbers)
                labels=np.array([float(row[column]) for column in POLICY['label_columns']])
                if not structure.is_ordered or not 2<=len(numbers)<=20 or numbers.max()>94:
                    reason='Outside fixed input domain.'
                elif not np.isfinite(labels).all() or not np.isfinite(structure.frac_coords).all() or structure.volume<=0:
                    reason='Nonfinite/invalid data.'
                examined.append(dict(split=split,id=row['material_id'],row_index=row_index,num_atoms=len(numbers),reason=reason))
                if reason:continue
                cell=symmetric_cell(structure.lattice.matrix)
                selection.append(dict(id=row['material_id'],split=split,row_index=row_index,numbers=numbers.tolist(),
                    cell=cell.tolist(),frac=np.remainder(structure.frac_coords,1).tolist(),labels=labels.tolist(),
                    cif_sha256=hashlib.sha256(row['cif'].encode()).hexdigest(),label_source=[row[k] for k in POLICY['label_columns']]))
                chosen+=1
                if chosen==count:break
            except Exception as error:
                examined.append(dict(split=split,id=row['material_id'],row_index=row_index,error=repr(error)))
        if chosen!=count:raise ValueError('Insufficient eligible structures: '+split)
        split_metadata[split]=dict(rows=len(rows),selected=chosen,csv_sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=blob)
    if any(ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Split ID overlap.')
    result=dict(policy=POLICY,cases=selection,examined=examined,splits=split_metadata,source=source,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(Path(__file__)))
    (root/'selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(split_metadata))


if __name__=='__main__':main()
