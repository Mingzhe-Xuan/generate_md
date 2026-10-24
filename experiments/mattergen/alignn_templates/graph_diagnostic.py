"""Explain optimizer-versus-saved-pose scores without changing either inference path."""
import argparse
import json
import os
from pathlib import Path

import torch

from device_control import topology_signature
from model import Predictor,load_models


def load(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run model diagnostics through Slurm.')
    torch.set_num_threads(4)
    output=args.output.resolve()
    if not output.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Use current task results.')
    output.mkdir(parents=True,exist_ok=False)
    models,metadata=load_models(output)
    predictor=Predictor(models)
    from pymatgen.core import Structure
    records=[]
    for key in ('mattergen-0068.vasp','mattergen-0102.vasp','mattergen-0000.vasp'):
        found=[]
        for cohort in ('neutral','unmasked'):
            path=args.run/'alignn200'/cohort
            data=load(path/'results.json')
            ids=data['protocol']['ids']
            if key in ids:
                index=ids.index(key)
                found.append((path,data['cycles'][0]['candidates'][index],index))
        if len(found)!=1:
            raise ValueError('Diagnostic cases must exist exactly once.')
        path,row,index=found[0]
        a=row['A']
        adapter=load(path/'alignn-adapter-audit.json')
        frozen=next(r for r in load(path/'evaluation-cycle1/cross-prediction/predictions.json') if r['candidate_id']==key and r['stage']=='before')
        structure=Structure.from_file(a['source'])
        cells=torch.tensor(a['cell'],dtype=torch.float32)[None]
        frac=torch.tensor(a['frac'],dtype=torch.float32)
        numbers=torch.tensor(a['atomic_numbers'])
        onehot=torch.nn.functional.one_hot(numbers-1,98).float()
        sizes=torch.tensor([len(numbers)])
        raw_graphs=predictor.graphs(cells,frac,onehot,sizes)
        saved_graphs=predictor.graphs(torch.tensor(structure.lattice.matrix)[None],
            torch.tensor(structure.frac_coords),onehot,sizes)
        with torch.no_grad():
            raw=predictor.predict(cells,frac,onehot,sizes,graphs=raw_graphs,native_control=True)
            saved=predictor.predict(cells,frac,onehot,sizes,graphs=saved_graphs,native_control=True)
        changed=topology_signature(raw_graphs[0])!=topology_signature(saved_graphs[0])
        errors={}
        for i,prop in enumerate(('gap','eform')):
            optimizer=adapter['property_points'][1][prop][index]
            official=frozen['alignn_'+prop]
            first,second=float(raw[i][0]),float(saved[i][0])
            errors[prop]=dict(recorded_optimizer=optimizer,recorded_frozen=official,
                same_optimizer_graph_CPU=first,same_saved_graph_CPU=second,
                same_graph_optimizer_error=abs(first-optimizer),same_graph_frozen_error=abs(second-official),
                observed_cross_pipeline_error=abs(optimizer-official),
                graph_change_explanation_error=abs((first-second)-(optimizer-official)))
        passed=all(v['same_graph_optimizer_error']<=1e-3 and v['same_graph_frozen_error']<=1e-3 and
            v['graph_change_explanation_error']<=1e-3 for v in errors.values())
        records.append(dict(candidate_id=key,changed_topology=changed,
            optimizer_graph_edges=raw_graphs[0][0].num_edges(),saved_graph_edges=saved_graphs[0][0].num_edges(),
            errors=errors,same_graph_checks_passed=passed,
            interpretation='Different graphs explain the original failed cross-pipeline score check; no original tolerance or outcome is changed.'))
    result=dict(passed=all(r['same_graph_checks_passed'] for r in records),records=records,
        native_controls=predictor.native_controls,metadata=metadata,
        scope='CPU Slurm diagnostic only; identical tensors/models compared with optimizer versus saved-pose graphs. '
              'Original cross-pipeline differences remain recorded and are not relabeled as passing.')
    (output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise ValueError('Graph changes do not fully explain the score discrepancy.')


if __name__=='__main__':
    main()
