"""Hold endpoint geometry fixed and vary ALIGNN connectivity only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from csp_model import sha
from property_model import Properties


def execute(source,root):
    if root.exists():raise ValueError('Preserve previous graph diagnostic.')
    root.mkdir(parents=True)
    complete=json.loads((source/'complete.json').read_text());model=Properties(root)
    started=time.perf_counter()
    report=dict(source_complete_sha256=sha(source/'complete.json'),source_plan_sha256=sha(source/'plan.json'),
        script_sha256=sha(Path(__file__)),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        model=model.metadata,replay_abs_tolerance=1e-8,geometry_changes=False,relaxations=0,rows=[],coverage=[])
    for case in complete['cases']:
        report['coverage'].append(dict(id=case['id'],status=case['status']))
        if case['status']!='completed':continue
        numbers=np.array(case['numbers']);cell=np.array(case['cell']);references={}
        for tolerance,reference in case['references'].items():
            cart=np.array(reference['physics']['frac'])@cell
            value,graph=model.evaluate(numbers,cell,cart)
            if not np.allclose(value['values'],reference['property']['values'],rtol=0,atol=1e-8):
                raise ValueError('Reference property replay changed.')
            references[tolerance]=(value,graph)
        for pair in case['pairs']:
            if pair['status']!='completed':continue
            value,graph=references[str(pair['tolerance_eV_A'])]
            for sign,index in [('plus',0),('minus',1)]:
                cart=np.array(pair['physics'][index]['frac'])@cell
                fresh,_=model.evaluate(numbers,cell,cart)
                if not np.allclose(fresh['values'],pair['relaxed_'+sign]['values'],rtol=0,atol=1e-8):
                    raise ValueError('Endpoint property replay changed.')
                fixed,_=model.evaluate(numbers,cell,cart,graphs=graph)
                total=fresh['values']-value['values']
                geometry=fixed['values']-value['values']
                topology=fresh['values']-fixed['values']
                if not np.allclose(total,geometry+topology,rtol=0,atol=1e-12):raise ValueError('Response decomposition failed.')
                report['rows'].append(dict(id=case['id'],direction=pair['direction'],amplitude_RMS_A=pair['amplitude_RMS_A'],
                    tolerance_eV_A=pair['tolerance_eV_A'],sign=sign,
                    endpoint_RMS_to_reference_A=pair['endpoint_RMS_to_reference_A'][index],
                    fresh_values=fresh['values'].tolist(),fixed_values=fixed['values'].tolist(),reference_values=value['values'].tolist(),
                    total_change=total.tolist(),fixed_graph_geometry_change=geometry.tolist(),connectivity_change=topology.tolist(),
                    fresh_graph_hash=fresh['graph_hash'],reference_graph_hash=value['graph_hash']))
        print('GRAPH_PARENT',case['id'],flush=True)
    expected=2*sum(p['status']=='completed' for c in complete['cases'] for p in c['pairs'])
    if len(report['rows'])!=expected:raise ValueError('Endpoint coverage incomplete.')
    report.update(execution_complete=True,seconds=time.perf_counter()-started,expected_endpoints=expected)
    (root/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    execute(args.source,args.run)


if __name__=='__main__':main()
