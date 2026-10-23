"""Export cycle1/3 immutable A/B inputs for the established evaluators."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm for batch exports.')
    if not json.loads((args.run/'complete.json').read_text())['completed']:raise ValueError('Incomplete run.')
    data=json.loads((args.run/'results.json').read_text());ids=data['protocol']['ids'];scope=[]
    for cycle in (1,3):
        rows=next(r['candidates'] for r in data['cycles'] if r['cycle']==cycle)
        if len(rows)!=4 or {r['candidate_id'] for r in rows}!=set(ids):raise ValueError('Cycle cohort mismatch.')
        root=args.run/('evaluation-cycle'+str(cycle));(root/'inventory-249').mkdir(parents=True,exist_ok=False)
        (root/'relaxation').mkdir();candidates=[];csv_rows=[]
        for row in rows:
            a=row['A'];identifier=row['candidate_id']
            if digest(a['source'])!=a['sha256']:raise ValueError('A endpoint changed.')
            candidate=dict(candidate_id=identifier,source=a['source'],sha256=a['sha256'],num_atoms=a['num_atoms'],
                formula=a['formula'],elements=a['elements'],original_gap=a['gap'],original_eform=a['eform'],
                original_valid=str(a['valid']),cycle=cycle,cycle_status=row['status'],a_updates=row['a_updates'])
            candidates.append(candidate)
            csv_rows.append({'original_fnames':identifier,'bandgap_onehot':a['gap'],'ef_onehot':a['eform'],
                'is_neutral_cmn':str(a['neutral']),'minbond_less_than_0.5':str(a['separated']),'valid_structure':str(a['valid'])})
            record=dict(row.get('b_result',dict(candidate_id=identifier,input_sha256=a['sha256'],status='not_eligible')))
            if record['input_sha256']!=a['sha256']:raise ValueError('B is not paired to this cycle A.')
            if row['status']!='completed' and record['status']=='converged':raise ValueError('Invalid completed B status.')
            destination=root/'relaxation'/Path(identifier).stem;destination.mkdir()
            if record['status'] in ('converged','unconverged'):
                if digest(record['relaxed_path'])!=record['relaxed_sha256']:raise ValueError('B endpoint changed.')
                shutil.copy2(record['relaxed_path'],destination/'relaxed.vasp')
                record['relaxed_path']=str((destination/'relaxed.vasp').resolve())
            if record['status']=='converged':scope.append(dict(candidate_id='cycle'+str(cycle)+'_'+identifier,elements=a['elements']))
            (destination/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        csv_path=root/'result.csv'
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
        inventory=dict(role='Adam200_AB_cycle_endpoint',cycle=cycle,denominator=4,candidates=candidates,
            source_csv_sha256=digest(csv_path),protocol=data['protocol'],
            note='Inactive later-cycle A inputs are provenance placeholders only; B is explicitly not eligible and never counted as a later success.')
        (root/'inventory-249/inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (args.run/'reference-scope.json').write_text(json.dumps(dict(denominator=len(scope),candidates=scope))+'\n')


if __name__=='__main__':main()
