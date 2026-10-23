"""Export final alternating iterates for the existing property/SUN evaluators."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import types


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    import os
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Submit batch processing through Slurm.')
    from pymatgen.core import Structure
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from run_section21 import SOURCE_MEMBER,SOURCE_SHA256
    complete=json.loads((args.run/'complete.json').read_text())
    if not complete['completed']:raise ValueError('Incomplete alternating run.')
    run=json.loads((args.run/'results.json').read_text());rows=run['candidates']
    if any(r['status']=='active' for r in rows):raise ValueError('Active candidates cannot be exported as final.')
    source=Path('third_party/SMOACS').resolve();sys.path[:0]=[str(source),str(source/'src')]
    os.environ['SMOACS_ROOT']=str(Path('results/smoacs/assets-199').resolve())
    with tarfile.open('results/smoacs/downloads/reproducibility.tar.gz') as tar:original=tar.extractfile(SOURCE_MEMBER).read()
    if hashlib.sha256(original).hexdigest()!=SOURCE_SHA256:raise ValueError('Author code changed.')
    author=types.ModuleType('alternating_export_author');author.__file__=str(source/SOURCE_MEMBER)
    sys.modules[author.__name__]=author;exec(compile(original,author.__file__,'exec',dont_inherit=True),author.__dict__)
    output=args.run/'postprocess';output.mkdir(exist_ok=False)
    (output/'inventory-249').mkdir();(output/'relaxation').mkdir()
    candidates=[];csv_rows=[];neutral_cache={}
    for row in rows:
        path=Path(row['a_source']);structure=Structure.from_file(path)
        counts=Counter(s.specie.symbol for s in structure);key=tuple(sorted(counts.items()))
        if key not in neutral_cache:
            neutral,_,_=author.elec_neutral_check_SUPER_COMMON(0,len(rows),[e for e,c in key],[[c] for e,c in key],True)
            neutral_cache[key]=bool(neutral)
        neutral=neutral_cache[key]
        separation=not any(n.nn_distance<.5 for neighbors in structure.get_all_neighbors(.5) for n in neighbors)
        valid=neutral and separation;identifier=row['candidate_id']
        candidate=dict(candidate_id=identifier,source=str(path.resolve()),sha256=digest(path),num_atoms=len(structure),
            formula=structure.composition.formula,elements=sorted(counts),original_gap=row['a_gap'],
            original_eform=row['a_eform'],original_valid=str(valid),alternating_status=row['status'],
            a_steps=row['a_steps'],initial_template_id=identifier)
        candidates.append(candidate)
        csv_rows.append({'original_fnames':identifier,'bandgap_onehot':row['a_gap'],'ef_onehot':row['a_eform'],
            'is_neutral_cmn':str(neutral),'minbond_less_than_0.5':str(separation),'valid_structure':str(valid)})
        folder=output/'relaxation'/Path(identifier).stem;folder.mkdir()
        record=dict(row['b_result'])
        if record['input_sha256']!=candidate['sha256']:raise ValueError('Final B input differs from final A.')
        if record['status']=='converged':
            shutil.copy2(record['relaxed_path'],folder/'relaxed.vasp')
            record['relaxed_path']=str((folder/'relaxed.vasp').resolve())
        (folder/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    csv_path=output/'result.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    inventory=dict(role='alternating_final_A_inputs',denominator=len(candidates),candidates=candidates,
        source_csv_sha256=digest(csv_path),alternating_protocol=run['protocol'],
        note='Candidate IDs pair initial templates; source geometry is the last A iterate, with B in relaxation/. '
             'B-converged final iterates are evaluated even if joint A/B convergence was not achieved; report status separately.')
    (output/'inventory-249/inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(exported=len(candidates),statuses=dict(Counter(r['status'] for r in rows)))),flush=True)


if __name__=='__main__':main()
