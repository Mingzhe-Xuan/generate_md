"""Variable-cohort export for the unchanged frozen CF/ALIGNN evaluation pipeline."""
import csv
from pathlib import Path
import shutil
import time
from probe import load,save,sha


def export_evaluation(root,method,selected,plan,scorer):
    parents=plan['parent_ids']
    if len(selected)!=len(parents) or len(set(parents))!=len(parents):raise ValueError('Export cohort coverage changed.')
    destination=root/f'evaluation-{method}';manifest=destination/'inventory-249/inventory.json'
    if not destination.resolve().is_relative_to(root.resolve()):raise ValueError('Invalid export destination.')
    signatures=[dict(parent_id=p,point_id=q['id'],input_sha256=q['B']['input_sha256'],pose_sha256=q['B']['relaxed_sha256']) for p,q in zip(parents,selected)]
    if manifest.exists():
        existing=load(manifest)
        if existing.get('selection_signatures')!=signatures or existing['source_csv_sha256']!=sha(destination/'result.csv'):raise ValueError('Stale completed evaluation export.')
        for parent,q in zip(parents,selected):
            if sha(q['pose'])!=q['B']['relaxed_sha256'] or sha(destination/'relaxation'/Path(parent).stem/'relaxed.vasp')!=q['B']['relaxed_sha256']:raise ValueError('Completed export pose changed.')
        return
    if destination.exists():destination.rename(destination.with_name(destination.name+f'-incomplete-{time.time_ns()}'))
    destination.mkdir();(destination/'inventory-249').mkdir();items=[];rows=[]
    for case,q in zip(plan['cases'],selected):
        parent=case['parent_id'];source=Path(q['input']);B=dict(q['B'])
        if sha(source)!=B['input_sha256'] or sha(q['pose'])!=B['relaxed_sha256']:raise ValueError('Selected input/terminal SHA changed.')
        gap,ef=scorer.predict(source);folder=destination/'relaxation'/Path(parent).stem;folder.mkdir(parents=True)
        shutil.copy2(q['pose'],folder/'relaxed.vasp');B.update(candidate_id=parent,relaxed_path=str((folder/'relaxed.vasp').resolve()));save(folder/'result.json',B)
        items.append(dict(case['inventory'],candidate_id=parent,source=str(source),sha256=B['input_sha256'],original_gap=gap,original_eform=ef,original_valid=str(case['neutral']),search_point_id=q['id']))
        rows.append(dict(original_fnames=parent,bandgap_onehot=gap,ef_onehot=ef,is_neutral_cmn=str(case['neutral']),**{'minbond_less_than_0.5':'True'},valid_structure=str(case['neutral'])))
    with (destination/'result.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(manifest,dict(role='Independent heldout-parent gradient validation',denominator=len(parents),candidates=items,
        selection_signatures=signatures,source_csv_sha256=sha(destination/'result.csv')))
