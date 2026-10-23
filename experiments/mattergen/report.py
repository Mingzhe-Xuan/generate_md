"""Verify MatterGen end-to-end provenance and report unchanged task metrics."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

from fetch import digest


def audit(root,selection,expected_count=128):
    root=Path(root);selection=Path(selection)
    path=Path(__file__).resolve().parents[1]/'smoacs/full_comparison/report.py'
    spec=importlib.util.spec_from_file_location('mattergen_task_statistics',path)
    stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
    inventory=json.loads((root/'inventory-249/inventory.json').read_text());candidates=inventory['candidates'];ids=[c['candidate_id'] for c in candidates]
    if inventory['denominator']!=expected_count or len(set(ids))!=expected_count or len(ids)!=expected_count:raise ValueError('Generated denominator differs.')
    generation=json.loads((root/'generated/complete.json').read_text());gpu=json.loads((root/'generated/gpu-audit.json').read_text())
    if generation['count']!=expected_count or not generation['completed'] or generation['sampling_steps']!=1000:raise ValueError('Generation incomplete.')
    if not gpu['all_cuda'] or not gpu['finite_positions_and_cells'] or gpu['score_calls']<1000:raise ValueError('Generation GPU audit failed.')
    sources=json.loads(Path(__file__).with_name('sources.json').read_text())
    if generation['source']!=sources or generation['conditioning']!={'dft_band_gap':2.5} or generation['guidance']!=2.:raise ValueError('Generation protocol differs.')
    if inventory['generation_manifest_sha256']!=digest(root/'generated/complete.json'):raise ValueError('Generation provenance changed.')
    metadata=json.loads((root/'cross-prediction/metadata.json').read_text())
    if metadata['denominator']!=expected_count or len(metadata['controls'])<2 or not metadata['crystalformer_control']['control_passed']:
        raise ValueError('Frozen predictor controls incomplete.')
    rows=json.loads((root/'cross-prediction/predictions.json').read_text());before=[r for r in rows if r['stage']=='before'];after=[r for r in rows if r['stage']=='after']
    stats.indexed(before,ids);stats.indexed(after,ids);relaxation=[]
    for c in candidates:
        if digest(c['source'])!=c['sha256']:raise ValueError('Generated source changed.')
        row=next(r for r in after if r['candidate_id']==c['candidate_id'])
        record=json.loads((root/'relaxation'/Path(c['candidate_id']).stem/'result.json').read_text());relaxation.append(record)
        if record['input_sha256']!=c['sha256'] or row['input_sha256']!=c['sha256']:raise ValueError('B input mismatch.')
        if record['status']=='converged':
            if digest(record['relaxed_path'])!=record['relaxed_sha256'] or row.get('geometry_sha256')!=record['relaxed_sha256']:raise ValueError('Evaluated B changed.')
    sun=json.loads((root/'SUN/candidates.json').read_text());stats.indexed(sun,ids)
    if not any(r['status']=='converged' for r in relaxation):raise ValueError('No force-converged B endpoints; end-to-end evaluation not established.')
    if not any(r['stable'] is not None for r in sun):raise ValueError('All stability values unknown; evaluation incomplete.')
    if not any(r['status']=='completed' for r in after):raise ValueError('No complete cross-predictor B evaluation.')
    result=dict(verified_end_to_end=True,denominator=expected_count,generation=generation,gpu_audit=gpu,
        before_B=stats.property_stats(before,ids),after_B=stats.property_stats(after,ids),SUN=stats.sun_stats(sun,after,ids),
        relaxation_status_counts=dict(Counter(r['status'] for r in relaxation)),
        reference_selection=str(selection),reference_inventory_sha256=digest(selection/'mattersim-inventory.json'),
        reference_summary=json.loads((root/'SUN/summary.json').read_text()),
        prediction_sha256=digest(root/'cross-prediction/predictions.json'),sun_sha256=digest(root/'SUN/candidates.json'),
        note='Official pretrained de novo gap-conditioned MatterGen; Ef is evaluated, not jointly conditioned. '
             'Same frozen CF/ALIGNN and candidate B/SUN definitions; all generated samples remain in the denominator. '
             'Unknown reference coverage is explicit. No DFT or original-paper training reproduction claim.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--expected-count',type=int,default=128);args=p.parse_args();result=audit(args.run,args.selection,args.expected_count)
    (args.run/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# MatterGen: current property and SUN task','', '| Metric | Before B | After B |','|---|---:|---:|']
    def cell(value):return f"{value['pass_count']}/{value['denominator']} ({100*value['confirmed_rate']:.2f}%; unknown {value['unknown_count']})"
    for metric,value in result['before_B'].items():lines.append('| '+metric+' | '+cell(value)+' | '+cell(result['after_B'][metric])+' |')
    lines+=['','| SUN metric | Passes |','|---|---:|']
    for metric,value in result['SUN'].items():lines.append('| '+metric+' | '+cell(value)+' |')
    lines+=['',result['note'],''];(args.run/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(verified=True,denominator=result['denominator'],relaxation=result['relaxation_status_counts'],SUN=result['SUN'])),flush=True)


if __name__=='__main__':main()
