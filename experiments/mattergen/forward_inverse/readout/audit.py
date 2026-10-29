"""Independent author-data, feature, fitted-head and held-out metric audit."""
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
from csp_model import sha


def same(actual,expected,atol=1e-10,rtol=1e-10):
    if not np.allclose(actual,expected,atol=atol,rtol=rtol):raise ValueError('Independent numerical reconstruction failed.')


def clipped(value):
    value=np.array(value,copy=True);value[...,0]=np.maximum(value[...,0],0);return value


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-30))


def code_hash(commit,name,digest):
    raw=subprocess.check_output(['git','show',commit+':experiments/mattergen/forward_inverse/readout/'+name])
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source blob mismatch: '+name)


def execute(root):
    from pymatgen.core import Structure
    selection=json.loads((root/'selection.json').read_text());feature_plan=json.loads((root/'feature-plan.json').read_text())
    complete=json.loads((root/'feature-complete.json').read_text());report=json.loads((root/'evaluation/report.json').read_text())
    output=root/'evaluation/independent-audit.json'
    if output.exists():raise ValueError('Preserve earlier audit.')
    if not complete['execution_complete'] or complete['parents']!=448:raise ValueError('Feature coverage incomplete.')
    if sha(root/'selection.json')!=feature_plan['selection_sha256'] or sha(root/'features.npz')!=complete['features_sha256']:
        raise ValueError('Feature/selection hash mismatch.')
    for key,name in [('heads_sha256','heads.npz'),('predictions_sha256','frozen-predictions.npz')]:
        if sha(root/'evaluation'/name)!=report[key]:raise ValueError('Frozen model/prediction hash mismatch.')
    code_hash(selection['commit'],'prepare.py',selection['script_sha256'])
    code_hash(feature_plan['commit'],'extract.py',feature_plan['script_sha256'])
    code_hash(feature_plan['commit'],'observer.py',feature_plan['observer_sha256'])
    code_hash(report['source_commit'],'evaluate.py',report['script_sha256'])
    diagnostic=Path('results/mattergen/forward-inverse-readout-feature-diagnostic-20261030')
    if sha(diagnostic/'complete.json')!=feature_plan['observer_validation']['diagnostic_complete_sha256']:
        raise ValueError('Observer diagnostic provenance changed.')
    original_diagnostic=json.loads((diagnostic/'complete.json').read_text())
    if sha(diagnostic/'vectors.npz')!=original_diagnostic['vectors_sha256']:raise ValueError('Observer diagnostic vectors changed.')
    expected_ids=[];author_labels=[];dataset=Path('results/mattergen/csp-benchmarks-20261029/mp20')
    for split,count in selection['policy']['counts'].items():
        raw=(dataset/(split+'.csv')).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=selection['splits'][split]['csv_sha256']:raise ValueError('CSV source mismatch.')
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        ordered=sorted(enumerate(rows),key=lambda pair:hashlib.sha256(f"20261056|{split}|{pair[1]['material_id']}".encode()).hexdigest())
        eligible=[]
        for index,row in ordered:
            try:
                structure=Structure.from_str(row['cif'],fmt='cif',primitive=False)
                z=np.array(structure.atomic_numbers);y=np.array([float(row[k]) for k in selection['policy']['label_columns']])
                valid=structure.is_ordered and 2<=len(z)<=20 and z.max()<=94 and np.isfinite(y).all() and np.isfinite(structure.frac_coords).all() and structure.volume>0
                if not valid:continue
            except Exception:continue
            eligible.append((index,row,structure,y))
            if len(eligible)==count:break
        chosen=[c for c in selection['cases'] if c['split']==split]
        if len(eligible)!=len(chosen):raise ValueError('Author selection coverage mismatch.')
        for (index,row,structure,y),case in zip(eligible,chosen):
            if index!=case['row_index'] or row['material_id']!=case['id'] or hashlib.sha256(row['cif'].encode()).hexdigest()!=case['cif_sha256']:
                raise ValueError('Author row/CIF selection changed.')
            same(case['labels'],y,atol=0,rtol=0);same(case['numbers'],structure.atomic_numbers,atol=0,rtol=0)
            same(case['frac'],np.remainder(structure.frac_coords,1),atol=1e-12,rtol=0)
            cell=np.array(case['cell']);same(cell@cell.T,structure.lattice.matrix@structure.lattice.matrix.T)
            expected_ids.append(case['id']);author_labels.append(y)
    if len(set(expected_ids))!=448:raise ValueError('Cross-split material overlap.')
    with np.load(root/'features.npz') as z:
        arrays={key:z[key] for key in z.files}
    if arrays['ids'].tolist()!=expected_ids:raise ValueError('Feature row alignment changed.')
    same(arrays['labels'],author_labels,atol=0,rtol=0)
    max_feature_error=0.;score_failures=0;states=0
    for i,(case,manifest) in enumerate(zip(selection['cases'],complete['parent_vectors'])):
        path=root/manifest['vectors']
        if case['id']!=manifest['id'] or sha(path)!=manifest['sha256']:raise ValueError('Parent vector hash/ID mismatch.')
        with np.load(path) as z:
            numbers=z['numbers'];cell=z['cell'];frac=z['frac']
            same(numbers,case['numbers'],atol=0,rtol=0);same(cell,case['cell'],atol=0,rtol=0);same(frac,case['frac'],atol=0,rtol=0)
            b=np.r_[np.bincount(numbers,minlength=99)[1:99]/len(numbers),np.log(len(numbers)),np.log(np.linalg.det(cell)/len(numbers))]
            same(z['composition_cell'],b);same(arrays['baseline'][i],np.repeat(b[None],4,axis=0))
            rng=np.random.default_rng(manifest['noise_seed'])
            for j in range(4):
                noise=rng.normal(size=frac.shape);noise-=noise.mean(0);noise*=z['std'][j]
                same(noise,z['fractional_noise'][j],atol=1e-14,rtol=0);same((frac+noise)@cell,z['noisy_cart'][j])
                for domain in ('clean','noisy'):
                    primary=z[domain+'_features'][j,100:];reference=z[domain+'_wrapper_embeddings'][j]
                    error=relative(primary,reference);max_feature_error=max(max_feature_error,error)
                    if error>1e-5:raise ValueError('Independent feature observer mismatch.')
                    metadata=manifest['controls'][j][domain]
                    same(error,metadata['feature_relative_error'],atol=1e-12,rtol=0)
                    if not metadata['same_graph'] or not metadata['feature_QC_passed']:raise ValueError('Feature graph QC failed.')
                    score_error=relative(z[domain+'_scores'][j],z[domain+'_repeat_scores'][j])
                    same(score_error,metadata['score_relative_error'],atol=1e-10,rtol=1e-7)
                    if (score_error<=1e-5)!=metadata['original_score_relative_criterion_passed']:raise ValueError('Score diagnostic relabeled.')
                    score_failures+=score_error>1e-5;states+=1
            for key in ('clean_features','noisy_features','clean_alignn','noisy_alignn'):
                same(z[key],arrays[key][i],atol=0,rtol=0)
    masks={name:arrays['splits']==name for name in ('train','val','test')}
    y=np.array(author_labels);train_y=np.repeat(y[masks['train'],None,:],4,axis=1).reshape(-1,2)
    val_y=np.repeat(y[masks['val'],None,:],4,axis=1).reshape(-1,2)
    with np.load(root/'evaluation/heads.npz') as z:heads={k:z[k] for k in z.files}
    with np.load(root/'evaluation/frozen-predictions.npz') as z:predictions={k:z[k] for k in z.files}
    if predictions['ids'].tolist()!=arrays['ids'][masks['test']].tolist():raise ValueError('Test prediction alignment changed.')
    permutation=np.random.default_rng(20261057).permutation(256);same(permutation,heads['permutation'],atol=0,rtol=0)
    shuffled_y=np.repeat(y[masks['train']][permutation,None,:],4,axis=1).reshape(-1,2)
    stationarity=[]
    for name,training,evaluation,clean_evaluation,target in [
        ('composition_cell','baseline','baseline','baseline',train_y),('clean_head','clean_features','noisy_features','clean_features',train_y),
        ('noise_head','noisy_features','noisy_features','clean_features',train_y),('shuffled_head','noisy_features','noisy_features','clean_features',shuffled_y)]:
        tx=arrays[training][masks['train']].reshape(-1,arrays[training].shape[-1])
        vx=arrays[evaluation][masks['val']].reshape(-1,arrays[evaluation].shape[-1])
        mean=tx.mean(0);scale=tx.std(0);scale[scale<=1e-8]=1.;ym=target.mean(0)
        same(mean,heads[name+'_mean']);same(scale,heads[name+'_scale']);same(ym,heads[name+'_ymean'])
        sx=(tx-mean)/scale;sv=(vx-mean)/scale;u,s,vt=np.linalg.svd(sx,full_matrices=False)
        coefficients=[];errors=[]
        for alpha in heads[name+'_alphas']:
            coef=vt.T@(s[:,None]*(u.T@(target-ym))/(s[:,None]**2+alpha))
            coefficients.append(coef);errors.append(np.mean((clipped(sv@coef+ym)-val_y)**2,axis=0))
        same(errors,heads[name+'_validation_MSE'],atol=1e-8,rtol=1e-9)
        index=np.argmin(errors,axis=0);selected=heads[name+'_alphas'][index]
        same(selected,heads[name+'_selected_alphas'],atol=0,rtol=0)
        coefficient=np.column_stack([coefficients[index[j]][:,j] for j in range(2)])
        same(coefficient,heads[name+'_coefficient'],atol=1e-8,rtol=1e-7)
        residual=sx.T@(sx@coefficient-(target-ym))+coefficient*selected
        denominator=np.linalg.norm(sx.T@(target-ym))+np.linalg.norm(coefficient*selected)
        stationarity.append(float(np.linalg.norm(residual)/max(denominator,1e-30)))
        if stationarity[-1]>1e-8:raise ValueError('Ridge normal-equation stationarity failed.')
        for domain,key in [('noisy',evaluation),('clean',clean_evaluation)]:
            pred=clipped(((arrays[key][masks['test']]-mean)/scale)@coefficient+ym)
            same(pred,predictions[name+'_'+domain],atol=1e-8,rtol=1e-8)
    for domain in ('clean','noisy'):same(arrays[domain+'_alignn'][masks['test']],predictions['ALIGNN_'+domain],atol=0,rtol=0)
    test_y=y[masks['test']]
    for name,result in report['results'].items():
        for domain,metrics in result.items():
            delta=predictions[name+'_'+domain]-test_y[:,None,:]
            same(metrics['MAE'],np.abs(delta).mean(axis=(0,1)));same(metrics['RMSE'],np.sqrt((delta**2).mean(axis=(0,1))))
            same(metrics['MAE_by_time'],np.abs(delta).mean(axis=0));same(metrics['RMSE_by_time'],np.sqrt((delta**2).mean(axis=0)))
    for baseline,comparison in report['paired_noise_head_minus_baseline'].items():
        delta=(np.abs(predictions['noise_head_noisy']-test_y[:,None,:])-np.abs(predictions[baseline+'_noisy']-test_y[:,None,:])).mean(axis=1)
        rng=np.random.default_rng(20261058);boot=delta[rng.integers(0,128,size=(2000,128))].mean(axis=1)
        interval=np.quantile(boot,[.025,.975],axis=0)
        same(delta.mean(0),comparison['mean_MAE_difference']);same(interval[0],comparison['low95']);same(interval[1],comparison['high95'])
    result=dict(passed=True,author_rows_verified=448,parent_vector_hashes=448,independent_feature_comparisons=states,
        max_feature_relative_error=max_feature_error,original_score_diagnostic_failures=int(score_failures),
        max_normal_equation_relative_residual=max(stationarity),observer_diagnostic_hashes_verified=True,
        readout_heads_reconstructed=4,selected_alphas_and_validation_curves_verified=True,test_parents=128,
        bootstrap_parent_unit_verified=True,heldout_label_fit_isolation='Heads independently reproduced using train+val only; author test labels used only for metrics.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(Path(__file__)),
        report_sha256=sha(root/'evaluation/report.json'),scope='Raw data/features/math/source/metrics audit, not a second GPU model replay or checkpoint exposure audit.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    execute(args.run)


if __name__=='__main__':main()
