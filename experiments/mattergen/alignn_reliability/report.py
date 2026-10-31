"""Audit frozen data/splits/heads and report selective prediction on held-out IDs."""
import argparse
import csv
import json
import os
from pathlib import Path
import zipfile
import numpy as np
from core import THRESHOLDS, GAP_SHA, sha, save
from head import Head


def auc_check(y, score):
    from scipy.stats import rankdata
    n = int(y.sum()); m = len(y)-n
    return float((rankdata(score)[y == 1].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    feature = json.loads((root/'feature-complete.json').read_text())
    trained = json.loads((root/'heads-complete.json').read_text())
    prep = json.loads((root/'prepare.json').read_text())
    if not feature['passed'] or not trained['passed'] or sha(root/'features.npz') != feature['features_sha256']:
        raise ValueError('Missing/changed features.')
    if feature['weights_before'] != feature['weights_after'] or not feature['frozen']:
        raise ValueError('Base model changed.')
    if feature['max_same_forward_reconstruction_error'] > 1e-9 or not feature['native_controls']:
        raise ValueError('Feature/native controls absent.')
    if any(max(c['max_errors']) > 1e-5 for c in feature['native_controls']): raise ValueError('Native equivalence failed.')
    with np.load(root/'features.npz', allow_pickle=False) as d: data = {k: d[k] for k in d.files}
    with np.load(root/'test-probabilities.npz', allow_pickle=False) as d: probabilities = {k: d[k] for k in d.files}
    model_zip = Path('results/smoacs/alignn-weights/mp_gappbe_alignnn.zip')
    if sha(model_zip) != GAP_SHA: raise ValueError('Predictor ZIP changed.')
    with zipfile.ZipFile(model_zip) as z:
        split = json.loads(z.read(next(n for n in z.namelist() if n.endswith('ids_train_val_test.json'))))
    if len(set(data['ids'])) != 9239 or set(data['ids']) != set(split['id_val']+split['id_test']):
        raise ValueError('Original split coverage changed.')
    if set(data['ids']) & set(split['id_train']): raise ValueError('Original training ID leakage.')
    train = data['role'] == 'meta_train'; test = data['role'] == 'test'
    if set(data['ids'][test]) != set(split['id_test']) or not np.array_equal(probabilities['ids'], data['ids'][test]):
        raise ValueError('Test prediction ID alignment failed.')
    if data['h'].shape != (9239, 256) or not np.isfinite(data['h']).all(): raise ValueError('Bad feature array.')
    if not np.array_equal(data['prediction'], np.maximum(data['raw_prediction'], 0)): raise ValueError('Gap clamp mismatch.')
    for a,b in [('meta_train','meta_val'),('meta_train','calibration'),('meta_val','calibration')]:
        if set(data['group'][data['role']==a]) & set(data['group'][data['role']==b]): raise ValueError('Meta composition leakage.')
    error = np.abs(data['prediction'][test]-data['truth'][test])
    rows = []; selected = {}; checks = []
    colors = dict(prediction_only='#d77b24', feature_linear='#31995a', feature_mlp='#286bb0')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    calfig, calaxes = plt.subplots(1, 3, figsize=(15, 4.5))
    for column, threshold in enumerate(THRESHOLDS):
        y = (error <= threshold+1e-12).astype(int)
        reports = [r for r in trained['reports'] if r['threshold'] == threshold]
        decision = next(c for c in trained['chosen'] if c['threshold'] == threshold)
        candidates = [r for r in reports if r['model'].startswith('feature_')]
        expected = max(candidates, key=lambda r: (tuple(r['validation_selection']),r['model']))
        if expected['model'] != decision['model']: raise ValueError('Test-based/changed model selection.')
        for report in reports:
            name = report['model']; stem=f'{name}-tau{threshold}'
            if sha(root/report['file']) != report['sha256']: raise ValueError('Head hash changed.')
            head = Head.load(root/report['file'])
            inp = data['h'][train] if str(head.arrays['input']) == 'h' else data['raw_prediction'][train, None]
            mean, scale = inp.mean(0), inp.std(0); scale = np.where(scale > 1e-12, scale, 1.)
            if not np.allclose(mean, head.arrays['mean'], atol=1e-10, rtol=0) or not np.allclose(scale, head.arrays['scale'], atol=1e-10, rtol=0):
                raise ValueError('Scaler not fitted only on meta training.')
            p = head.predict_proba(data['h'][test], data['raw_prediction'][test])
            if not np.array_equal(p, probabilities[stem]): raise ValueError('Saved probability/head disagreement.')
            measured_auc = auc_check(y, p)
            if measured_auc is not None and abs(measured_auc-report['test']['auroc']) > 1e-12: raise ValueError('Independent AUC mismatch.')
            if abs(np.mean((p-y)**2)-report['test']['brier']) > 1e-12: raise ValueError('Brier mismatch.')
            keep = p >= .9; accepted = report['test']['accept_0.9']
            if int(keep.sum()) != accepted['n']: raise ValueError('Acceptance count mismatch.')
            if keep.any() and abs(y[keep].mean()-accepted['correct_fraction']) > 1e-12: raise ValueError('Acceptance precision mismatch.')
            order = np.argsort(-p, kind='stable'); n = np.arange(1, len(y)+1)
            axes[0,column].plot(n/len(y), np.cumsum(1-y[order])/n, color=colors[name], label=name)
            axes[1,column].plot(n/len(y), np.cumsum(error[order])/n, color=colors[name], label=name)
            chunks = np.array_split(np.argsort(p), 10)
            xs = [p[ix].mean() for ix in chunks]; ys = [y[ix].mean() for ix in chunks]
            calaxes[column].plot(xs,ys,marker='o',ms=3,color=colors[name],label=name)
            row = dict(threshold_eV=threshold, model=name, base_correct_fraction=float(y.mean()),
                auroc=measured_auc, ap_bad=report['test']['ap_bad'], brier=report['test']['brier'],
                acceptance_p90_n=accepted['n'], acceptance_p90_coverage=accepted['coverage'],
                acceptance_p90_precision=accepted['correct_fraction'], acceptance_p90_mae=accepted['mae'],
                mae_at_50coverage=next(c['mae'] for c in report['test']['coverage_curve'] if c['coverage']>=.5),
                parameters=report['parameters'], calibration_ece=float(sum(len(ix)/len(y)*abs(p[ix].mean()-y[ix].mean()) for ix in chunks)))
            rows.append(row); checks.append(stem)
            if name == decision['model']: selected[str(threshold)] = row
        axes[0,column].axhline(1-y.mean(),color='gray',ls=':',label='All-data/random baseline')
        axes[1,column].axhline(error.mean(),color='gray',ls=':')
        axes[0,column].set_title(f'Accuracy tolerance {threshold} eV')
        axes[0,column].set_ylabel('Fraction outside tolerance among accepted')
        axes[1,column].set_ylabel('Bandgap MAE among accepted (eV)')
        for ax in axes[:,column]: ax.set_xlabel('Accepted fraction (coverage)');ax.set_xlim(.05,1.);ax.grid(alpha=.2)
        calaxes[column].plot([0,1],[0,1],ls=':',color='gray');calaxes[column].set_xlim(0,1);calaxes[column].set_ylim(0,1)
        calaxes[column].set_title(f'Tolerance {threshold} eV');calaxes[column].set_xlabel('Predicted probability of being accurate')
        calaxes[column].set_ylabel('Observed fraction within tolerance');calaxes[column].grid(alpha=.2)
    axes[0,0].legend(fontsize=7);calaxes[0].legend(fontsize=7)
    fig.suptitle('Frozen ALIGNN reliability heads | original held-out test IDs, n=4239\n256-d pre-fc features; three thresholds; no noisy/OOD guarantee',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));calfig.suptitle('Reliability calibration | separate calibration partition; test not used for tuning',fontsize=12)
    calfig.tight_layout(rect=(0,0,1,.91))
    for figure,prefix in [(fig,'risk-coverage'),(calfig,'calibration')]:
        for ext in ('png','svg','pdf'):figure.savefig(root/f'{prefix}.{ext}',dpi=180)
        plt.close(figure)
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    selected_scores=np.column_stack([probabilities[f"{selected[str(t)]['model']}-tau{t}"] for t in THRESHOLDS])
    monotonic_violations=float(np.mean(np.any(np.diff(selected_scores,axis=1)<-1e-12,axis=1)))
    save(root/'summary.json',dict(selected=selected,all_models=rows,partition_counts=trained['partition_counts'],
        base_test_mae=float(error.mean()),unseen_meta_composition_test_n=trained['unseen_test_compositions'],
        separately_trained_threshold_probability_monotonicity_violation_fraction=monotonic_violations,
        limitations=[prep['warning'],trained['warning']]))
    text=['# Frozen ALIGNN accuracy classifier','',
        'Reference DFT-gap labels, thresholds0.1/0.2/0.5eV. Original60000predictortrainingIDs excluded.',
        'Original validation5000 split by reduced composition into meta-train/validation/calibration; originaltest4239 untouched by head/scaler/calibration/model selection.',
        'Base-model validation selection exposure remains. Equilibrium ID benchmark does not certify noisy structures or adaptive RL rollouts.','',
        '|Tolerance/eV|Chosen feature head|Base within tolerance|AUROC|AP_bad|Coverage at p>=0.9|Actual precision at p>=0.9|',
        '|---|---|---:|---:|---:|---:|---:|']
    for threshold,row in selected.items():
        precision='none' if row['acceptance_p90_precision'] is None else f"{row['acceptance_p90_precision']:.4f}"
        text.append(f"|{threshold}|{row['model']}|{row['base_correct_fraction']:.4f}|{row['auroc']:.4f}|{row['ap_bad']:.4f}|{row['acceptance_p90_coverage']:.4f}|{precision}|")
    text += ['',f'Base current-pipeline test MAE: {error.mean():.6f} eV.',
        'Model choice uses meta-validation bad-class AP, not these test values. Probability cutoffs are empirical predictions, not guaranteed accuracy.',
        f'Threshold heads are trained separately; fraction violating probability monotonicity across tolerances: {monotonic_violations:.6f}. They are not a jointly calibrated error CDF.',
        'See heads-complete.json for prediction-only baseline, calibration, unseen-meta-composition/nonmetal/predicted-semiconductor subset metrics and selected head files.',
        'Head deployment uses head.Head.load(file).predict_proba(h), with h from the identical frozen predictor and graph preprocessing. No DFT label enters deployment.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    names=['comparison.csv','summary.json','REPORT.md']+[f'{prefix}.{ext}' for prefix in ('risk-coverage','calibration') for ext in ('png','svg','pdf')]
    save(root/'audit.json',dict(passed=True,feature_rows=9239,test_rows=4239,heads_checked=checks,
        source_complete_sha256=sha(root/'heads-complete.json'),features_sha256=sha(root/'features.npz'),
        predictor_training_ids_excluded=True,scalers_train_only=True,independent_auc=True,
        artifacts={name:sha(root/name) for name in names},report_code_sha256=sha(Path(__file__))))
    print(json.dumps(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(a.run.absolute())
