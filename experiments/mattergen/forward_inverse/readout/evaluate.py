"""Validation-selected frozen readouts; test labels only used after predictions freeze."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from csp_model import sha
from regression import fit_head,predict,shuffle_parents,paired_mae_interval


def metrics(prediction,target):
    error=prediction-target[:,None,:]
    return dict(MAE=np.mean(np.abs(error),axis=(0,1)).tolist(),RMSE=np.sqrt(np.mean(error**2,axis=(0,1))).tolist(),
        MAE_by_time=np.mean(np.abs(error),axis=0).tolist(),RMSE_by_time=np.sqrt(np.mean(error**2,axis=0)).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm.')
    root=args.run;output=root/'evaluation'
    if output.exists():raise ValueError('Preserve existing evaluation.')
    selection=json.loads((root/'selection.json').read_text());complete=json.loads((root/'feature-complete.json').read_text())
    if not complete['execution_complete'] or sha(root/'features.npz')!=complete['features_sha256']:raise ValueError('Feature integrity failure.')
    with np.load(root/'features.npz') as z:
        ids=z['ids'];splits=z['splits'];labels=z['labels'];clean=z['clean_features'];noisy=z['noisy_features'];baseline=z['baseline']
        clean_alignn=z['clean_alignn'];noisy_alignn=z['noisy_alignn']
    if ids.tolist()!=[c['id'] for c in selection['cases']] or len(set(ids))!=len(ids):raise ValueError('Feature ID coverage changed.')
    masks={s:splits==s for s in ('train','val','test')}
    for split,mask in masks.items():
        if int(mask.sum())!=selection['policy']['counts'][split]:raise ValueError('Split count changed.')
    train_y=np.repeat(labels[masks['train'],None,:],len(selection['policy']['times']),axis=1).reshape(-1,2)
    val_y=np.repeat(labels[masks['val'],None,:],len(selection['policy']['times']),axis=1).reshape(-1,2)
    shuffled,permutation=shuffle_parents(labels[masks['train']])
    shuffled_y=np.repeat(shuffled[:,None,:],len(selection['policy']['times']),axis=1).reshape(-1,2)
    heads={};predictions={};frozen={}
    for name,training,evaluation,clean_evaluation,target in [
        ('composition_cell',baseline,baseline,baseline,train_y),
        ('clean_head',clean,noisy,clean,train_y),
        ('noise_head',noisy,noisy,clean,train_y),
        ('shuffled_head',noisy,noisy,clean,shuffled_y)]:
        tx=training[masks['train']].reshape(-1,training.shape[-1])
        vx=evaluation[masks['val']].reshape(-1,evaluation.shape[-1])
        head=fit_head(tx,target,vx,val_y);heads[name]=head
        predictions[name]=dict(noisy=predict(head,evaluation[masks['test']]),clean=predict(head,clean_evaluation[masks['test']]))
        for key,array in head.items():frozen[name+'_'+key]=array
    predictions['ALIGNN']=dict(noisy=noisy_alignn[masks['test']],clean=clean_alignn[masks['test']])
    output.mkdir()
    np.savez_compressed(output/'heads.npz',permutation=permutation,**frozen)
    np.savez_compressed(output/'frozen-predictions.npz',ids=ids[masks['test']],
        **{name+'_'+domain:values for name,domains in predictions.items() for domain,values in domains.items()})
    # Test targets enter metrics only after every model and prediction is frozen.
    test_y=labels[masks['test']]
    results={name:{domain:metrics(pred,test_y) for domain,pred in domains.items()} for name,domains in predictions.items()}
    comparisons={name:paired_mae_interval(predictions['noise_head']['noisy'],predictions[name]['noisy'],test_y)
                 for name in ('clean_head','composition_cell','ALIGNN','shuffled_head')}
    report=dict(policy=selection['policy'],counts={s:int(m.sum()) for s,m in masks.items()},feature_shape=list(clean.shape),
        feature_complete_sha256=sha(root/'feature-complete.json'),selection_sha256=sha(root/'selection.json'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(Path(__file__)),
        heads_sha256=sha(output/'heads.npz'),predictions_sha256=sha(output/'frozen-predictions.npz'),
        selected_alphas={name:head['selected_alphas'].tolist() for name,head in heads.items()},results=results,
        paired_noise_head_minus_baseline=comparisons,
        caveats=['Frozen backbone checkpoint structure exposure is unknown; readout labels are held out.',
                 'Small fixed label budget and one positional corruption per time; no unseen-CSP or causal pretraining claim.',
                 'Property prediction accuracy is not validation of guidance gradients or posterior likelihood calibration.',
                 'Test intervals resample parents, preserving correlated noise levels.'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot(report,output)
    print(json.dumps(dict(results=results,comparisons=comparisons)))


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    labels=dict(composition_cell='Composition + cell',clean_head='CSP clean-trained head',
        noise_head='CSP noise-trained head',shuffled_head='Shuffled-label control',ALIGNN='Pretrained ALIGNN control')
    for j,(axis,prop) in enumerate(zip(axes,report['policy']['properties'])):
        for name,result in report['results'].items():
            axis.plot(report['policy']['times'],np.array(result['noisy']['MAE_by_time'])[:,j],'o-',label=labels[name])
        axis.set(title=prop,xlabel='Diffusion time of coordinate corruption',ylabel='MAE against clean author labels')
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Frozen CSP readout heads: 256 shared training labels; 128 label-held-out test structures\nPretrained ALIGNN is a reference with a different supervision budget',fontsize=10)
    fig.savefig(output/'readout.png',dpi=180);fig.savefig(output/'readout.svg');plt.close(fig)


if __name__=='__main__':main()
