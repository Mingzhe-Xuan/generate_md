"""Train-only ridge geometry and whole-parent paired uncertainty."""
import numpy as np


ALPHAS=np.logspace(-6,6,13)


def nonnegative_gap(prediction):
    result=np.array(prediction,copy=True)
    result[...,0]=np.maximum(result[...,0],0)
    return result


def fit_head(train_x,train_y,val_x,val_y,alphas=ALPHAS):
    x=np.asarray(train_x,dtype=float);y=np.asarray(train_y,dtype=float)
    vx=np.asarray(val_x,dtype=float);vy=np.asarray(val_y,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),2) or vy.shape!=(len(vx),2):raise ValueError('Invalid regression arrays.')
    if any(not np.isfinite(a).all() for a in (x,y,vx,vy)):raise ValueError('Nonfinite fit data.')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-8,scale,1.)
    ymean=y.mean(0)
    standardized=(x-mean)/scale;validation=(vx-mean)/scale
    u,s,vt=np.linalg.svd(standardized,full_matrices=False)
    response=u.T@(y-ymean)
    coefficients=[];errors=[]
    for alpha in alphas:
        if alpha<=0:raise ValueError('Positive regularization required.')
        coef=(vt.T*(s/(s*s+alpha)))@response
        coefficients.append(coef)
        prediction=nonnegative_gap(validation@coef+ymean)
        errors.append(np.mean((prediction-vy)**2,axis=0))
    chosen=np.argmin(errors,axis=0)
    coefficient=np.column_stack([coefficients[index][:,j] for j,index in enumerate(chosen)])
    return dict(mean=mean,scale=scale,ymean=ymean,coefficient=coefficient,
                alphas=np.asarray(alphas),selected_alphas=np.asarray(alphas)[chosen],validation_MSE=np.asarray(errors))


def predict(head,x):
    return nonnegative_gap(((np.asarray(x)-head['mean'])/head['scale'])@head['coefficient']+head['ymean'])


def shuffle_parents(labels,seed=20261057):
    permutation=np.random.default_rng(seed).permutation(len(labels))
    return np.asarray(labels)[permutation],permutation


def paired_mae_interval(first,second,target,seed=20261058):
    first=np.asarray(first);second=np.asarray(second);target=np.asarray(target)
    if first.shape!=second.shape or first.shape!=(len(target),first.shape[1],2):raise ValueError('Invalid paired prediction shape.')
    delta=(np.abs(first-target[:,None,:])-np.abs(second-target[:,None,:])).mean(axis=1)
    rng=np.random.default_rng(seed)
    sampled=delta[rng.integers(0,len(target),size=(2000,len(target)))].mean(axis=1)
    intervals=np.quantile(sampled,[.025,.975],axis=0)
    return dict(mean_MAE_difference=delta.mean(0).tolist(),low95=intervals[0].tolist(),high95=intervals[1].tolist(),
                parents=len(target),unit='Whole test structure; all four noise levels remain together.')
