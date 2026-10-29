"""Two independent observers of the same immutable native GemNet output."""
import numpy as np


def pooled(captured,n):
    if len(captured)!=1 or captured[0].ndim!=2 or len(captured[0])!=n or not captured[0].is_cuda:
        raise ValueError('Unexpected native node-embedding interface.')
    nodes=captured[0].detach().cpu().double().numpy()
    if not np.isfinite(nodes).all():raise ValueError('Nonfinite embedding.')
    return nodes.mean(0)


def hook_embedding(model,numbers,cell,cart,t):
    captured=[]
    handle=model.native.model.gemnet.register_forward_hook(lambda m,args,out:captured.append(out.node_embeddings))
    try:field=model.score(numbers,cell,cart,t,dtype='single')
    finally:handle.remove()
    return pooled(captured,len(numbers)),field


def wrapper_embedding(model,numbers,cell,cart,t):
    gemnet=model.native.model.gemnet;original=gemnet.forward;captured=[]
    def observed(*args,**kw):
        out=original(*args,**kw);captured.append(out.node_embeddings);return out
    gemnet.forward=observed
    try:field=model.score(numbers,cell,cart,t,dtype='single')
    finally:gemnet.forward=original
    return pooled(captured,len(numbers)),field
