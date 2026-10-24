"""Separate device numerical agreement from discrete kNN graph sensitivity."""
import json

import torch


def topology_signature(record):
    graph,_,images=record
    src,dst=graph.edges(order='eid')
    rows=zip(src.tolist(),dst.tolist(),images.to(dtype=torch.int64).tolist())
    return sorted((s,d,*image) for s,d,image in rows)


def prepare(context,predictor):
    from gpu import move_context,require_cuda
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for ALIGNN optimization.')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    original=predictor.predict
    captured=[]
    replay=None
    position=0

    def controlled(cells,frac,onehot,sizes,graphs=None,native_control=False):
        nonlocal position
        if graphs is None:
            if replay is not None:
                graphs=replay[position]
                position+=1
            else:
                graphs=predictor.graphs(cells,frac,onehot,sizes)
                captured.append(graphs)
        return original(cells,frac,onehot,sizes,graphs=graphs,native_control=native_control)

    predictor.predict=controlled
    try:
        with torch.no_grad():
            cpu_gap,cpu_ef,_,_=context['forward'](.01)
        cpu_graphs=captured
        move_context(context,'cuda',torch.float32)
        require_cuda(context)
        replay=cpu_graphs
        with torch.no_grad():
            gpu_gap,gpu_ef,_,_=context['forward'](.01)
        if position!=len(cpu_graphs):
            raise ValueError('CPU/GPU graph replay groups differ.')
        error=max(float((gpu_gap.cpu()-cpu_gap).abs().max()),float((gpu_ef.cpu()-cpu_ef).abs().max()))
        if error>1e-3:
            raise ValueError('Same-topology CPU/GPU control failed: '+str(error))
        replay=None
        captured=[]
        with torch.no_grad():
            fresh_gap,fresh_ef,_,_=context['forward'](.01)
        cpu_flat=[g for group in cpu_graphs for g in group]
        gpu_flat=[g for group in captured for g in group]
        if len(cpu_flat)!=len(context['ids']) or len(gpu_flat)!=len(cpu_flat):
            raise ValueError('Graph control cohort changed.')
        changed=[]
        for identifier,first,second in zip(context['ids'],cpu_flat,gpu_flat):
            if topology_signature(first)!=topology_signature(second):
                changed.append(dict(candidate_id=identifier,cpu_edges=first[0].num_edges(),gpu_edges=second[0].num_edges()))
        changed_ids={r['candidate_id'] for r in changed}
        unchanged=[i for i,k in enumerate(context['ids']) if k not in changed_ids]
        unaffected_error=max(float((a[unchanged]-b[unchanged]).abs().max()) for a,b in ((fresh_gap,gpu_gap),(fresh_ef,gpu_ef))) if unchanged else 0.
        if unaffected_error>1e-3:
            raise ValueError('Fresh-graph discrepancy is not explained by topology changes.')
        sensitivity=dict(changed_topologies=changed,
            same_graph_CPU_GPU_max_error=error,
            unchanged_topology_fresh_graph_max_error=unaffected_error,
            fresh_graph_prediction_change=max(float((fresh_gap-gpu_gap).abs().max()),float((fresh_ef-gpu_ef).abs().max())),
            per_case_fresh_graph_gap_change=(fresh_gap-gpu_gap).tolist(),
            per_case_fresh_graph_eform_change=(fresh_ef-gpu_ef).tolist(),
            scope='Same signed topology isolates CPU/GPU arithmetic; fresh GPU graphs are restored for every optimization forward. No tolerance changes.')
        (context['output']/'device-graph-control.json').write_text(json.dumps(sensitivity,indent=2)+'\n')
    finally:
        predictor.predict=original
    return dict(a_device='cuda',a_dtype='float32',gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda,
        cpu_gpu_control_error=error,cpu_gpu_control_scope='Identical signed neighbor topology; fresh-graph sensitivity reported separately.',
        backend=context['backend'],tf32=False,initial_graph_sensitivity=sensitivity)
