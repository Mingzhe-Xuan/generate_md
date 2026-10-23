"""Move actual Crystalformer optimization state to CUDA and observe its execution."""
import json
import torch


def move_context(context,device,dtype):
    for parameter in context['targets']:parameter.data=parameter.data.to(device=device,dtype=dtype)
    for model in (context['gap_model'],context['ef_model']):model.to(device=device,dtype=dtype)
    for key,value in context['data'].items():
        if torch.is_tensor(value):context['data'][key]=value.to(device=device,dtype=dtype if value.is_floating_point() else value.dtype)
    # The forward closure holds these tensor objects, so preserve their identities.
    for key in ('base','lower','upper','angle_lower','angle_upper'):
        context[key].data=context[key].data.to(device=device,dtype=dtype)
    context['batch']=context['data']['batch'];context['size']=context['data']['size']


def require_cuda(context):
    tensors=list(context['targets'])+[context[k] for k in ('base','lower','upper','angle_lower','angle_upper','batch','size')]
    for model in (context['gap_model'],context['ef_model']):tensors+=list(model.parameters())+list(model.buffers())
    tensors += [v for v in context['data'].values() if torch.is_tensor(v)]
    if not tensors or not all(t.is_cuda for t in tensors):raise RuntimeError('Crystalformer models/targets/inputs must all be on CUDA; CPU fallback forbidden.')


def prepare_cuda(context):
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the Adam experiment.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with torch.no_grad():gap,ef,_,_=context['forward'](.01)
    move_context(context,'cuda',torch.float32);require_cuda(context)
    with torch.no_grad():gpu_gap,gpu_ef,_,_=context['forward'](.01)
    if not gpu_gap.is_cuda or not gpu_ef.is_cuda:raise RuntimeError('Predictor outputs are not CUDA tensors.')
    error=max(float((gpu_gap.cpu()-gap.cpu()).abs().max()),float((gpu_ef.cpu()-ef.cpu()).abs().max()))
    if error>1e-3:raise ValueError('CPU/GPU initial predictor control failed: '+str(error))
    return dict(a_device='cuda',a_dtype='float32',gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,cuda=torch.version.cuda,cpu_gpu_control_error=error,
        backend='PyTorch CUDA indexed encoder with corrected Gaussian input derivatives',tf32=False)


class GPUAudit:
    def __init__(self,context,path):
        require_cuda(context);self.path=path;self.handles=[];self.closed=False
        self.report=dict(forward_calls=0,backward_calls=0,finite_backward_calls=0,all_cuda=True,passed=False)
        def backward(gradient):
            if not gradient.is_cuda:
                self.report['all_cuda']=False
                raise RuntimeError('Crystalformer backward executed on CPU.')
            self.report['backward_calls']+=1
            self.report['finite_backward_calls']+=int(bool(torch.isfinite(gradient).all()))
        def forward(module,inputs,output):
            if not torch.is_tensor(output) or not output.is_cuda:
                self.report['all_cuda']=False
                raise RuntimeError('Crystalformer encoder forward executed on CPU.')
            self.report['forward_calls']+=1
            if output.requires_grad:output.register_hook(backward)
        for model in (context['gap_model'],context['ef_model']):self.handles.append(model.encoder.register_forward_hook(forward))

    def close(self):
        if self.closed:return
        for handle in self.handles:handle.remove()
        self.report['passed']=self.report['all_cuda'] and self.report['forward_calls']>0 and self.report['backward_calls']>0 and self.report['finite_backward_calls']>0
        self.path.write_text(json.dumps(self.report,indent=2)+'\n');self.closed=True
