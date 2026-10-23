"""Restore phase-boundary state without replaying completed Adam updates."""
import json
from pathlib import Path

INVARIANTS=('ids','baseline_npz_sha256','author_source_sha256','cycles','steps_per_A','a_base_rates',
            'temperature','lr_cycle_policy','lr_period','b_fmax','b_maxstep','b_max_steps','denominator','a_device','a_dtype')


def restore(output,protocol,targets,optimizers,bounds):
    import torch
    output=Path(output);data=json.loads((output/'results.json').read_text())
    old=data['protocol']
    for key in INVARIANTS:
        if json.dumps(old[key])!=json.dumps(protocol[key]):raise ValueError('Resume protocol mismatch: '+key)
    state=torch.load(output/'checkpoint.pt',map_location=targets[0].device,weights_only=True)
    cycle,phase=state['cycle'],state['phase']
    if (cycle,phase)!=(data['current_cycle'],data['phase']):raise ValueError('JSON/checkpoint phase mismatch.')
    if phase not in ('initial','B_pending','cycle_completed'):raise ValueError('Not a resumable phase boundary.')
    if len(state['targets'])!=len(targets) or len(state['optimizers'])!=len(optimizers) or len(state['bounds'])!=len(bounds):
        raise ValueError('Checkpoint parameter groups differ.')
    with torch.no_grad():
        for parameter,value in zip(list(targets)+list(bounds),state['targets']+state['bounds']):
            if parameter.shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid checkpoint tensor.')
            parameter.copy_(value)
    for optimizer,value in zip(optimizers,state['optimizers']):optimizer.load_state_dict(value)
    active=state['active']
    if active.shape!=(len(old['ids']),) or active.dtype!=torch.bool:raise ValueError('Invalid active mask.')
    return data,active,cycle if phase=='B_pending' else cycle+1,phase=='B_pending'
