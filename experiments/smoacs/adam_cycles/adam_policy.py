"""Fixed Adam schedule and paired accounting for cyclic relaxation."""
import math

BASE_RATES=(.002,.007,.005)


def schedule_position(cycle, local_step, steps, cycles, schedule, phase_end=False):
    if not 1<=cycle<=cycles or not 0<=local_step<=steps:
        raise ValueError('Schedule position is outside its declared budget.')
    if schedule=='global':
        position, period=(cycle-1)*steps+local_step, steps*cycles
    elif schedule=='restart200':
        position, period=(200 if phase_end else local_step%200),200
    elif schedule=='per_phase':
        position, period=local_step,steps
    else:
        raise ValueError('Unknown learning-rate schedule.')
    if period not in (200,600):
        raise ValueError('Cosine period must match the200 or600 update budget.')
    return position,period


def cosine_fraction(step,steps=200):
    if not 0<=step<=steps or steps not in (200,600):raise ValueError('Expected a200- or600-update A phase.')
    return .5*(1+math.cos(math.pi*step/steps))


def set_rates(optimizers,step,steps=200):
    if len(optimizers)!=3:raise ValueError('Expected original three Adam groups.')
    for optimizer,base in zip(optimizers,BASE_RATES):optimizer.param_groups[0]['lr']=base*cosine_fraction(step,steps)


def valid_proposal(loss,cell,coords):
    import torch
    return all(bool(torch.isfinite(v).all()) for v in (loss,cell,coords)) and float(torch.linalg.det(cell).detach().abs())>1e-6


def freeze_inactive(optimizer,parameters,before,masks):
    """Freeze stopped parameters and clear their unused moment slices."""
    import torch
    with torch.no_grad():
        for parameter,old,mask in zip(parameters,before,masks):
            parameter.copy_(torch.where(mask,parameter,old))
            state=optimizer.state.get(parameter,{})
            for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if name in state:state[name].copy_(torch.where(mask,state[name],0.))


def flags(gap,energy,valid):
    if gap is None or energy is None:return dict(gap=None,eform=None,valid=None,joint=None)
    return dict(gap=2.3<=gap<=2.7,eform=energy<-.5,valid=bool(valid),
                joint=2.3<=gap<=2.7 and energy<-.5 and bool(valid))


def rate(values,denominator):
    if len(values)!=denominator:raise ValueError('Missing values must be retained explicitly.')
    passed=sum(v is True for v in values);unknown=sum(v is None for v in values)
    return dict(pass_count=passed,unknown_count=unknown,denominator=denominator,
                confirmed_rate=passed/denominator,possible_rate_bounds=[passed/denominator,(passed+unknown)/denominator])


def paired_flags(first,third,ids):
    if len(set(ids))!=len(ids) or set(first)!=set(ids) or set(third)!=set(ids):raise ValueError('Paired IDs differ.')
    transitions=dict(pass_to_pass=0,pass_to_fail=0,fail_to_pass=0,fail_to_fail=0,unknown=0)
    for identifier in ids:
        a,b=first[identifier],third[identifier]
        key='unknown' if a is None or b is None else ('pass' if a else 'fail')+'_to_'+('pass' if b else 'fail')
        transitions[key]+=1
    a,b=rate(list(first.values()),len(ids)),rate(list(third.values()),len(ids))
    transitions.update(cycle1=a,cycle3=b,confirmed_difference_pp=100*(b['confirmed_rate']-a['confirmed_rate']),
        possible_difference_pp=[100*(b['possible_rate_bounds'][0]-a['possible_rate_bounds'][1]),
                                100*(b['possible_rate_bounds'][1]-a['possible_rate_bounds'][0])])
    return transitions
