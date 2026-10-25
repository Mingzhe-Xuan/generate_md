"""Freeze the existing final Best10 CF-joint cohort before any SUN evaluation."""
GROUP='CF20_noise0.30'


def selected_cases(records,parents):
    if len(parents)!=128 or len(set(parents))!=128:raise ValueError('Expected128 original parents.')
    candidates={c['candidate_id']:c for c in records['candidates'] if c['group']==GROUP}
    cohorts={}
    for gate in ('force_eform','legacy'):
        selected=records['selections'][gate+'/looser/best_then_classify'][GROUP]
        if set(selected)!=set(parents):raise ValueError('Parent coverage differs.')
        cases=[]
        for parent in parents:
            value=selected[parent]
            if value is None:continue
            case=candidates[value['candidate_id']]
            if case['parent_id']!=parent or not 1<=case['branch_index']<=10:raise ValueError('Wrong parent or zero branch.')
            if case['prediction']['crystalformer_flags']['joint'] is True:cases.append(case)
        cohorts[gate]=cases
    cases=cohorts['force_eform']
    if {c['candidate_id'] for c in cases}!={c['candidate_id'] for c in cohorts['legacy']}:raise ValueError('CF-joint cohort depends on gate.')
    if len(cases)!=12 or len({c['parent_id'] for c in cases})!=12:raise ValueError('Expected the previously reported12 CF-joint parents.')
    return cases


def needed(elements,systems):
    return any(set(elements)<=set(s) for s in systems)
