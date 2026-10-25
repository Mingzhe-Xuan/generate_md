"""Bounded neutral assignments and property-blind template selection."""


def neutral_assignments(options, limit=10):
    if not options or limit < 1:
        raise ValueError('Expected atoms and a positive assignment limit.')
    values = [sorted(set(v)) for v in options]
    suffix = [set() for _ in range(len(values)+1)]
    suffix[-1] = {0}
    for i in range(len(values)-1, -1, -1):
        suffix[i] = {q+s for q in values[i] for s in suffix[i+1]}
    result = []

    def visit(i, charge, path):
        if len(result) >= limit:
            return
        if i == len(values):
            if charge == 0:
                result.append(tuple(path))
            return
        for q in values[i]:
            if -charge-q in suffix[i+1]:
                visit(i+1, charge+q, path+[q])
            if len(result) >= limit:
                break

    visit(0, 0, [])
    return result


def select_pilot(rows):
    remaining = [r for r in rows if r['eligible']]
    if len(remaining) < 4:
        raise ValueError('Four eligible templates required for the pilot.')
    chosen = []
    for target in (4, 8, 12, 20):
        row = min(remaining, key=lambda r: (abs(r['num_atoms']-target), r['candidate_id']))
        chosen.append(row)
        remaining.remove(row)
    return chosen


def select_cohort(rows, mode):
    if mode == 'pilot':
        return select_pilot(rows)
    if mode not in ('neutral', 'unmasked'):
        raise ValueError('Unknown initialization strategy.')
    chosen = [r for r in rows if r['eligible'] == (mode == 'neutral')]
    if not chosen or len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('Empty cohort or duplicate source IDs.')
    return chosen


def select_explicit(rows,ids):
    """Original-order subsets may include nonneutral parents in fixed-species runs."""
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Explicit parent IDs must be nonempty and unique.')
    selected=[r for r in rows if r['candidate_id'] in set(ids)]
    if [r['candidate_id'] for r in selected]!=ids:
        raise ValueError('Explicit parents must exist in original inventory order.')
    return selected
