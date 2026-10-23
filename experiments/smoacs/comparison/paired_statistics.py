"""Paired relaxation statistics with fixed denominators and explicit missingness."""
import math
from statistics import mean, median


METRICS = ('stable', 'unique', 'novel', 'sun', 'gap_success', 'eform_success', 'valid_structure',
           'property_success', 'stable_and_property', 'sun_and_property')


def conjunction(*flags):
    if any(x is False for x in flags):
        return False
    return True if all(x is True for x in flags) else None


def index_rows(rows, expected_ids):
    expected = list(expected_ids)
    indexed = {r['candidate_id']: r for r in rows}
    if len(indexed) != len(rows) or len(set(expected)) != len(expected):
        raise ValueError('Duplicate candidate IDs.')
    if set(indexed) != set(expected):
        raise ValueError('Candidate IDs do not exactly match the fixed inventory.')
    return indexed


def property_flags(gap, eform, valid, target=2.5, tolerance=.2, ceiling=-.5):
    if not all(math.isfinite(v) for v in (gap, eform, target, tolerance, ceiling)) or tolerance < 0:
        raise ValueError('Invalid property values or thresholds.')
    gap_ok = target-tolerance <= gap <= target+tolerance
    ef_ok = eform < ceiling
    return dict(gap=gap, eform=eform, gap_absolute_error=abs(gap-target),
                gap_success=gap_ok, eform_success=ef_ok,
                valid_structure=valid, property_success=conjunction(gap_ok, ef_ok, valid))


def summarize(rows):
    """A None outcome is never silently removed from the denominator."""
    report = dict(denominator=len(rows), metrics={}, continuous={})
    for metric in METRICS:
        before = [r['before'].get(metric) for r in rows]
        after = [r['after'].get(metric) for r in rows]
        if any(v is not None and type(v) is not bool for v in before+after):
            raise ValueError('Metric flags must be bool or None.')
        def counts(values):
            passed = sum(v is True for v in values)
            unknown = sum(v is None for v in values)
            return dict(passed=passed, failed=sum(v is False for v in values), unknown=unknown,
                        rate_lower_bound=passed/len(values) if values else None,
                        rate_upper_bound=(passed+unknown)/len(values) if values else None)
        b, a = counts(before), counts(after)
        transitions = dict(retained=0, lost=0, gained=0, neither=0, unknown=0)
        labels = {(True, True):'retained', (True, False):'lost',
                  (False, True):'gained', (False, False):'neither'}
        for x,y in zip(before, after):
            transitions[labels.get((x,y), 'unknown')] += 1
        report['metrics'][metric] = dict(before=b, after=a, transitions=transitions,
            delta_lower_bound_percentage_points=100*(a['rate_lower_bound']-b['rate_lower_bound']) if rows else None)
    for field in ('energy_eV_atom', 'hull_eV_atom', 'gap', 'gap_absolute_error', 'eform'):
        pairs = [(r['before'].get(field), r['after'].get(field)) for r in rows]
        pairs = [(b,a) for b,a in pairs if b is not None and a is not None]
        if any(not all(math.isfinite(v) for v in p) for p in pairs):
            raise ValueError('Nonfinite paired continuous metric.')
        delta = [a-b for b,a in pairs]
        report['continuous'][field] = dict(paired_count=len(pairs),
            before_mean=mean(b for b,a in pairs) if pairs else None,
            after_mean=mean(a for b,a in pairs) if pairs else None,
            delta_mean=mean(delta) if delta else None, delta_median=median(delta) if delta else None,
            decreased=sum(d < 0 for d in delta), unchanged=sum(d == 0 for d in delta),
            increased=sum(d > 0 for d in delta))
    return report
