"""Extract optimization counts and second-resolution timing from archived logs."""
import math
import re
from statistics import mean


def duration(text):
    parts = [int(p) for p in text.split(':')]
    if len(parts) not in (2, 3) or any(p < 0 for p in parts):
        raise ValueError('Expected MM:SS or HH:MM:SS.')
    if parts[-1] >= 60 or (len(parts) == 3 and parts[-2] >= 60):
        raise ValueError('Invalid clock fields.')
    return sum(p*60**i for i,p in enumerate(reversed(parts)))


def crystalformer_steps(text, expected):
    observations = {}
    pattern = r'(?:^|[\r\n])step:[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[(\d+(?::\d+){1,2})(?=[<\],])'
    for match in re.finditer(pattern, text):
        step, total, stamp = match.groups()
        if int(total) != expected:
            raise ValueError('Unexpected optimization loop length.')
        step, elapsed = int(step), duration(stamp)
        if step in observations and observations[step] != elapsed:
            raise ValueError('Conflicting repeated progress observations.')
        observations[step] = elapsed
    if set(observations) != set(range(expected+1)) or observations[0] != 0:
        raise ValueError('Missing step observations or nonzero initial elapsed time.')
    rows = []
    for step in range(1, expected+1):
        interval = observations[step]-observations[step-1]
        if interval < 0:
            raise ValueError('Nonmonotone elapsed time.')
        rows.append(dict(step=step, elapsed_seconds=observations[step], seconds=interval))
    return rows


def mattersim_steps(text, expected=None):
    observations = []
    for match in re.finditer(r'^BFGS:\s+(\d+)\s+(\d\d:\d\d:\d\d)\s', text, re.MULTILINE):
        step, stamp = match.groups()
        observations.append((int(step), duration(stamp)))
    if not observations:
        if expected is not None:
            raise ValueError('Completed result has no BFGS timing log.')
        return dict(logged_updates=None, logged_span_seconds=None, mean_seconds_per_step=None, intervals=[])
    if [i for i,t in observations] != list(range(len(observations))):
        raise ValueError('BFGS step indices must be contiguous from initial step0.')
    if expected is not None and observations[-1][0] != expected:
        raise ValueError('BFGS log disagrees with optimizer_steps.')
    intervals = []
    for (before, start),(step, end) in zip(observations, observations[1:]):
        difference = end-start
        if difference < -12*3600:
            difference += 24*3600
        if difference < 0:
            raise ValueError('Nonmonotone BFGS clock, not a midnight rollover.')
        intervals.append(dict(step=step, seconds=difference))
    updates = observations[-1][0]
    total = sum(r['seconds'] for r in intervals)
    return dict(logged_updates=updates, logged_span_seconds=total,
                mean_seconds_per_step=total/updates if updates else None, intervals=intervals)


def describe(values):
    values = sorted(values)
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite descriptive statistics input.')
    def percentile(q):
        position = (len(values)-1)*q
        lower = int(position)
        upper = min(lower+1, len(values)-1)
        return values[lower]+(values[upper]-values[lower])*(position-lower)
    if not values:
        return dict(n=0, minimum=None, p05=None, p25=None, median=None, mean=None,
                    p75=None, p95=None, maximum=None)
    return dict(n=len(values), minimum=values[0], p05=percentile(.05), p25=percentile(.25),
                median=percentile(.5), mean=mean(values), p75=percentile(.75),
                p95=percentile(.95), maximum=values[-1])
