"""Target adherence and coverage; missing predictions never become zero errors."""
import numpy as np

ARMS = ('cfg2', 'cfg0')


def summary(rows):
    keys = sorted({(r['step'], r['stage']) for r in rows}, key=lambda x: (x[0], x[1] != 'prior'))
    result = []
    for step, stage in keys:
        selected = [r for r in rows if (r['step'], r['stage']) == (step, stage)]
        paired = set.intersection(*[{r['parent'] for r in selected if r['arm'] == arm and r.get('gap') is not None} for arm in ARMS])
        for arm in ARMS:
            rr = [r for r in selected if r['arm'] == arm]
            good = [r for r in rr if r.get('gap') is not None]
            matched = [r for r in good if r['parent'] in paired]
            item = dict(step=step, stage=stage, arm=arm, total=len(rr), n=len(good), paired_n=len(matched),
                        known=sum(r['masked'] == 0 for r in rr),
                        mask_fraction=float(np.mean([r['masked']/r['num_atoms'] for r in rr])))
            for name, group in [('mae', good), ('paired_mae', matched)]:
                item[name] = float(np.mean([abs(r['gap']-r['target']) for r in group])) if group else None
            item['median_error'] = float(np.median([abs(r['gap']-r['target']) for r in good])) if good else None
            result.append(item)
    return result


def fixed_cohort(rows, start):
    """Intersection across BOTH arms and EVERY observed predictor state >=start."""
    late = [r for r in rows if r['stage'] == 'terminal' or (r['stage'] == 'predictor' and r['step'] >= start)]
    keys = {(r['arm'], r['stage'], r['step']) for r in late}
    groups = [{r['parent'] for r in late if (r['arm'], r['stage'], r['step']) == key and r.get('gap') is not None} for key in keys]
    return sorted(set.intersection(*groups)) if groups else []


def geometry(numbers, cell, frac, max_z):
    reasons = []
    if np.any(numbers == 101): reasons.append('masked_types')
    if np.any((numbers < 1) | ((numbers > max_z) & (numbers != 101))): reasons.append('unsupported_types')
    if not np.isfinite(cell).all() or not np.isfinite(frac).all():
        return reasons + ['nonfinite_geometry'], None
    if np.linalg.det(cell) <= 1e-6 or np.linalg.svd(cell, compute_uv=False).min() < .3:
        return reasons + ['invalid_cell'], None
    if np.linalg.det(cell)/len(numbers) < 2.: reasons.append('volume_guard')
    from ase.geometry import find_mic
    cart = frac @ cell
    vectors = (cart[:, None]-cart[None, :]).reshape(-1, 3)
    distance = np.linalg.norm(find_mic(vectors, cell, pbc=True)[0], axis=1).reshape(len(numbers), -1)
    np.fill_diagonal(distance, np.inf)
    nearest = float(distance.min()) if len(numbers) > 1 else None
    if nearest is not None and nearest < .5: reasons.append('collision_guard')
    return reasons, nearest
