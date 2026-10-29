"""Coordinate-correct, parent-held-out direction probes; no fitted update rule."""
import numpy as np


EPS = 1e-10


def project(v):
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite vector.')
    return (v-v.mean(axis=0)).ravel()


def cartesian_score(fractional_score, cell):
    # Row coordinates r=f L: gradients transform as s_r=s_f L^{-T}.
    return project(np.asarray(fractional_score) @ np.linalg.inv(cell).T)


def unit(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v/n if n > EPS else np.zeros_like(v)


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if min(np.linalg.norm(a), np.linalg.norm(b)) > EPS else None


def design(row, features):
    y = unit(row['target'])
    x = np.stack([unit(row[k]) for k in features], axis=1)
    return x, y


def heldout(rows, features):
    """Equal weight per state; all replicas/anchors of a parent stay in one fold.

    Fit global coefficients on unit vectors, not a separate two-direction
    projection in each test state. R2 uses the zero-vector baseline (no
    coordinate-dependent intercept). Negative R2 is deliberately retained.
    """
    parents = sorted({r['parent_id'] for r in rows})
    if len(parents) < 3:
        raise ValueError('Need at least three independent parents.')
    folds, errors, labels, cosines = [], [], [], []
    for parent in parents:
        train = [design(r, features) for r in rows if r['parent_id'] != parent]
        x = np.concatenate([r[0] for r in train])
        y = np.concatenate([r[1] for r in train])
        coefficient, _, rank, singular = np.linalg.lstsq(x, y, rcond=1e-8)
        condition = float(singular.max()/singular.min()) if len(singular) and singular.min() > EPS else None
        fold_errors = []
        for row in rows:
            if row['parent_id'] != parent:
                continue
            a, b = design(row, features)
            pred = a @ coefficient
            error = float(np.sum((pred-b)**2))
            fold_errors.append(error)
            errors.append(error)
            labels.append(parent)
            cosines.append(cosine(pred, b))
        folds.append(dict(parent_id=parent, coefficient=coefficient.tolist(), rank=int(rank),
                          condition=condition, MSE=float(np.mean(fold_errors))))
    return dict(features=list(features), states=len(rows), parents=len(parents),
                heldout_R2_zero_baseline=float(1-np.mean(errors)),
                prediction_cosine_mean=float(np.mean([c for c in cosines if c is not None])) if any(c is not None for c in cosines) else None,
                folds=folds, errors=errors, error_parent_ids=labels)


def local_projection(target, force, grad):
    x = np.stack([unit(force), unit(grad)], axis=1)
    y = unit(target)
    coef, _, rank, singular = np.linalg.lstsq(x, y, rcond=1e-8)
    return dict(explained_fraction=float(1-np.sum((x@coef-y)**2)),
                coefficient=coef.tolist(), rank=int(rank), singular_values=singular.tolist())


def paired_parent_interval(first, second, seed=20261049):
    parents = sorted(set(first['error_parent_ids']))
    if first['error_parent_ids'] != second['error_parent_ids']:
        raise ValueError('Unpaired held-out rows.')
    difference = np.asarray(second['errors'])-np.asarray(first['errors'])
    means = np.array([difference[np.array(first['error_parent_ids']) == p].mean() for p in parents])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), size=(4000, len(means)))].mean(1)
    return dict(R2_gain=float(means.mean()), parent_bootstrap_CI95=np.quantile(boot, [.025, .975]).tolist(), parents=len(parents))
