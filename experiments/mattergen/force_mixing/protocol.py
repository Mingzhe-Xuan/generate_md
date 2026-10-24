"""Deterministic physical-pose hashes for stateless force requests."""
import hashlib
import json


def pose_hash(task):
    values={key:task[key] for key in ('cell','frac','numbers')}
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
