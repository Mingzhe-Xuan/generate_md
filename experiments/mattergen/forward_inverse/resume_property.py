"""Strict continuation metadata: preserve every terminal parent and original protocol."""
import json
from pathlib import Path
from csp_model import sha


def continuation(source, selection_path, policy):
    plan=json.loads((source/'plan.json').read_text())
    if plan['selection_sha256']!=sha(selection_path) or sha(source/'selection.json')!=sha(selection_path):
        raise ValueError('Resume selection changed.')
    if plan['policy']!=policy:raise ValueError('Resume scientific policy changed.')
    progress=json.loads((source/'progress.json').read_text())
    selected=[r['id'] for r in plan['selection']['selected']]
    ids=[r['id'] for r in progress]
    if ids!=selected[:len(ids)] or len(ids)!=len(set(ids)):raise ValueError('Resume coverage is not an exact ordered prefix.')
    carried={}
    for row in progress:
        if row['status']=='pending':raise ValueError('Cannot carry nonterminal parent.')
        result=source/row['id']/'result.json'
        if json.loads(result.read_text())!=row:raise ValueError('Resume progress differs from original parent result.')
        carried[row['id']]=dict(row,resumed_result=str(result),resumed_result_sha256=sha(result))
    return carried,dict(source=str(source),source_plan_sha256=sha(source/'plan.json'),
                        source_progress_sha256=sha(source/'progress.json'),
                        source_failure_sha256=sha(source/'failure.json') if (source/'failure.json').exists() else None)
