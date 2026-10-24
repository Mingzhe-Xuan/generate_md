"""Recheck only engine-contaminated final B endpoints, preserving all A updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from context import SMOACS

sys.path[:0] = [str(SMOACS/'postprocess'), str(SMOACS/'alternating')]
from mattersim_support import needs_isolated_retry, validate_atomic_numbers, error_metadata
from rpc import read_response

MODEL_SHA = 'e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5'


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def close_worker(worker, log):
    try:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            worker.wait(timeout=20)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        worker.terminate()
        worker.wait(timeout=10)
    finally:
        log.close()


def start_worker(folder, tag):
    log_root = folder/'repair-workers'
    log_root.mkdir(exist_ok=True)
    log = (log_root/(tag+'.err')).open('w')
    worker = subprocess.Popen([str(Path.cwd()/'.venv-smoacs-postprocess/bin/python'), str(SMOACS/'alternating/worker.py'),
        '--checkpoint', str(Path.cwd()/'results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    try:
        ready = read_response(worker.stdout, log_root/(tag+'.stdout'))
        if not ready['ready'] or ready['checkpoint_sha256'] != MODEL_SHA:
            raise ValueError('Repair worker model provenance differs.')
        return worker, log, ready
    except BaseException:
        close_worker(worker, log)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error('Use one to four isolated workers.')
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        p.error('Use Slurm.')
    root = args.run.resolve()
    if not root.is_relative_to((Path.cwd()/'results/mattergen').resolve()):
        raise ValueError('Repair only task MatterGen outputs.')
    data = load(root/'results.json')
    if not load(root/'complete.json')['completed']:
        raise ValueError('The original optimization must be complete before endpoint repair.')
    if data['phase'] != 'complete' or data['protocol']['cycles'] != 1 or data['protocol']['steps_per_A'] != 600:
        raise ValueError('Posthoc repair requires a completed single-cycle A600/B run.')
    rows = data['cycles'][0]['candidates']
    affected = [r for r in rows if needs_isolated_retry(r.get('b_result', {}))]
    if not affected:
        print(json.dumps(dict(repaired=0, note='No retryable CUDA failures; A/B artifacts untouched.')), flush=True)
        return
    protected = {name: digest(root/name) for name in ('A-trace.jsonl', 'checkpoint.pt', 'protocol.json')}
    worker, log, ready = start_worker(root, 'model-probe-'+job)
    maximum = int(ready['model_max_z'])
    close_worker(worker, log)
    write(root/'repair-model-support.json', ready)
    plan_path = root/'cuda-repair-plan.json'
    if plan_path.exists():
        plan = load(plan_path)
        if plan['protected'] != protected or plan['model_max_z'] != maximum:
            raise ValueError('Repair inputs or model changed.')
    else:
        archive = root/('B-before-cuda-repair-'+job)
        plan = dict(job_id=job, model_max_z=maximum, protected=protected, cases=[dict(
            candidate_id=r['candidate_id'], source=r['A']['source'], source_sha256=r['A']['sha256'],
            archive=str(archive/Path(r['candidate_id']).stem), old_record=r['b_result']) for r in affected])
        write(plan_path, plan)
    by_id = {r['candidate_id']: r for r in rows}
    tasks = []
    for case in plan['cases']:
        key = case['candidate_id']
        row = by_id[key]
        if digest(case['source']) != case['source_sha256'] or row['A']['sha256'] != case['source_sha256']:
            raise ValueError('Immutable A endpoint changed.')
        folder = root/'cycle-1/B'/Path(key).stem
        archive = Path(case['archive'])
        if not folder.resolve().is_relative_to(root) or not archive.resolve().is_relative_to(root):
            raise ValueError('Repair path escaped the task.')
        if not archive.exists():
            archive.parent.mkdir(exist_ok=True)
            folder.rename(archive)
        folder.mkdir(exist_ok=True)
        result_path = folder/'result.json'
        if result_path.exists():
            previous = load(result_path)
            if previous.get('isolated_retry') and previous['input_sha256'] == case['source_sha256']:
                continue
            raise ValueError('Unexpected nonisolated record in repaired output.')
        try:
            validate_atomic_numbers(row['A']['atomic_numbers'], maximum)
        except ValueError as error:
            write(result_path, dict(candidate_id=key, input_sha256=case['source_sha256'], checkpoint_sha256=MODEL_SHA,
                status='failed', maxstep=.05, step_limit=1600, error=repr(error), **error_metadata(error),
                isolated_retry=True, repair_job_id=job, evaluated=False, original_error=case['old_record'].get('error')))
            continue
        tasks.append(dict(candidate_id=key, source=case['source'], output=str(folder), expected_sha256=case['source_sha256']))

    def evaluate(task):
        tag = Path(task['candidate_id']).stem+'-'+str(time.time_ns())
        worker, log, information = start_worker(root, tag)
        try:
            if information['model_max_z'] != maximum:
                raise ValueError('Repair model encoder changed.')
            worker.stdin.write(json.dumps(dict(tasks=[task]))+'\n')
            worker.stdin.flush()
            returned = read_response(worker.stdout, root/'repair-workers'/(tag+'.stdout'))['results']
            if len(returned) != 1 or returned[0]['input_sha256'] != task['expected_sha256']:
                raise ValueError('Isolated B response differs from immutable A input.')
            record = returned[0]
            record.update(isolated_retry=True, repair_job_id=job, model_max_z=maximum,
                          original_error=next(c['old_record'].get('error') for c in plan['cases'] if c['candidate_id']==task['candidate_id']))
            write(Path(task['output'])/'result.json', record)
            return record
        finally:
            close_worker(worker, log)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate, task) for task in tasks]):
            row = future.result()
            print(json.dumps(dict(rechecked=row['candidate_id'], status=row['status'])), flush=True)
    for case in plan['cases']:
        row = by_id[case['candidate_id']]
        record = load(root/'cycle-1/B'/Path(case['candidate_id']).stem/'result.json')
        row['b_result'] = record
        row['status'] = 'B_converged_posthoc_repair' if record['status']=='converged' else 'B_'+record['status']
        row['repair'] = dict(job_id=job, original_record=case['old_record'], final_B_only=True, A_replayed=False)
    for state in data['states']:
        if state['candidate_id'] in {c['candidate_id'] for c in plan['cases']}:
            record = by_id[state['candidate_id']]['b_result']
            state['previous_status'] = state['status']
            state['status'] = 'posthoc_B_converged_no_further_A' if record['status']=='converged' else 'B_'+record['status']
            state['repair_job_id'] = job
    data['posthoc_B_repair'] = dict(job_id=job, affected_ids=[c['candidate_id'] for c in plan['cases']], model_max_z=maximum,
                                  note='Fresh-process final B endpoints; saved optimizer/checkpoint tensors are unchanged and were not fed back for another A phase.')
    write(root/'results.json', data)
    for name, value in protected.items():
        if digest(root/name) != value:
            raise ValueError('Repair modified protected A/checkpoint/protocol bytes.')
    if (root/'evaluation-cycle1').exists():
        (root/'evaluation-cycle1').rename(root/('evaluation-cycle1-before-cuda-repair-'+job))
    plan['completed'] = True
    write(plan_path, plan)
    print(json.dumps(dict(repaired=len(plan['cases']), re_evaluated=len(tasks), model_max_z=maximum, A_replayed=False)), flush=True)


if __name__ == '__main__':
    main()
