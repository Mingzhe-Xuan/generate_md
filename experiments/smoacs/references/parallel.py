"""Validate and run independent reference optimizers in one Slurm GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(candidates, workers):
    if workers < 1:
        raise ValueError('At least one worker required.')
    groups = [[] for _ in range(workers)]
    loads = [0]*workers
    has_large = any(c['num_atoms'] > 128 for c in candidates)
    for candidate in sorted(candidates, key=lambda c:(-c['num_atoms'], c['candidate_id'])):
        choices = range(workers)
        if workers > 1 and has_large:
            choices = [0] if candidate['num_atoms'] > 128 else range(1, workers)
        chosen = min(choices, key=lambda i:loads[i])
        groups[chosen].append(candidate)
        loads[chosen] += candidate['num_atoms']
    ids = [c['candidate_id'] for group in groups for c in group]
    if len(ids) != len(set(ids)) or set(ids) != {c['candidate_id'] for c in candidates}:
        raise ValueError('Partition is not disjoint and exhaustive.')
    return groups


def run_workers(manifest, inventory, candidates, output, workspace, workers):
    workspace.mkdir(parents=True, exist_ok=False)
    script = Path('experiments/smoacs/postprocess/relax.py').resolve()
    checkpoint = Path('results/smoacs/postprocess-206/models/mattersim-v1.0.0-5M.pth').resolve()
    processes, handles = [], []
    started = time.monotonic()
    try:
        groups = partition(candidates, workers)
        print(json.dumps(dict(workspace=str(workspace), partitions=[dict(worker=i, count=len(g),
                         max_atoms=max((c['num_atoms'] for c in g), default=0)) for i,g in enumerate(groups)])), flush=True)
        for index, group in enumerate(groups):
            if not group:
                continue
            shard = dict(inventory, candidates=group, source_manifest_sha256=digest(manifest), worker=index)
            path = workspace/f'worker-{index}.json'
            path.write_text(json.dumps(shard, indent=2)+'\n')
            stdout = (workspace/f'worker-{index}.out').open('w')
            stderr = (workspace/f'worker-{index}.err').open('w')
            handles.extend([stdout, stderr])
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', SMOACS_RELAX_WORKER_ID=str(index))
            processes.append(subprocess.Popen([sys.executable, str(script), '--inventory', str(path),
                              '--output', str(output), '--checkpoint', str(checkpoint)],
                             env=environment, stdout=stdout, stderr=stderr))
        codes = [process.wait() for process in processes]
        if any(c != 0 for c in codes):
            raise RuntimeError('A reference worker failed: '+repr(codes))
    finally:
        for handle in handles:
            handle.close()
    return time.monotonic()-started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--validated-benchmark', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    job = os.environ['SLURM_JOB_ID']
    if args.workers > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('Workers exceed allocated CPUs.')
    manifest = args.selection/'mattersim-inventory.json'
    inventory = json.loads(manifest.read_text())
    script = Path('experiments/smoacs/postprocess/relax.py')
    main_output = args.selection/'mattersim-relaxation'
    workspace = args.selection/(('parallel-benchmark-' if args.benchmark else 'parallel-run-')+job)
    workspace.mkdir(exist_ok=False)
    if args.benchmark:
        candidates = []
        originals = {}
        for candidate in inventory['candidates']:
            path = main_output/Path(candidate['candidate_id']).stem/'result.json'
            if path.exists() and candidate['num_atoms'] <= 64:
                record = json.loads(path.read_text())
                if record['status'] == 'converged' and record['input_sha256'] == candidate['sha256']:
                    candidates.append(candidate)
                    originals[candidate['candidate_id']] = record
            if len(candidates) == 16:
                break
        if len(candidates) != 16:
            raise ValueError('Need16 completed reference controls for benchmark.')
        benchmark_inventory = dict(inventory, role='MP_reference_parallel_benchmark', denominator=len(candidates))
        serial = run_workers(manifest, benchmark_inventory, candidates, workspace/'serial', workspace/'serial-workers', 1)
        parallel = run_workers(manifest, benchmark_inventory, candidates, workspace/'parallel', workspace/'parallel-workers', args.workers)
        errors, volumes = [], []
        for candidate in candidates:
            identifier = candidate['candidate_id']
            one = json.loads((workspace/'serial'/Path(identifier).stem/'result.json').read_text())
            many = json.loads((workspace/'parallel'/Path(identifier).stem/'result.json').read_text())
            for record in (one, many):
                if record['status'] != 'converged' or record['max_filter_force'] > .01 or record['input_sha256'] != candidate['sha256']:
                    raise ValueError('Benchmark optimizer failed its unchanged convergence/provenance criterion.')
            for other in (many, originals[identifier]):
                errors.append(abs(one['final_energy_eV']-other['final_energy_eV'])/candidate['num_atoms'])
                volumes.append(abs(one['final_volume_A3']-other['final_volume_A3'])/one['final_volume_A3'])
        passed = max(errors) <= 1e-3 and max(volumes) <= .001 and parallel < .9*serial
        result = dict(passed=passed, serial_seconds=serial, parallel_seconds=parallel,
                      speedup=serial/parallel, max_energy_delta_eV_atom=max(errors),
                      max_relative_volume_delta=max(volumes), source_manifest_sha256=digest(manifest),
                      relax_source_sha256=digest(script), workers=args.workers, controls=len(candidates))
        (workspace/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if not passed:
            raise RuntimeError('Parallel equivalence/speed benchmark failed; do not switch production.')
    else:
        if not args.validated_benchmark:
            raise ValueError('Require a validated serial/parallel benchmark.')
        benchmark = json.loads((args.validated_benchmark/'result.json').read_text())
        if (benchmark.get('passed') is not True or benchmark['source_manifest_sha256'] != digest(manifest)
                or benchmark['relax_source_sha256'] != digest(script) or benchmark['workers'] != args.workers):
            raise ValueError('Parallel production benchmark provenance mismatch.')
        elapsed = run_workers(manifest, inventory, inventory['candidates'], main_output, workspace/'workers', args.workers)
        print(json.dumps(dict(reference_records=len(inventory['candidates']), elapsed_seconds=elapsed,
                              workers=args.workers, validated_benchmark=str(args.validated_benchmark))), flush=True)


if __name__ == '__main__':
    main()
