"""Sample whole-device and own Slurm-job GPU memory, recording only task labels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def parse_gpu(text):
    values = [int(float(v.strip())) for v in text.strip().splitlines()[0].split(',')]
    if len(values) != 3 or not 0 <= values[0] <= values[1]:
        raise ValueError('Invalid GPU memory sample.')
    return dict(used_mib=values[0], total_mib=values[1], utilization_percent=values[2])


def task_labels(data):
    wanted = {b'SLURM_JOB_ID', b'MATTERGEN_CF_COHORT'}
    return {k.decode(): v.decode() for part in data.split(b'\0') if b'=' in part
            for k, v in [part.split(b'=', 1)] if k in wanted}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop-file', type=Path, required=True)
    p.add_argument('--interval', type=float, default=2)
    args = p.parse_args()
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        p.error('Use Slurm.')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(job_id=job, valid_samples=0, peak_device_used_mib=0, peak_own_job_used_mib=0,
                   peak_by_strategy_mib={}, note='Sampled NVIDIA memory in MiB; Torch exact allocator peaks are reported separately per optimizer process.')
    while not args.stop_file.exists():
        try:
            gpu = parse_gpu(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=8))
            text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True, timeout=8)
            processes, groups = [], {}
            for line in text.splitlines():
                pid, memory = map(str.strip, line.split(','))
                try:
                    labels = task_labels(Path('/proc', pid, 'environ').read_bytes())
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if labels.get('SLURM_JOB_ID') != job:
                    continue
                cohort = labels.get('MATTERGEN_CF_COHORT', 'other')
                memory = int(float(memory))
                processes.append(dict(pid=int(pid), memory_mib=memory, strategy=cohort))
                groups[cohort] = groups.get(cohort, 0)+memory
            sample = dict(timestamp=time.time(), device=gpu, own_processes=processes, strategy_mib=groups)
            with (args.output/'memory-samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample)+'\n')
            summary['valid_samples'] += 1
            summary['total_mib'] = gpu['total_mib']
            summary['peak_device_used_mib'] = max(summary['peak_device_used_mib'], gpu['used_mib'])
            summary['peak_own_job_used_mib'] = max(summary['peak_own_job_used_mib'], sum(groups.values()))
            summary['minimum_device_free_mib'] = min(summary.get('minimum_device_free_mib', gpu['total_mib']), gpu['total_mib']-gpu['used_mib'])
            for name, memory in groups.items():
                summary['peak_by_strategy_mib'][name] = max(summary['peak_by_strategy_mib'].get(name, 0), memory)
            temporary = args.output/'memory-summary.tmp'
            temporary.write_text(json.dumps(summary, indent=2)+'\n')
            temporary.replace(args.output/'memory-summary.json')
        except (subprocess.SubprocessError, ValueError, OSError) as error:
            with (args.output/'memory-errors.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(timestamp=time.time(), error=repr(error)))+'\n')
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
