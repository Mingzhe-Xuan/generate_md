"""Keep author initialization captures separate when resuming a Slurm job."""
from pathlib import Path


def capture_name(output, job_id, resume):
    name = 'capture-' + str(job_id)
    if not resume:
        return name
    attempt = 1
    while (Path(output) / f'{name}-resume-{attempt}').exists():
        attempt += 1
    return f'{name}-resume-{attempt}'
