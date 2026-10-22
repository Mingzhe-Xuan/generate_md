"""Prevent legacy VaspJob termination from falling back to global killall."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import psutil

VASP_BINARIES = {'/opt/vasp.5.4.4/bin/vasp_std', '/opt/vasp.5.4.4/bin/vasp_gam'}


def terminate_launchers(directory, root, job_id, children):
    directory, root = Path(directory).resolve(), Path(root).resolve()
    if not directory.is_relative_to(root) or not job_id:
        raise ValueError('Refusing termination outside this DFT attempt.')
    actions = []
    for process in children:
        try:
            command = process.cmdline()
            if (not command or Path(command[0]).name != 'srun'
                    or not VASP_BINARIES.intersection(command)
                    or Path(process.cwd()).resolve() != directory
                    or process.environ().get('SLURM_JOB_ID') != job_id):
                continue
            action = dict(pid=process.pid, command=command, cwd=str(directory), job_id=job_id)
            # SIGTERM to the owned srun cancels its own step, not other VASP jobs.
            process.terminate()
            actions.append(action)
        except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    with (directory/'scoped-termination.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(actions=actions, global_fallback=False))+'\n')
    return actions


@contextmanager
def scoped_vasp_jobs(root):
    import psutil
    import atomate2.vasp.run as runner
    from custodian.vasp.jobs import VaspJob

    class ScopedVaspJob(VaspJob):
        def terminate(self, directory='./'):
            children = psutil.Process().children(recursive=True)
            return terminate_launchers(directory, root, os.environ.get('SLURM_JOB_ID'), children)

    # Give Monty's diagnostic encoder an importable class identity.
    ScopedVaspJob.__module__ = __name__
    ScopedVaspJob.__qualname__ = 'ScopedVaspJob'
    globals()['ScopedVaspJob'] = ScopedVaspJob
    previous = runner.VaspJob
    runner.VaspJob = ScopedVaspJob
    try:
        yield
    finally:
        runner.VaspJob = previous
