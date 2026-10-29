"""Network-only resumable downloads; all hashing and dataset scans run in Slurm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument('--proxy')
    route.add_argument('--direct', action='store_true')
    parser.add_argument('--submit-verification', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    source = json.loads((here / 'sources.json').read_text())
    env = dict(os.environ)
    network = ['--proxy', args.proxy] if args.proxy else ['--noproxy', '*']
    if args.direct:
        for name in ('ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy'):
            env.pop(name, None)
    try:
        def fetch(spec):
            path = root / spec['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.stat().st_size != spec['size']:
                    raise ValueError('Existing downloaded file has wrong length: ' + str(path))
                return spec['path']
            partial = path.with_name(path.name + '.partial')
            with path.with_name(path.name + '.download.log').open('a') as log:
                subprocess.run(['curl', *network, '--fail', '--location', '--connect-timeout', '15',
                    '--max-time', '900', '--speed-time', '30', '--speed-limit', '1024', '--retry', '2',
                    '-C', '-', '--output', str(partial), spec['url']], env=env, stdout=log, stderr=log, check=True)
            if partial.stat().st_size != spec['size']:
                raise ValueError('Incomplete download: ' + str(partial))
            partial.rename(path)
            print('DOWNLOADED', spec['path'], spec['size'], flush=True)
            return spec['path']
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(fetch, source['assets']))
        record = dict(source=source, files=results, route=args.proxy or 'authorized-direct')
        (root / 'download-complete.json').write_text(json.dumps(record, indent=2) + '\n')
        if args.submit_verification:
            jobfile = root / 'verification-job-id'
            if jobfile.exists():
                raise ValueError('Verification already submitted; inspect recorded job before retrying.')
            result = subprocess.check_output(['sbatch', '--parsable',
                '--export=ALL,CSP_BENCHMARK_ROOT=' + str(root), str(here / 'verify.sbatch')], text=True).strip()
            jobfile.write_text(result + '\n')
            print('VERIFICATION_JOB', result, flush=True)
    except Exception:
        (root / 'download-failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
