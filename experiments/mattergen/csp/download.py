"""Bounded parallel HTTP downloads only; full checksum/assembly belongs in Slurm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from validation import byte_ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    route = parser.add_mutually_exclusive_group(required=True)
    route.add_argument('--direct', action='store_true', help='Requires existing direct-download authorization')
    route.add_argument('--proxy')
    parser.add_argument('--assets', type=Path, default=Path('results/mattergen/csp-assets'))
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name('sources.json').read_text())
    folder = args.assets / 'parts'
    folder.mkdir(parents=True, exist_ok=True)
    network = ['--noproxy', '*'] if args.direct else ['--proxy', args.proxy]
    env = dict(os.environ)
    if args.direct:
        for name in ('ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'http_proxy', 'https_proxy'):
            env.pop(name, None)
    def fetch(item):
        index, (start, end) = item
        path = folder / f'part-{index:02d}'
        if path.exists():
            if path.stat().st_size != end - start + 1:
                raise ValueError('Existing part has wrong size; inspect it before retrying.')
            return dict(index=index, start=start, end=end, bytes=path.stat().st_size)
        temporary = path.with_suffix('.partial')
        with path.with_suffix('.log').open('w') as log:
            subprocess.run(['curl', *network, '--fail', '--location', '--connect-timeout', '15',
                '--max-time', '600', '--speed-time', '30', '--speed-limit', '1024',
                '--range', f'{start}-{end}', '--dump-header', str(path.with_suffix('.headers')),
                '--output', str(temporary), spec['checkpoint_url']], env=env, stdout=log, stderr=log, check=True)
        if temporary.stat().st_size != end - start + 1:
            raise ValueError('Server did not supply exactly the requested range.')
        temporary.rename(path)
        result = dict(index=index, start=start, end=end, bytes=path.stat().st_size)
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, enumerate(byte_ranges(spec['checkpoint_size_bytes']))))
    (folder / 'download-complete.json').write_text(json.dumps(dict(parts=results,
        route='authorized-direct' if args.direct else args.proxy, source=spec), indent=2) + '\n')


if __name__ == '__main__':
    main()
