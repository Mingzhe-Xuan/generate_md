"""Acquire public MP reference structures without an API key, inside Slurm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq
import requests

PREFIX = 'https://materialsproject-build.s3.amazonaws.com/'
SUMMARY = 'collections/summary/version=2026-04-13/part-00000-d0a74fbe-68f8-4a2d-9d43-67cb6a1bc912-c000.zstd.parquet'
THERMO = [
    'collections/thermo/version=2026-04-13/thermo_type=GGA_GGA%2BU/part-00000-7a9c2707-8568-4474-b5c0-2bbb1119f3ad-c000.zstd.parquet',
    'collections/thermo/version=2026-04-13/thermo_type=GGA_GGA%2BU_R2SCAN/part-00000-81a3e72d-8f16-48d2-9244-b474f1957850-c000.zstd.parquet',
    'collections/thermo/version=2026-04-13/thermo_type=r2SCAN/part-00000-ffd18923-d96e-4c5a-be0f-2c5f831e08a3-c000.zstd.parquet',
]


class CachedHTTP(io.RawIOBase):
    """Seekable range reader; immutable object identity guards resumed cache use."""
    block_size = 1024*1024

    def __init__(self, key, cache, proxy):
        super().__init__()
        self.url = PREFIX+quote(key, safe='/=')
        self.proxy = {'http': proxy, 'https': proxy}
        response = requests.head(self.url, proxies=self.proxy, timeout=(15, 40))
        response.raise_for_status()
        self.size = int(response.headers['Content-Length'])
        self.etag = response.headers['ETag']
        self.position = 0
        self.cache = Path(cache)/hashlib.sha256(key.encode()).hexdigest()
        self.cache.mkdir(parents=True, exist_ok=True)
        meta = dict(key=key, size=self.size, etag=self.etag, url=self.url)
        path = self.cache/'object.json'
        if path.exists() and json.loads(path.read_text()) != meta:
            raise ValueError('Remote object differs from cached identity.')
        path.write_text(json.dumps(meta, indent=2))
        self.pool = ThreadPoolExecutor(max_workers=4)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if target < 0:
            raise ValueError('Negative seek.')
        self.position = target
        return target

    def _part(self, index):
        start = index*self.block_size
        end = min(start+self.block_size, self.size)-1
        path = self.cache/f'{index:06d}.part'
        digest = path.with_suffix('.sha256')
        if path.exists() and digest.exists():
            data = path.read_bytes()
            if len(data) == end-start+1 and hashlib.sha256(data).hexdigest() == digest.read_text():
                return data
            raise ValueError('Cached byte-range hash mismatch.')
        response = requests.get(self.url, headers={'Range': f'bytes={start}-{end}',
                                'If-Match': self.etag}, proxies=self.proxy, timeout=(15, 120))
        response.raise_for_status()
        if (response.status_code != 206 or response.headers.get('ETag') != self.etag
                or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}'):
            raise ValueError('Unverified HTTP range response.')
        data = response.content
        if len(data) != end-start+1:
            raise ValueError('Truncated HTTP range.')
        path.write_bytes(data)
        digest.write_text(hashlib.sha256(data).hexdigest())
        return data

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size-self.position
        end = min(self.position+size, self.size)
        if end <= self.position:
            return b''
        first, last = self.position//self.block_size, (end-1)//self.block_size
        chunks = list(self.pool.map(self._part, range(first, last+1)))
        data = b''.join(chunks)[self.position-first*self.block_size:end-first*self.block_size]
        self.position = end
        return data

    def close(self):
        if hasattr(self, 'pool'):
            self.pool.shutdown(wait=True)
        super().close()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k,v in value.items() if v is not None}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def is_reference(row, systems, thermo_ids):
    elements = frozenset(row['chemsys'].split('-'))
    if row.get('deprecated') or not any(elements <= s for s in systems):
        return False
    hull = row.get('energy_above_hull')
    if hull is None:
        return len(elements) == 1 and row['nsites'] <= 10
    return row['material_id'] in thermo_ids and math.isfinite(hull) and hull <= .3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=Path('results/smoacs/postprocess-206/inventory-249/inventory.json'))
    parser.add_argument('--output', type=Path, default=Path('results/smoacs/mp-public-reference'))
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Submit through Slurm.')
    proxy = os.environ['SMOACS_REFERENCE_PROXY']
    inventory = json.loads(args.inventory.read_text())
    systems = [frozenset(c['elements']) for c in inventory['candidates']]
    args.output.mkdir(parents=True, exist_ok=True)
    selected_dir = args.output/('selected-'+os.environ['SLURM_JOB_ID'])
    selected_dir.mkdir(exist_ok=False)
    thermo_ids = set()
    for key in THERMO:
        with CachedHTTP(key, args.output/'range-cache', proxy) as source:
            table = pq.read_table(source, columns=['material_id'], pre_buffer=False, use_threads=False)
            thermo_ids.update(table.column('material_id').to_pylist())
            print('THERMO_IDS', key, len(thermo_ids), flush=True)
    seen, references, terminals = set(), [], set()
    counts = dict(rows=0, nondeprecated=0, references=0)
    columns = ['material_id', 'chemsys', 'nsites', 'deprecated', 'energy_above_hull', 'structure']
    with CachedHTTP(SUMMARY, args.output/'range-cache', proxy) as source:
        parquet = pq.ParquetFile(source, pre_buffer=False)
        print('SUMMARY_METADATA_ROWS', parquet.metadata.num_rows, flush=True)
        with gzip.open(selected_dir/'novelty-structures.jsonl.gz', 'wt') as novelty:
            for batch in parquet.iter_batches(batch_size=512, columns=columns, use_threads=False):
                for row in batch.to_pylist():
                    counts['rows'] += 1
                    mid = row['material_id']
                    if mid in seen:
                        raise ValueError('Duplicate MP ID in summary snapshot.')
                    seen.add(mid)
                    if not row['deprecated']:
                        counts['nondeprecated'] += 1
                        novelty.write(json.dumps(clean(row))+'\n')
                    if is_reference(row, systems, thermo_ids):
                        row = clean(row)
                        references.append(row)
                        if '-' not in row['chemsys']:
                            terminals.add(row['chemsys'])
                    if counts['rows'] % 10000 == 0:
                        print('SUMMARY_PROCESSED', counts['rows'], 'REFERENCES', len(references), flush=True)
        if counts['rows'] != parquet.metadata.num_rows:
            raise ValueError('Incomplete summary scan.')
    required = set(e for c in inventory['candidates'] for e in c['elements'])
    counts['references'] = len(references)
    counts['missing_elemental_terminals'] = sorted(required-terminals)
    with gzip.open(selected_dir/'reference-structures.jsonl.gz', 'wt') as stream:
        for row in references:
            stream.write(json.dumps(row)+'\n')
    manifest = dict(snapshot='2026-04-13', summary_key=SUMMARY, thermo_keys=THERMO,
                    counts=counts, thermo_ids=len(thermo_ids), denominator=128,
                    inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                    entry_id_mapping='parent thermo IDs; nested entry IDs not yet audited',
                    stability='unknown_until_reference_relaxation_and_coverage_validation')
    (selected_dir/'summary.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
