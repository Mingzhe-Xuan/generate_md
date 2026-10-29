"""Re-select all needed MP competitors from verified local immutable range cache."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


class OfflineRanges(io.RawIOBase):
    block_size=1024*1024

    def __init__(self,key,cache):
        super().__init__()
        self.folder=Path(cache)/hashlib.sha256(key.encode()).hexdigest()
        self.meta=json.loads((self.folder/'object.json').read_text())
        if self.meta['key']!=key:raise ValueError('Wrong cached object.')
        self.size=self.meta['size'];self.position=0;self.verified=set()

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if target<0:raise ValueError('Negative seek.')
        self.position=target;return target

    def read(self,size=-1):
        if size is None or size<0:size=self.size-self.position
        end=min(self.position+size,self.size)
        if end<=self.position:return b''
        start=self.position;chunks=[]
        for index in range(start//self.block_size,(end-1)//self.block_size+1):
            path=self.folder/f'{index:06d}.part'
            data=path.read_bytes()
            if len(data)!=min(self.block_size,self.size-index*self.block_size):raise ValueError('Truncated cached part.')
            if index not in self.verified:
                if hashlib.sha256(data).hexdigest()!=path.with_suffix('.sha256').read_text():raise ValueError('Cached part checksum mismatch.')
                self.verified.add(index)
            chunks.append(data)
        self.position=end
        return b''.join(chunks)[start%self.block_size:start%self.block_size+end-start]


def main(args):
    import pyarrow.parquet as pq
    sys.path.insert(0,str(HERE.parents[1]/'smoacs/references'))
    import prepare as official
    root=args.run.resolve();source=root/'combined/inventory-249/inventory.json'
    execution=dict(job_id=os.environ.get('SLURM_JOB_ID'),retry_of=os.environ.get('ORTHOGONAL_REF_RETRY_OF'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.executable,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_prepare_sha256=hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest())
    (root/('reference-selection-execution-'+str(execution['job_id'])+'.json')).write_text(json.dumps(execution,indent=2)+'\n')
    inventory=json.loads(source.read_text())
    eligible=[c for c in inventory['candidates'] if json.loads((root/'combined/relaxation'/Path(c['candidate_id']).stem/'result.json').read_text())['status']=='converged']
    systems=[frozenset(c['elements']) for c in eligible]
    selection=root/'references';selection.mkdir(exist_ok=False)
    novelty=Path('results/mattergen/gap2p5-task128-20261024/full128/references/selected-322/novelty-structures.jsonl.gz').resolve()
    if not systems:
        with gzip.open(selection/'reference-structures.jsonl.gz','wt') as f:pass
        (selection/'novelty-structures.jsonl.gz').symlink_to(novelty)
        manifest=dict(snapshot='2026-04-13',inventory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            reference_filter='published Zenodo20149017 fetch_mp_structures: public hull <=0.3, excludes missing hull',
            eligible_proposal_ids=[],counts=dict(rows=0,references=0,missing_elemental_terminals=[]),
            reason='All proposals failed B; no hull queries needed. Retain all proposal failures and denominators.')
        (selection/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return
    cache=Path('results/smoacs/mp-public-reference/range-cache')
    thermo_ids=set();objects=[]
    for key in official.THERMO:
        with OfflineRanges(key,cache) as reader:
            parquet=pq.ParquetFile(reader,pre_buffer=False)
            paths=[parquet.metadata.schema.column(i).path for i in range(parquet.metadata.num_columns)]
            columns=['material_id']+[p for p in paths if p.startswith('entries.') and (p.endswith('.key') or p.endswith('.data.material_id') or '.entry_id.' in p)]
            ids,_=official.entry_material_ids(parquet.read(columns=columns,use_threads=False));thermo_ids.update(ids)
            objects.append(dict(**reader.meta,verified_parts=len(reader.verified)))
    count=0;selected=[];terminals=set()
    with OfflineRanges(official.SUMMARY,cache) as reader:
        parquet=pq.ParquetFile(reader,pre_buffer=False)
        for batch in parquet.iter_batches(batch_size=512,columns=['material_id','chemsys','nsites','deprecated','energy_above_hull','structure'],use_threads=False):
            for row in batch.to_pylist():
                count+=1
                if official.is_reference(row,systems,thermo_ids):
                    row=official.clean(row);selected.append(row)
                    if '-' not in row['chemsys']:terminals.add(row['chemsys'])
            if count%10240==0:print(json.dumps(dict(scanned=count,selected=len(selected))),flush=True)
        if count!=parquet.metadata.num_rows:raise ValueError('Incomplete full MP scan.')
        objects.append(dict(**reader.meta,verified_parts=len(reader.verified)))
    with gzip.open(selection/'reference-structures.jsonl.gz','wt') as f:
        for row in selected:f.write(json.dumps(row)+'\n')
    (selection/'novelty-structures.jsonl.gz').symlink_to(novelty)
    manifest=dict(snapshot='2026-04-13',inventory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_filter='published Zenodo20149017 fetch_mp_structures: public hull <=0.3, excludes missing hull',
        reference_execution='Offline full-column scan, same reference selector, no competitor-pruned old scope.',
        cache_objects=objects,eligible_proposal_ids=[c['candidate_id'] for c in eligible],
        counts=dict(rows=count,references=len(selected),missing_elemental_terminals=sorted(set().union(*systems)-terminals)),
        novelty_source=str(novelty),novelty_source_sha256=hashlib.sha256(novelty.read_bytes()).hexdigest())
    (selection/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(references=len(selected),scope_systems=len(systems),scanned=count)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm.')
    main(a)
