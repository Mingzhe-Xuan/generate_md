"""Fetch pinned CUDA wheels with task-local DNS resolution and normal TLS checks."""
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import zipfile

from fetch import digest


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for the cache preparation.')
    root=Path('results/mattergen');sources=json.loads(Path(__file__).with_name('sources.json').read_text())
    hosts=json.loads((root/'network-hosts.json').read_text());addresses=hosts['data.pyg.org']
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:raise ValueError('Require a public address for the official CDN.')
    environment={k:v for k,v in os.environ.items() if k.lower() not in ('all_proxy','http_proxy','https_proxy')}
    output=root/'wheels';output.mkdir(exist_ok=True)
    for name,sha in sources['wheel_sha256'].items():
        target=output/name
        if target.exists() and digest(target)==sha:continue
        temporary=target.with_name(target.name+'.direct')
        url='https://data.pyg.org/whl/torch-2.7.0%2Bcu128/'+name.replace('+','%2B')
        for address in addresses:
            command=['curl','--fail','--location','--resolve','data.pyg.org:443:'+address,
                '--connect-timeout','10','--max-time','120','--output',str(temporary),url]
            if subprocess.run(command,env=environment).returncode==0:break
        else:raise RuntimeError('No validated CDN address completed the download.')
        if digest(temporary)!=sha:raise ValueError('Official wheel SHA mismatch.')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('Official wheel CRC failed.')
        temporary.replace(target);print(json.dumps(dict(wheel=name,sha256=sha,bytes=target.stat().st_size)),flush=True)


if __name__=='__main__':main()
