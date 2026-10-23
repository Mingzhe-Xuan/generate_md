"""Slurm-only isolated inference environment and verified official MatterGen assets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

from fetch import download


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for setup and native imports.')
    root=Path.cwd();module=Path(__file__).resolve().parent;assets=root/'results/mattergen'
    spec=json.loads((module/'sources.json').read_text());archive=assets/'mattergen-source.tar.gz'
    if digest(archive)!=spec['source_archive_sha256']:raise ValueError('Official source archive hash mismatch.')
    source=assets/'source'
    if not source.exists():
        source.mkdir()
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                target=(source/member.name).resolve()
                if not target.is_relative_to(source.resolve()) or member.issym() or member.islnk() or member.isdev():
                    raise ValueError('Unexpected archive member.')
            handle.extractall(source)
    config=source/'checkpoints/dft_band_gap/config.yaml'
    if digest(config)!=spec['config_sha256']:raise ValueError('Official config changed.')
    base=root/'.venv-smoacs-blackwell/bin/python';environment=root/'.venv-mattergen';python=environment/'bin/python'
    if not python.exists():subprocess.run([str(base),'-m','venv','--without-pip',str(environment)],check=True)
    site_path=environment/'lib/python3.10/site-packages'
    if not site_path.exists():raise ValueError('Expected task Python3.10 environment.')
    (site_path/'blackwell_readonly.pth').write_text(str(root/'.venv-smoacs-blackwell/lib/python3.10/site-packages')+'\n')
    import requests
    wheels=assets/'wheels';wheel_records=[]
    with requests.Session() as session:
        for name,version in [('torch_scatter','2.1.2'),('torch_sparse','0.6.18'),('torch_cluster','1.6.3')]:
            filename=f'{name}-{version}+pt27cu128-cp310-cp310-linux_x86_64.whl'
            url='https://data.pyg.org/whl/torch-2.7.0%2Bcu128/'+filename.replace('+','%2B')
            sha=download(session,url,wheels/filename,spec['wheel_sha256'][filename])
            wheel_records.append(dict(filename=filename,url=url,sha256=sha))
    (assets/'wheel-manifest.json').write_text(json.dumps(wheel_records,indent=2)+'\n')
    subprocess.run([str(base),'-m','pip','--python',str(python),'install','-r',str(module/'requirements.txt'),
        '--find-links',str(wheels),'--only-binary','torch-scatter,torch-sparse,torch-cluster'],check=True)
    checkpoint=assets/'checkpoints/dft_band_gap/checkpoints/last.ckpt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists() or digest(checkpoint)!=spec['checkpoint_sha256']:
        with requests.Session() as session:download(session,spec['checkpoint_url'],checkpoint,spec['checkpoint_sha256'],spec['checkpoint_size_bytes'])
    (checkpoint.parent.parent/'config.yaml').write_bytes(config.read_bytes())
    freeze=subprocess.check_output([str(base),'-m','pip','--python',str(python),'freeze'],text=True)
    (assets/'environment-freeze.txt').write_text(freeze)
    subprocess.run([str(base),'-m','pip','--python',str(python),'check'],check=True)
    (assets/'setup-complete.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
        source=spec,environment=str(environment),base_readonly=str(base),upstream_torch_pin_override='2.7.1+cu128 for Blackwell',
        network_route=os.environ.get('MATTERGEN_NETWORK_ROUTE'),
        package_scope='Inference dependencies; official source imported unmodified, no training extras.'),indent=2)+'\n')


if __name__=='__main__':main()
