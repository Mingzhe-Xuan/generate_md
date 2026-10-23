"""Slurm-only isolated inference environment and verified official MatterGen assets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


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
    subprocess.run([str(base),'-m','pip','--python',str(python),'install','-r',str(module/'requirements.txt'),
        '--find-links','https://data.pyg.org/whl/torch-2.7.0+cu128.html','--only-binary','torch-scatter,torch-sparse,torch-cluster'],check=True)
    checkpoint=assets/'checkpoints/dft_band_gap/checkpoints/last.ckpt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists() or digest(checkpoint)!=spec['checkpoint_sha256']:
        import requests
        temporary=checkpoint.with_suffix('.download')
        with requests.get(spec['checkpoint_url'],stream=True,timeout=(30,120)) as response:
            response.raise_for_status()
            with temporary.open('wb') as out:
                for part in response.iter_content(1024*1024):out.write(part)
        if temporary.stat().st_size!=spec['checkpoint_size_bytes'] or digest(temporary)!=spec['checkpoint_sha256']:
            raise ValueError('Downloaded checkpoint hash/size mismatch.')
        temporary.replace(checkpoint)
    (checkpoint.parent.parent/'config.yaml').write_bytes(config.read_bytes())
    freeze=subprocess.check_output([str(base),'-m','pip','--python',str(python),'freeze'],text=True)
    (assets/'environment-freeze.txt').write_text(freeze)
    subprocess.run([str(base),'-m','pip','--python',str(python),'check'],check=True)
    (assets/'setup-complete.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
        source=spec,environment=str(environment),base_readonly=str(base),upstream_torch_pin_override='2.7.1+cu128 for Blackwell',
        package_scope='Inference dependencies; official source imported unmodified, no training extras.'),indent=2)+'\n')


if __name__=='__main__':main()
