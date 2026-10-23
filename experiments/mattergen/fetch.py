"""Resume verified byte ranges from immutable official asset URLs."""
import hashlib
import json
from pathlib import Path
import re
import zipfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def download(session,url,target,expected_sha=None,expected_size=None,chunk_size=1024*1024):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if expected_sha and digest(target)!=expected_sha:raise ValueError('Existing asset hash mismatch.')
        if expected_size and target.stat().st_size!=expected_size:raise ValueError('Existing asset size mismatch.')
        if target.suffix=='.whl':
            with zipfile.ZipFile(target) as archive:
                if archive.testzip() is not None:raise ValueError('Existing wheel CRC failed.')
        return digest(target)
    with session.get(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},timeout=(30,60)) as response:
        response.raise_for_status();match=re.fullmatch(r'bytes 0-0/(\d+)',response.headers.get('Content-Range',''))
        if response.status_code!=206 or not match or len(response.content)!=1:raise ValueError('Asset server did not honor byte range.')
        size=int(match[1]);etag=response.headers.get('ETag')
    if expected_size and size!=expected_size:raise ValueError('Official asset size changed.')
    part=target.with_name(target.name+'.part');journal=target.with_name(target.name+'.transfer.json')
    identity=dict(url=url,size=size,etag=etag)
    if journal.exists() and json.loads(journal.read_text())!=identity:raise ValueError('Asset identity changed during transfer.')
    if part.exists() and not journal.exists():raise ValueError('Partial asset lacks identity journal.')
    journal.write_text(json.dumps(identity,indent=2)+'\n');offset=part.stat().st_size if part.exists() else 0
    if offset>size:raise ValueError('Partial asset exceeds total size.')
    while offset<size:
        end=min(size,offset+chunk_size)-1
        with session.get(url,headers={'Range':f'bytes={offset}-{end}','Accept-Encoding':'identity'},timeout=(30,90)) as response:
            response.raise_for_status();body=response.content
            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{size}' or len(body)!=end-offset+1:
                raise ValueError('Truncated or incorrect asset range; verified prefix preserved.')
            if response.headers.get('ETag')!=etag:raise ValueError('Asset ETag changed.')
        with part.open('ab') as stream:stream.write(body)
        offset=end+1
        if offset% (32*1024*1024)==0 or offset==size:print(json.dumps(dict(asset=target.name,received=offset,total=size)),flush=True)
    sha=digest(part)
    if expected_sha and sha!=expected_sha:raise ValueError('Completed asset SHA mismatch.')
    if target.suffix=='.whl':
        with zipfile.ZipFile(part) as archive:
            if archive.testzip() is not None:raise ValueError('Downloaded wheel CRC failed.')
    part.replace(target);return sha
