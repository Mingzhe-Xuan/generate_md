"""Byte integrity and archive safety for author-released benchmark datasets."""
import hashlib
from pathlib import Path, PurePosixPath
import tarfile


def hashes(path):
    path = Path(path)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    git = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            for result in (sha256, md5, git):
                result.update(block)
    return dict(size=path.stat().st_size, sha256=sha256.hexdigest(), md5=md5.hexdigest(), git_sha1=git.hexdigest())


def verify_asset(path, spec):
    actual = hashes(path)
    if actual['size'] != spec['size'] or actual[spec['algorithm']] != spec['checksum']:
        raise ValueError('Author asset checksum/length mismatch: ' + str(path))
    return actual


def extract_checked(archive, destination):
    destination = Path(destination).resolve()
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            target = (destination / member.name).resolve()
            if name.is_absolute() or '..' in name.parts or '\\' in member.name or ':' in member.name:
                raise ValueError('Unsafe archive path: ' + member.name)
            if not target.is_relative_to(destination) or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe archive member: ' + member.name)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            payload = handle.extractfile(member).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError('Refuse to overwrite different reference data: ' + member.name)
            else:
                target.write_bytes(payload)
