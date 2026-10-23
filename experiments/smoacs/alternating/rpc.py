"""Dependency-free framing for persistent worker responses."""
import json
from pathlib import Path


def read_response(stream,diagnostic):
    for line in stream:
        prefix,found,body=line.partition('__AB_RPC__')
        if prefix:
            with Path(diagnostic).open('a') as log:log.write(prefix)
        if found:return json.loads(body)
    raise RuntimeError('B worker exited without a protocol response; inspect its stdout/stderr logs.')
