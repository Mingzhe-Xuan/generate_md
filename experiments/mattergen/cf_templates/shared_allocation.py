"""Retain a parent allocation until its explicitly registered additional steps finish."""
import json
from pathlib import Path
import time


def wait_for_children(root, timeout=21600, interval=5):
    root = Path(root)
    started = time.monotonic()
    while True:
        records = [json.loads(p.read_text()) for p in sorted((root/'shared-children').glob('*.json'))]
        if all(r.get('completed') is True for r in records):
            return records
        if time.monotonic()-started >= timeout:
            raise TimeoutError('Registered additional GPU step did not finish within its allocation wait budget.')
        time.sleep(interval)
