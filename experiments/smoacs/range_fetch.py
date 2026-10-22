"""Download verified bounded ranges through the explicitly supplied SSH SOCKS proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote

import requests


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_file(session, base: str, item: dict, destination: Path,
                  chunk_size: int = 262144, retries: int = 3) -> dict:
    name, size = item["name"], item["size"]
    if not re.fullmatch(r"[A-Za-z0-9_.+\-]+\.whl", name) or ".." in name:
        raise ValueError("Invalid wheel filename.")
    if not isinstance(size, int) or size <= 0 or not 0 < chunk_size <= 262144 or retries < 1:
        raise ValueError("Invalid transfer size or retry count.")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    part = destination / (name + ".part")
    journal = destination / (name + ".range.json")
    state = {"name": name, "size": size, "offset": 0, "complete": False}
    if journal.exists():
        state = json.loads(journal.read_text(encoding="utf-8"))
        if state["name"] != name or state["size"] != size or not 0 <= state["offset"] <= size:
            raise ValueError("Checkpoint does not match the source manifest.")
    if target.exists():
        if target.stat().st_size == size and (
            state["complete"] or (state["offset"] == size and not part.exists())
        ):
            state["complete"] = True
            save_state(journal, state)
            return state
        raise FileExistsError("Refusing to overwrite an unverified existing wheel.")
    offset = state["offset"]
    if part.exists():
        if part.stat().st_size < offset:
            raise ValueError("Partial file is shorter than its verified checkpoint.")
        with part.open("r+b") as stream:
            stream.truncate(offset)
    elif offset:
        raise ValueError("Checkpoint exists without its partial file.")
    else:
        part.touch()
    save_state(journal, state)
    while offset < size:
        end = min(offset + chunk_size, size) - 1
        for attempt in range(retries):
            try:
                with session.get(
                    base.rstrip("/") + "/" + quote(name),
                    headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                    timeout=(10, 30), stream=True,
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError("Server did not return a byte range.")
                    if response.headers.get("Content-Range") != f"bytes {offset}-{end}/{size}":
                        raise ValueError("Content-Range mismatch.")
                    if int(response.headers.get("Content-Length", "-1")) != end - offset + 1:
                        raise ValueError("Content-Length mismatch.")
                    data = response.content
                    if len(data) != end - offset + 1:
                        raise ValueError("Truncated range.")
                    if hashlib.sha256(data).hexdigest() != response.headers.get("X-Chunk-SHA256"):
                        raise ValueError("Chunk SHA-256 mismatch.")
                break
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt + 1 == retries:
                    raise
                time.sleep(1)
        with part.open("ab") as stream:
            stream.write(data)
        offset = end + 1
        state["offset"] = offset
        save_state(journal, state)
    part.replace(target)
    state["complete"] = True
    save_state(journal, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    if not args.proxy.startswith("socks5h://"):
        parser.error("An explicit socks5h proxy is required.")
    destination = args.destination.resolve()
    allowed = (Path.cwd() / "results/smoacs").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed:
        parser.error("destination must be a child under results/smoacs.")
    with requests.Session() as session:
        session.trust_env = False
        session.proxies.update({"http": args.proxy, "https": args.proxy})
        response = session.get(args.base_url.rstrip("/") + "/_manifest", timeout=(10, 30))
        response.raise_for_status()
        manifest = response.json()
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("Source manifest must be a nonempty list.")
        requested = set(args.only or [])
        selected = [item for item in manifest if not requested or item["name"] in requested]
        if requested - {item["name"] for item in selected}:
            parser.error("Some requested wheels are absent from the source manifest.")
        completed = []
        for item in selected:
            print("START " + item["name"], flush=True)
            completed.append(download_file(session, args.base_url, item, destination))
            print("DONE " + item["name"], flush=True)
    report = {"scope": "subset" if requested else "all",
              "files": len(completed), "completed": completed,
              "verification": "Every received range matched its source SHA-256."}
    save_state(destination / "download_status.json", report)
    print(json.dumps({"scope": report["scope"], "files": report["files"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
