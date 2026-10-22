"""Serve bounded ranges of public wheel files on Windows loopback for SSH SOCKS transfer."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
from urllib.parse import unquote, urlsplit


def parse_range(header: str, size: int, limit: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d+)-(\d+)", header or "")
    if not match:
        raise ValueError("An explicit bounded byte range is required.")
    start, end = map(int, match.groups())
    if not (0 <= start <= end < size) or end - start + 1 > limit:
        raise ValueError("Invalid or oversized range.")
    return start, end


def build_server(directory: Path, limit: int = 262144) -> ThreadingHTTPServer:
    files = {p.name: p for p in directory.glob("*.whl") if p.is_file() and not p.is_symlink()}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            name = unquote(urlsplit(self.path).path.lstrip("/"))
            if name == "_manifest":
                data = json.dumps([{"name": n, "size": p.stat().st_size}
                                   for n, p in sorted(files.items())]).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
                return
            if name not in files:
                self.send_error(404)
                return
            path = files[name]
            size = path.stat().st_size
            try:
                start, end = parse_range(self.headers.get("Range", ""), size, limit)
            except ValueError:
                self.send_error(416)
                return
            with path.open("rb") as stream:
                stream.seek(start)
                data = stream.read(end - start + 1)
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("X-Chunk-SHA256", hashlib.sha256(data).hexdigest())
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory must exist")
    server = build_server(args.directory.resolve())
    print(json.dumps({"host": "127.0.0.1", "port": server.server_port, "pid": os.getpid()}), flush=True)

    def control():
        for line in sys.stdin:
            if line.strip() == "quit":
                server.shutdown()
                break

    threading.Thread(target=control, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
