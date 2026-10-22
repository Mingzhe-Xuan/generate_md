"""Tiny loopback-only transport tests; no model computations."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

spec = importlib.util.spec_from_file_location("range_server", Path(__file__).with_name("range_server.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("header", ["", "bytes=4-3", "bytes=0-8", "bytes=0-5", "bytes=-4"])
def test_rejects_bad_or_oversized_range(header):
    with pytest.raises(ValueError):
        module.parse_range(header, 8, 4)


def test_accepts_last_short_chunk():
    assert module.parse_range("bytes=6-7", 8, 4) == (6, 7)


def test_http_range_hash_and_allowlist(tmp_path):
    (tmp_path / "sample.whl").write_bytes(b"abcdefgh")
    (tmp_path / "private.txt").write_text("never serve")
    server = module.build_server(tmp_path, limit=4)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = f"http://127.0.0.1:{server.server_port}/"
    try:
        with urlopen(Request(root + "sample.whl", headers={"Range": "bytes=2-5"})) as response:
            assert response.status == 206
            assert response.read() == b"cdef"
            assert response.headers["Content-Range"] == "bytes 2-5/8"
            assert response.headers["X-Chunk-SHA256"] == hashlib.sha256(b"cdef").hexdigest()
        with urlopen(root + "_manifest") as response:
            assert json.load(response) == [{"name": "sample.whl", "size": 8}]
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        try:
            connection.request("GET", "/sample.whl", headers={"Range": "bytes=0-3"})
            assert connection.getresponse().read() == b"abcd"
            first_socket = connection.sock
            assert first_socket is not None
            connection.request("GET", "/sample.whl", headers={"Range": "bytes=4-7"})
            assert connection.getresponse().read() == b"efgh"
            assert connection.sock is first_socket
        finally:
            connection.close()
        for name, code in [("private.txt", 404), ("../private.txt", 404), ("sample.whl", 416)]:
            with pytest.raises(HTTPError) as failure:
                urlopen(root + name)
            assert failure.value.code == code
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
