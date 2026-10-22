"""Real loopback tests for interruption recovery and integrity gates."""
import importlib.util
import json
from pathlib import Path
import threading

import pytest
import requests

ROOT = Path(__file__).parent
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

server_module = load("range_server_client_tests", "range_server.py")
client = load("range_fetch_tests", "range_fetch.py")


@pytest.fixture
def endpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.whl").write_bytes(b"abcdefgh")
    server = server_module.build_server(source, limit=4)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    session = requests.Session()
    session.trust_env = False
    yield session, f"http://127.0.0.1:{server.server_port}", tmp_path / "destination"
    session.close()
    server.shutdown()
    server.server_close()
    worker.join()


def test_complete_transfer_and_checkpoint(endpoint):
    session, base, destination = endpoint
    result = client.download_file(session, base, {"name": "sample.whl", "size": 8},
                                  destination, chunk_size=4)
    assert result["complete"] and result["offset"] == 8
    assert (destination / "sample.whl").read_bytes() == b"abcdefgh"
    assert json.loads((destination / "sample.whl.range.json").read_text())["complete"]


def test_network_failure_resumes_only_verified_prefix(endpoint, monkeypatch):
    session, base, destination = endpoint
    original = session.get
    calls = []
    def failing(url, **kwargs):
        calls.append(kwargs["headers"]["Range"])
        if len(calls) > 1:
            raise requests.ConnectionError("injected interruption")
        return original(url, **kwargs)
    monkeypatch.setattr(session, "get", failing)
    item = {"name": "sample.whl", "size": 8}
    with pytest.raises(requests.ConnectionError):
        client.download_file(session, base, item, destination, chunk_size=4, retries=1)
    assert (destination / "sample.whl.part").read_bytes() == b"abcd"
    with (destination / "sample.whl.part").open("ab") as stream:
        stream.write(b"uncommitted tail")
    resumed = []
    def tracking(url, **kwargs):
        resumed.append(kwargs["headers"]["Range"])
        return original(url, **kwargs)
    monkeypatch.setattr(session, "get", tracking)
    client.download_file(session, base, item, destination, chunk_size=4)
    assert resumed == ["bytes=4-7"]
    assert (destination / "sample.whl").read_bytes() == b"abcdefgh"


def test_bad_hash_does_not_commit_data(endpoint, monkeypatch):
    session, base, destination = endpoint
    original = session.get
    def corrupted(url, **kwargs):
        response = original(url, **kwargs)
        response.headers["X-Chunk-SHA256"] = "0" * 64
        return response
    monkeypatch.setattr(session, "get", corrupted)
    with pytest.raises(ValueError, match="SHA-256"):
        client.download_file(session, base, {"name": "sample.whl", "size": 8},
                             destination, chunk_size=4)
    assert not (destination / "sample.whl").exists()
    assert (destination / "sample.whl.part").stat().st_size == 0


def test_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        client.download_file(None, "", {"name": "../bad.whl", "size": 8}, tmp_path)


def test_recovers_crash_after_final_rename(endpoint, monkeypatch):
    session, base, destination = endpoint
    item = {"name": "sample.whl", "size": 8}
    client.download_file(session, base, item, destination, chunk_size=4)
    journal = destination / "sample.whl.range.json"
    state = json.loads(journal.read_text())
    state["complete"] = False
    journal.write_text(json.dumps(state))
    def unexpected_network(*args, **kwargs):
        raise AssertionError("Verified renamed file must not be downloaded again")
    monkeypatch.setattr(session, "get", unexpected_network)
    result = client.download_file(session, base, item, destination, chunk_size=4)
    assert result["complete"]
    assert (destination / "sample.whl").read_bytes() == b"abcdefgh"
