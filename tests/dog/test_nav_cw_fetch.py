"""tools/deploy/nav_cw_fetch.py: download a list of files (apt --print-uris format or plain), check hashes, against a
local HTTP server."""
import hashlib
import http.server
import importlib.util
import os
import threading

import pytest

from .conftest import ROOT

spec = importlib.util.spec_from_file_location("nav_cw_fetch", os.path.join(ROOT, "tools", "deploy", "nav_cw_fetch.py"))
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    (root / "pool").mkdir(parents=True)
    files = {"pool/libfoo-dev_1.0_arm64.deb": b"foo" * 1000, "pool/v0.5.1.tar.gz": b"acados"}
    for name, data in files.items():
        (root / name).write_bytes(data)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d/" % httpd.server_address[1]
    old = {k: os.environ.pop(k, None) for k in ("http_proxy", "HTTP_PROXY")}   # the local server, not a proxy
    os.environ["no_proxy"] = "127.0.0.1"
    yield base, {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    httpd.shutdown()
    for k, v in old.items():
        if v is not None:
            os.environ[k] = v


def test_downloads_checks_and_writes_sums(server, tmp_path, capsys):
    base, sha = server
    lst = tmp_path / "list.txt"
    lst.write_text("# from apt-get install --print-uris on the dog\n"
                   "'%spool/libfoo-dev_1.0_arm64.deb' libfoo-dev_1.0_arm64.deb 3000 SHA256:%s\n"
                   "%spool/v0.5.1.tar.gz sha256=%s name=acados-v0.5.1.tar.gz\n" % (
                       base, sha["pool/libfoo-dev_1.0_arm64.deb"], base, sha["pool/v0.5.1.tar.gz"]))
    out = tmp_path / "out"
    assert fetch.main([str(lst), str(out)]) == 0
    assert (out / "acados-v0.5.1.tar.gz").read_bytes() == b"acados"
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert "%s  libfoo-dev_1.0_arm64.deb" % sha["pool/libfoo-dev_1.0_arm64.deb"] in sums
    assert "2 of 2 files" in capsys.readouterr().out
    assert fetch.main([str(lst), str(out)]) == 0               # again: nothing downloaded
    assert capsys.readouterr().out.count("have") == 2


def test_a_wrong_hash_or_a_missing_file_fails(server, tmp_path, capsys):
    base, _ = server
    lst = tmp_path / "list.txt"
    lst.write_text("%spool/v0.5.1.tar.gz sha256=%s\n%spool/nothere.deb\n" % (base, "0" * 64, base))
    out = tmp_path / "out"
    assert fetch.main([str(lst), str(out), "--tries", "1"]) == 1
    text = capsys.readouterr().out
    assert "BAD HASH v0.5.1.tar.gz" in text and "FAILED nothere.deb" in text
    assert not (out / "v0.5.1.tar.gz").exists()


def test_rewrite_to_a_mirror(server, tmp_path):
    base, sha = server
    lst = tmp_path / "list.txt"
    lst.write_text("'http://ports.ubuntu.com/ubuntu-ports/pool/libfoo-dev_1.0_arm64.deb' libfoo-dev_1.0_arm64.deb "
                   "3000 SHA256:%s\n" % sha["pool/libfoo-dev_1.0_arm64.deb"])
    assert fetch.main([str(lst), str(tmp_path / "out"), "--rewrite",
                       "http://ports.ubuntu.com/ubuntu-ports/=" + base]) == 0


def test_the_same_name_twice_is_refused():
    with pytest.raises(SystemExit):
        fetch.parse(["http://a/x/v1.tar.gz", "http://b/y/v1.tar.gz"])
