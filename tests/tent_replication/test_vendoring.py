"""Verify vendored files differ from upstream only by the 3-line header."""
from __future__ import annotations

import re
import ssl
from pathlib import Path
from urllib.request import urlopen

import pytest

UPSTREAM_FILE = Path("experiments/tent_replication/UPSTREAM.md")
VENDOR_DIR = Path("experiments/tent_replication")


def _read_upstream_sha() -> str:
    text = UPSTREAM_FILE.read_text()
    m = re.search(r"\*\*Vendored commit:\*\*\s*([0-9a-f]{40})", text)
    assert m, "UPSTREAM.md missing 'Vendored commit:' 40-char SHA"
    return m.group(1)


_SYSTEM_CA_BUNDLE = "/etc/pki/tls/cert.pem"


def _fetch(url: str) -> bytes:
    try:
        import os
        cafile = _SYSTEM_CA_BUNDLE if os.path.exists(_SYSTEM_CA_BUNDLE) else None
        ctx = ssl.create_default_context(cafile=cafile)
        return urlopen(url, timeout=15, context=ctx).read()
    except Exception as exc:
        pytest.skip(f"network unavailable: {exc}")


HEADER_LINES = 4  # 3 comment lines + 1 blank


@pytest.mark.parametrize(
    "vendored, upstream_path",
    [
        ("tent_official.py", "tent.py"),
        ("norm_official.py", "norm.py"),
        ("conf.py", "conf.py"),
    ],
)
def test_python_file_matches_upstream_after_header(vendored, upstream_path):
    sha = _read_upstream_sha()
    url = f"https://raw.githubusercontent.com/DequanWang/tent/{sha}/{upstream_path}"
    upstream = _fetch(url).decode()
    local = (VENDOR_DIR / vendored).read_text()
    local_body = "".join(local.splitlines(keepends=True)[HEADER_LINES:])
    assert local_body == upstream, (
        f"{vendored}: body differs from upstream {upstream_path} at SHA {sha[:7]}"
    )


@pytest.mark.parametrize(
    "name", ["source.yaml", "norm.yaml", "tent.yaml"]
)
def test_yaml_byte_identical(name):
    sha = _read_upstream_sha()
    url = f"https://raw.githubusercontent.com/DequanWang/tent/{sha}/cfgs/{name}"
    upstream = _fetch(url)
    local = (VENDOR_DIR / "cfgs" / name).read_bytes()
    assert local == upstream, f"cfgs/{name}: not byte-identical to upstream"
