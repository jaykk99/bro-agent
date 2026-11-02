"""Train adapter uploads: aggregate cap across all jobs.

Proven gap (2026-11-02, before the fix): the per-file (2 GB) and
per-job (4 GB) caps never bounded the adapters tree as a whole. Every
job could hold its own 4 GB, so uploads spread across many jobs filled
the disk job by job with no cap ever tripping. There was no function
computing a cross-job total at all.

The fix adds a total cap (default 20 GB, BRO_ADAPTER_TOTAL_MAX_BYTES,
read per request like the per-job cap), enforced two ways so concurrent
uploads to different jobs cannot beat it either: each upload reserves
its streamed bytes as it goes (a concurrent upload's not-yet-landed
bytes already count), and the final replace re-scans committed bytes
under a cross-job commit lock (an upload that landed and released
mid-stream cannot slip past a stale snapshot).
"""

import asyncio
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-total-cap-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "totalcap.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from httpx import AsyncClient, ASGITransport  # noqa: E402

from app import train  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

MB = 1024 * 1024


@pytest.fixture(autouse=True)
def _caps(monkeypatch):
    # Per-job cap huge, so only the total cap can trip in these tests.
    # The total cap is set per test, relative to whatever other tests
    # already stored in the shared adapters dir — tests must never
    # assume the tree starts empty.
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", str(100 * MB))


def _upload(job_id, name, content):
    return client.post(
        f"/api/train/{job_id}/upload",
        files={"file": (name, io.BytesIO(content), "application/octet-stream")},
        data={"access_code": ""},
    )


def test_total_cap_across_jobs(monkeypatch):
    base = train.adapters_total_bytes()
    monkeypatch.setenv("BRO_ADAPTER_TOTAL_MAX_BYTES", str(base + 2500))
    jid_a = train.create_job("model-total-a")
    jid_b = train.create_job("model-total-b")
    jid_c = train.create_job("model-total-c")
    assert _upload(jid_a, "a.bin", b"x" * 1000).status_code == 200
    assert _upload(jid_b, "b.bin", b"y" * 1000).status_code == 200
    # base + 2000 stored, third 1000-byte file would take the tree to
    # base + 3000, over the base + 2500 cap.
    r = _upload(jid_c, "c.bin", b"z" * 1000)
    assert r.status_code == 413
    assert not os.path.exists(os.path.join(train.adapter_job_dir(jid_c), "c.bin"))
    assert train.adapters_total_bytes() == base + 2000
    # Reservations never leak: a finished run leaves the ledger empty.
    assert train.inflight_reserved_bytes() == 0


def test_reupload_same_name_does_not_double_count_against_total(monkeypatch):
    base = train.adapters_total_bytes()
    monkeypatch.setenv("BRO_ADAPTER_TOTAL_MAX_BYTES", str(base + 2500))
    jid = train.create_job("model-total-recap")
    assert _upload(jid, "a.bin", b"x" * 1000).status_code == 200
    # Replacing a.bin with a 1500-byte file takes the tree to
    # base + 1500, not base + 2500: the old 1000 bytes are subtracted,
    # not double-counted.
    assert _upload(jid, "a.bin", b"x" * 1500).status_code == 200
    assert train.adapters_total_bytes() == base + 1500
    # ...but a replacement that genuinely exceeds the total still 413s
    # and leaves the good previous adapter untouched.
    r = _upload(jid, "a.bin", b"x" * 3000)
    assert r.status_code == 413
    assert train.adapters_total_bytes() == base + 1500
    with open(os.path.join(train.adapter_job_dir(jid), "a.bin"), "rb") as f:
        assert f.read() == b"x" * 1500


def test_total_bytes_skips_symlinks_and_temp_files(tmp_path):
    base = train.adapters_total_bytes()
    jid = train.create_job("model-total-scan")
    d = train.adapter_job_dir(jid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "real.bin"), "wb") as f:
        f.write(b"r" * 100)
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE")
    os.symlink(str(outside), os.path.join(d, "link.bin"))
    with open(os.path.join(d, ".upload-deadbeef.tmp"), "wb") as f:
        f.write(b"t" * 500)
    # Symlinked job dir: skipped whole, its files never counted.
    linked_dir = os.path.join(train.ADAPTERS_DIR, "linked-job")
    os.symlink(str(d), linked_dir)
    assert train.adapters_total_bytes() == base + 100


def test_concurrent_uploads_to_different_jobs_cannot_exceed_total_cap(monkeypatch):
    # 30 MB total cap, two 20 MB files to DIFFERENT jobs: the per-job
    # lock cannot save this (it is per job by design), so without the
    # reservation ledger both would fit on their own numbers and land
    # 40 MB. Exactly one must 200 and the other 413.
    base = train.adapters_total_bytes()
    monkeypatch.setenv("BRO_ADAPTER_TOTAL_MAX_BYTES", str(base + 30 * MB))
    jid_a = train.create_job("model-total-race-a")
    jid_b = train.create_job("model-total-race-b")
    body = b"x" * (20 * MB)

    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            return await asyncio.gather(
                c.post(
                    f"/api/train/{jid_a}/upload",
                    files={"file": ("a.bin", io.BytesIO(body), "application/octet-stream")},
                    data={"access_code": ""},
                ),
                c.post(
                    f"/api/train/{jid_b}/upload",
                    files={"file": ("b.bin", io.BytesIO(body), "application/octet-stream")},
                    data={"access_code": ""},
                ),
            )

    r1, r2 = asyncio.run(_run())
    assert sorted([r1.status_code, r2.status_code]) == [200, 413]
    assert train.adapters_total_bytes() == base + 20 * MB
    assert train.inflight_reserved_bytes() == 0
    for jid in (jid_a, jid_b):
        leftovers = [n for n in os.listdir(train.adapter_job_dir(jid))
                     if n.startswith(".upload-")]
        assert leftovers == []
