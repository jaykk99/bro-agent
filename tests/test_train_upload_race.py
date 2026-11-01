"""Train adapter uploads: concurrent uploads to one job must not beat the
per-job cap.

Proven gap (2026-11-01, before the fix): the per-job cap from 2026-10-30
was checked against a total snapshotted before streaming, and nothing
serialized uploads to the same job. Two 20 MB uploads fired concurrently
at a 30 MB job cap both computed their base total before either one
landed, both fit under the cap on their own numbers, and both returned
200 — leaving 40 MB stored for one job, 10 MB over the cap. Sequentially
the second upload 413s, so this was purely a check-then-stream race, not
a misconfigured cap.

The fix serializes uploads per job: one job's cap ledger is only ever
read, streamed against, and replaced under that job's upload lock, so
the second upload sees the first one's landed bytes and 413s.
"""

import asyncio
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-race-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "race.db")
os.environ.pop("BRO_ACCESS_CODE", None)  # gate off for these tests

from httpx import AsyncClient, ASGITransport  # noqa: E402

from app import train  # noqa: E402
from app.main import app  # noqa: E402

MB = 1024 * 1024


def test_concurrent_uploads_cannot_exceed_job_cap(monkeypatch):
    # 30 MB job cap, two 20 MB files: whichever lands first fits, the
    # other must 413. Files exceed Starlette's 1 MB spool threshold so
    # each chunk read yields the loop and the two uploads genuinely
    # overlap, the way two workers uploading at once would.
    monkeypatch.setenv("BRO_ADAPTER_JOB_MAX_BYTES", str(30 * MB))
    jid = train.create_job("model-race")
    body = b"x" * (20 * MB)

    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            return await asyncio.gather(
                c.post(
                    f"/api/train/{jid}/upload",
                    files={"file": ("a.bin", io.BytesIO(body), "application/octet-stream")},
                    data={"access_code": ""},
                ),
                c.post(
                    f"/api/train/{jid}/upload",
                    files={"file": ("b.bin", io.BytesIO(body), "application/octet-stream")},
                    data={"access_code": ""},
                ),
            )

    r1, r2 = asyncio.run(_run())
    assert sorted([r1.status_code, r2.status_code]) == [200, 413]
    assert train.adapter_total_bytes(jid) == 20 * MB
    leftovers = [n for n in os.listdir(train.adapter_job_dir(jid))
                 if n.startswith(".upload-")]
    assert leftovers == []
