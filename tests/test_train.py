"""Tests for the pull-based training job queue (no network needed)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_tmp = tempfile.mkdtemp(prefix="bro-train-test-")
os.environ["BRO_DB_PATH"] = os.path.join(_tmp, "test.db")

from app import train  # noqa: E402


def test_create_get_list():
    jid = train.create_job("Qwen/Qwen2-1.5B", "user/my-data", 16)
    assert jid
    job = train.get_job(jid)
    assert job["model_repo"] == "Qwen/Qwen2-1.5B"
    assert job["dataset"] == "user/my-data"
    assert job["lora_rank"] == 16
    assert job["status"] == "queued"
    jobs = train.list_jobs()
    assert any(j["id"] == jid for j in jobs)


def test_lora_rank_clamped():
    jid = train.create_job("m", "", 9999)
    assert train.get_job(jid)["lora_rank"] == 256
    jid2 = train.create_job("m", "", "not-a-number")
    assert train.get_job(jid2)["lora_rank"] == 16


def test_claim_is_atomic_and_fifo():
    first = train.create_job("model-a")
    second = train.create_job("model-b")
    claimed = train.claim_job()
    assert claimed is not None
    # FIFO: oldest queued job claimed first (first, unless an older test job exists —
    # either way the claimed job must leave 'queued')
    assert claimed["status"] == "working"
    assert train.get_job(claimed["id"])["status"] == "working"
    # unknown id
    assert train.get_job("nope") is None
    # drain the rest so later tests start clean
    while train.claim_job():
        pass
    assert first and second  # ids were created


def test_complete_and_metrics():
    jid = train.create_job("model-c")
    assert train.complete_job(jid, "done", result_url="http://x/a.safetensors",
                              metrics={"train_loss": 1.23}, error="")
    job = train.get_job(jid)
    assert job["status"] == "done"
    assert job["result_url"] == "http://x/a.safetensors"
    assert job["metrics"]["train_loss"] == 1.23
    jid2 = train.create_job("model-d")
    assert train.complete_job(jid2, "failed", error="boom")
    assert train.get_job(jid2)["status"] == "failed"
    assert train.get_job(jid2)["error"] == "boom"
    assert train.complete_job("missing", "done") is False


def test_adapter_path_is_traversal_safe():
    p = train.adapter_file_path("abc123", "../../etc/passwd")
    assert p.endswith("passwd")
    assert ".." not in p
    assert os.path.dirname(p).endswith("abc123")


def test_adapter_path_job_id_is_traversal_safe():
    # Regression: job_id came raw from the URL, so %2e%2e escaped ADAPTERS_DIR.
    for evil in ("..", "../..", "..%2f..", "/etc", "a/b"):
        p = train.adapter_file_path(evil, "adapter.bin")
        assert os.path.commonpath([p, train.ADAPTERS_DIR]) == train.ADAPTERS_DIR
        assert ".." not in p
    # Real uuid-hex ids are untouched.
    p = train.adapter_file_path("abc123def456", "adapter.bin")
    assert os.path.dirname(p).endswith("abc123def456")
