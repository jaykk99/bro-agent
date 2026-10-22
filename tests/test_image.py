"""Tests for the image generation tool. HTTP faked; no network."""

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.tools_image as ti  # noqa: E402
from app.tools_image import GenerateImageTool, build_image_url  # noqa: E402

_FAKE_JPEG = b"\xff\xd8\xff" + b"x" * 5000


class FakeClient:
    response = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        FakeClient.last_url = url
        return FakeClient.response


class FakeResp:
    def __init__(self, status, ctype, content):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.content = content


def _install(monkeypatch, status=200, ctype="image/jpeg", content=_FAKE_JPEG):
    FakeClient.response = FakeResp(status, ctype, content)
    monkeypatch.setattr(ti.httpx, "AsyncClient", FakeClient)


def run(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setenv("BRO_WORKDIR", str(tmp_path / "work"))
    # The real UniHarness BaseAgentTool has no .run(); tools are invoked
    # through __call__, which validates kwargs against args_schema.
    return asyncio.run(GenerateImageTool()(**kwargs)).output


def test_build_image_url():
    url = build_image_url("a sunset", 512, 512, 42, "flux")
    assert url.startswith("https://image.pollinations.ai/prompt/a%20sunset?")
    assert "width=512" in url and "seed=42" in url and "nologo=true" in url


def test_generate_saves_file(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = run(monkeypatch, tmp_path, prompt="a red barn", width=512, height=512, filename="barn.jpg")
    assert "Saved images/barn.jpg" in out
    p = tmp_path / "work" / "images" / "barn.jpg"
    assert p.is_file() and p.stat().st_size == len(_FAKE_JPEG)
    assert "seed" in FakeClient.last_url


def test_bad_model_rejected(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = run(monkeypatch, tmp_path, prompt="x", model="dall-e-9")
    assert "Unknown model" in out


def test_empty_prompt(monkeypatch, tmp_path):
    _install(monkeypatch)
    assert "Empty prompt" in run(monkeypatch, tmp_path, prompt="  ")


def test_rate_limited(monkeypatch, tmp_path):
    _install(monkeypatch, status=429)
    assert "rate-limited" in run(monkeypatch, tmp_path, prompt="x")


def test_non_image_rejected(monkeypatch, tmp_path):
    _install(monkeypatch, ctype="text/html", content=b"<html>nope</html>" * 200)
    assert "non-image" in run(monkeypatch, tmp_path, prompt="x")


def test_tiny_file_rejected(monkeypatch, tmp_path):
    _install(monkeypatch, content=b"tiny")
    assert "suspiciously small" in run(monkeypatch, tmp_path, prompt="x")


def test_filename_traversal_sanitized(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = run(monkeypatch, tmp_path, prompt="x", filename="../../evil.jpg")
    assert "Saved images/" in out
    assert not (tmp_path / "evil.jpg").exists()
