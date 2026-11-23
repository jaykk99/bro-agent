"""Pin the remaining untested paths in app/tools_image.py.

Coverage gaps left after test_image.py: the prompt-length cap, the
auto .jpg extension, the sandbox-violation mapping, the API error
status, the oversize-image discard, the fetch exception branch, the
save-failure branch, and the image_tools() registry. HTTP is faked.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app.tools_image as ti  # noqa: E402
from app.tools_image import GenerateImageTool, image_tools  # noqa: E402
from app.sandbox import SandboxViolation  # noqa: E402

_FAKE_JPEG = b"\xff\xd8\xff" + b"x" * 5000


class FakeResp:
    def __init__(self, status, ctype, content):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.content = content


class FakeClient:
    response = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        return FakeClient.response


class RaisingClient(FakeClient):
    async def get(self, url):
        raise RuntimeError("network down")


def _install(monkeypatch, client=FakeClient, status=200, ctype="image/jpeg", content=_FAKE_JPEG):
    FakeClient.response = FakeResp(status, ctype, content)
    monkeypatch.setattr(ti.httpx, "AsyncClient", client)


def run(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setenv("BRO_WORKDIR", str(tmp_path / "work"))
    return asyncio.run(GenerateImageTool()(**kwargs)).output


def test_prompt_too_long(monkeypatch, tmp_path):
    _install(monkeypatch)
    assert "Prompt too long" in run(monkeypatch, tmp_path, prompt="x" * 1501)


def test_filename_without_extension_gets_jpg(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = run(monkeypatch, tmp_path, prompt="x", filename="pic")
    assert "Saved images/pic.jpg" in out
    assert (tmp_path / "work" / "images" / "pic.jpg").is_file()


def test_sandbox_violation_is_blocked(monkeypatch, tmp_path):
    _install(monkeypatch)

    def boom(root, path):
        raise SandboxViolation("escapes workdir")

    monkeypatch.setattr(ti, "resolve", boom)
    out = run(monkeypatch, tmp_path, prompt="x")
    assert out.startswith("Blocked:")


def test_api_error_status(monkeypatch, tmp_path):
    _install(monkeypatch, status=500)
    assert "Image API error: HTTP 500" in run(monkeypatch, tmp_path, prompt="x")


def test_oversize_image_discarded(monkeypatch, tmp_path):
    _install(monkeypatch, content=b"x" * (ti.MAX_IMAGE_BYTES + 1))
    assert "Image too large" in run(monkeypatch, tmp_path, prompt="x")


def test_fetch_exception_is_reported(monkeypatch, tmp_path):
    _install(monkeypatch, client=RaisingClient)
    assert "Image generation failed: network down" in run(monkeypatch, tmp_path, prompt="x")


def test_save_failure_is_reported(monkeypatch, tmp_path):
    _install(monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    # A regular file where the images/ dir should be: makedirs fails.
    (work / "images").write_text("not a dir")
    out = run(monkeypatch, tmp_path, prompt="x")
    assert out.startswith("Save failed:")


def test_image_tools_registry():
    assert [t.name for t in image_tools()] == ["generate_image"]
