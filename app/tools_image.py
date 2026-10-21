"""Image generation tool — keyless Pollinations image API.

Jay makes faceless ad creative and explainers; the agent can now generate
images itself: prompt in, JPEG out, saved under workdir/images/.

API: GET https://image.pollinations.ai/prompt/{urlencoded}?width=&height=
&seed=&model=&nologo=true&private=true  (no key needed)
The tool downloads the bytes itself (validating content-type is image/*,
15 MB cap) so the file lands on disk even if the model only sees the path.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from .sandbox import SandboxViolation, display_path, resolve, workdir
from .tool_base import BaseAgentTool, ToolResult

IMAGE_API = "https://image.pollinations.ai/prompt"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_MODELS = {"flux", "turbo"}


def build_image_url(prompt: str, width: int, height: int, seed: int, model: str) -> str:
    params = (
        f"width={width}&height={height}&seed={seed}&model={model}"
        "&nologo=true&private=true&enhance=true"
    )
    return f"{IMAGE_API}/{quote(prompt, safe='')}?{params}"


class GenerateImageParams(BaseModel):
    prompt: str = Field(description="Image description, in English, concrete and specific")
    width: int = Field(default=1024, description="Width in px (256-2048)")
    height: int = Field(default=1024, description="Height in px (256-2048)")
    seed: int = Field(default=0, description="Seed for reproducibility (0 = random)")
    model: str = Field(default="flux", description="flux (quality) or turbo (fast)")
    filename: str = Field(default="", description="Output filename (default: auto from timestamp)")


class GenerateImageTool(BaseAgentTool[GenerateImageParams]):
    name: str = "generate_image"
    description: str = (
        "Generate an AI image from a text prompt (keyless). Saves a JPEG "
        "under workdir/images/ and returns its path. Good for ad creative, "
        "thumbnails, mockups, and explainer visuals."
    )
    args_schema = GenerateImageParams

    async def execute(self, params: GenerateImageParams) -> ToolResult:
        prompt = (params.prompt or "").strip()
        if not prompt:
            return ToolResult(output="Empty prompt.")
        if len(prompt) > 1500:
            return ToolResult(output="Prompt too long (1500 char cap).")
        width = max(256, min(params.width, 2048))
        height = max(256, min(params.height, 2048))
        seed = params.seed if params.seed else random.randint(1, 999_999_999)
        model = (params.model or "flux").lower()
        if model not in ALLOWED_MODELS:
            return ToolResult(output=f"Unknown model {model!r} (flux or turbo).")

        root = workdir()
        name = (params.filename or "").strip()
        if name:
            name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))[:80]
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                name += ".jpg"
        else:
            name = f"img-{int(time.time())}-{seed % 10000}.jpg"
        try:
            dest = resolve(root, os.path.join("images", name))
        except SandboxViolation as e:
            return ToolResult(output=f"Blocked: {e}")

        url = build_image_url(prompt, width, height, seed, model)
        try:
            async with httpx.AsyncClient(timeout=120, headers={"User-Agent": "bro-agent/1.0"}) as c:
                r = await c.get(url)
                if r.status_code == 429:
                    return ToolResult(output="Image API rate-limited (429) — try again in a bit.")
                if r.status_code >= 400:
                    return ToolResult(output=f"Image API error: HTTP {r.status_code}")
                ctype = r.headers.get("content-type", "")
                if "image" not in ctype:
                    return ToolResult(output=f"Image API returned non-image content ({ctype[:60]}).")
                data = r.content
                if len(data) > MAX_IMAGE_BYTES:
                    return ToolResult(output="Image too large, discarded.")
                if len(data) < 1024:
                    return ToolResult(output="Image API returned a suspiciously small file, discarded.")
        except Exception as e:
            return ToolResult(output=f"Image generation failed: {e}")

        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
        except OSError as e:
            return ToolResult(output=f"Save failed: {e}")
        return ToolResult(
            output=f"Saved {display_path(root, dest)} ({len(data)} bytes, {width}x{height}, seed {seed}, model {model})"
        )


def image_tools() -> list[Any]:
    return [GenerateImageTool()]
