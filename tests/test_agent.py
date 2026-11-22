"""Agent factory + cache: provider chain, fallback, create, fingerprint.

No network, no real LLM, no UniHarness: langchain_openai and
uniharness are stubbed in sys.modules, and the tool factories are
monkeypatched to return marker objects. The real HordeChatModel
constructor is used (it is cheap and keyless).
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

import app.agent as agent_mod  # noqa: E402
from app.agent import (  # noqa: E402
    POLLINATIONS_API_URL,
    POLLINATIONS_MODEL,
    SYSTEM_PROMPT,
    _pollinations_model,
    agent_cache_fingerprint,
    build_llm,
    create_bro_agent,
    get_cached_agent,
    invalidate_agent_cache,
)


# ---- fakes ---------------------------------------------------------------

class FakeChatOpenAI:
    """Records the kwargs ChatOpenAI was constructed with."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeChatOpenAI.instances.append(self)


class FakeModelProfile:
    def __init__(self, model, context_window):
        self.model = model
        self.context_window = context_window


class FakeComputer:
    pass


@pytest.fixture(autouse=True)
def _clean_cache():
    invalidate_agent_cache()
    FakeChatOpenAI.instances.clear()
    yield
    invalidate_agent_cache()
    FakeChatOpenAI.instances.clear()


@pytest.fixture
def fake_pollinations(monkeypatch):
    """Install a stub langchain_openai module exposing FakeChatOpenAI."""
    mod = types.ModuleType("langchain_openai")
    mod.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", mod)
    return mod


@pytest.fixture
def fake_uniharness(monkeypatch):
    """Install stub uniharness + uniharness.computer modules."""
    created = {}

    async def fake_create_agent(*, model, computer, extra_tools, system_prompt):
        created.update(
            model=model, computer=computer,
            extra_tools=list(extra_tools), system_prompt=system_prompt,
        )
        agent = object()
        created["agent"] = agent
        return agent

    uh = types.ModuleType("uniharness")
    uh.ModelProfile = FakeModelProfile
    uh.create_agent = fake_create_agent
    comp = types.ModuleType("uniharness.computer")
    comp.LocalNativeComputer = FakeComputer
    monkeypatch.setitem(sys.modules, "uniharness", uh)
    monkeypatch.setitem(sys.modules, "uniharness.computer", comp)
    return created


@pytest.fixture
def stub_tool_factories(monkeypatch):
    """Replace the six tool factories with ones returning markers."""
    seen = {}
    for mod_name, fn_name in [
        ("app.tools_web", "web_tools"),
        ("app.tools_composio", "composio_tools"),
        ("app.tools_files", "file_tools"),
        ("app.tools_exec", "exec_tools"),
        ("app.tools_git", "git_tools"),
        ("app.tools_image", "image_tools"),
    ]:
        marker = object()
        seen[fn_name] = marker
        monkeypatch.setattr(f"{mod_name}.{fn_name}", lambda _m=marker: [_m])
    return seen


# ---- _pollinations_model -------------------------------------------------

def test_pollinations_model_keyless_openai_compatible(fake_pollinations):
    model = _pollinations_model()
    assert isinstance(model, FakeChatOpenAI)
    assert model.kwargs["model"] == POLLINATIONS_MODEL
    assert model.kwargs["base_url"] == POLLINATIONS_API_URL
    assert model.kwargs["api_key"] == "not-needed"
    assert model.kwargs["temperature"] == 0.7
    assert len(FakeChatOpenAI.instances) == 1


# ---- build_llm -----------------------------------------------------------

def test_build_llm_horde_first_by_default():
    model, source = build_llm()
    assert source == "horde"
    assert type(model).__name__ == "HordeChatModel"


def test_build_llm_pollinations_when_ordered_first(fake_pollinations):
    model, source = build_llm(("pollinations", "horde"))
    assert source == "pollinations"
    assert isinstance(model, FakeChatOpenAI)


def test_build_llm_skips_unknown_provider(fake_pollinations):
    model, source = build_llm(("not-a-provider", "pollinations"))
    assert source == "pollinations"
    assert isinstance(model, FakeChatOpenAI)


def test_build_llm_init_failure_falls_through_to_next(monkeypatch, fake_pollinations):
    def boom():
        raise RuntimeError("pollinations down")

    monkeypatch.setattr(agent_mod, "_pollinations_model", boom)
    model, source = build_llm(("pollinations", "horde"))
    assert source == "horde"
    assert type(model).__name__ == "HordeChatModel"


def test_build_llm_all_providers_fail_raises(monkeypatch):
    monkeypatch.setattr(agent_mod, "_pollinations_model",
                        lambda: (_ for _ in ()).throw(RuntimeError("p down")))
    with pytest.raises(RuntimeError, match="no LLM provider available"):
        build_llm(("pollinations", "nope"))
    with pytest.raises(RuntimeError, match=r"tried \('pollinations', 'nope'\)"):
        build_llm(("pollinations", "nope"))


def test_build_llm_empty_order_raises():
    with pytest.raises(RuntimeError, match="no LLM provider available"):
        build_llm(())


def test_build_llm_unknown_only_raises():
    with pytest.raises(RuntimeError, match="no LLM provider available"):
        build_llm(("ghost",))


# ---- create_bro_agent ----------------------------------------------------

def test_create_bro_agent_default_chain_prefers_horde(
    tmp_path, monkeypatch, fake_uniharness, stub_tool_factories,
):
    workdir = tmp_path / "bro-work"
    monkeypatch.setenv("BRO_WORKDIR", str(workdir))
    agent, source = asyncio.run(create_bro_agent())
    assert source == "horde"
    assert agent is fake_uniharness["agent"]
    assert fake_uniharness["model"].context_window == 128_000
    assert type(fake_uniharness["model"].model).__name__ == "HordeChatModel"
    assert isinstance(fake_uniharness["computer"], FakeComputer)
    assert fake_uniharness["system_prompt"] == SYSTEM_PROMPT
    assert len(fake_uniharness["extra_tools"]) == 6
    assert set(fake_uniharness["extra_tools"]) == set(stub_tool_factories.values())
    assert workdir.is_dir()


def test_create_bro_agent_preferred_pollinations(
    tmp_path, monkeypatch, fake_uniharness, stub_tool_factories, fake_pollinations,
):
    monkeypatch.setenv("BRO_WORKDIR", str(tmp_path / "w"))
    agent, source = asyncio.run(create_bro_agent(preferred="pollinations"))
    assert source == "pollinations"
    assert isinstance(fake_uniharness["model"].model, FakeChatOpenAI)
    assert agent is fake_uniharness["agent"]


def test_create_bro_agent_preferred_unknown_raises(
    tmp_path, monkeypatch, fake_uniharness, stub_tool_factories,
):
    monkeypatch.setenv("BRO_WORKDIR", str(tmp_path / "w"))
    with pytest.raises(RuntimeError, match="no LLM provider available"):
        asyncio.run(create_bro_agent(preferred="ghost"))


def test_create_bro_agent_default_workdir_under_home(
    monkeypatch, fake_uniharness, stub_tool_factories,
):
    monkeypatch.delenv("BRO_WORKDIR", raising=False)
    monkeypatch.setenv("HOME", "/tmp/bro-agent-test-home")
    agent, source = asyncio.run(create_bro_agent(preferred="horde"))
    assert source == "horde"
    assert os.path.isdir("/tmp/bro-agent-test-home/bro-work")


# ---- fingerprint ---------------------------------------------------------

def test_fingerprint_stable_and_tracks_inputs(monkeypatch):
    a = agent_cache_fingerprint("horde")
    assert agent_cache_fingerprint("horde") == a
    assert agent_cache_fingerprint(None) != a
    assert agent_cache_fingerprint("pollinations") != a
    monkeypatch.setenv("HORDE_TIMEOUT_S", "45")
    assert agent_cache_fingerprint("horde") != a
    monkeypatch.setenv("BRO_WORKDIR", "/tmp/bro-work-fp-test")
    assert agent_cache_fingerprint("horde") != a


# ---- get_cached_agent ----------------------------------------------------

def test_get_cached_agent_double_checked_lock_builds_once(monkeypatch):
    builds = []

    async def fake_build(preferred=None):
        await asyncio.sleep(0.05)
        builds.append(1)
        return object(), preferred or "horde"

    monkeypatch.setattr(agent_mod, "create_bro_agent", fake_build)

    async def both():
        return await asyncio.gather(
            get_cached_agent(preferred="horde"),
            get_cached_agent(preferred="horde"),
        )

    (a1, s1, hit1), (a2, s2, hit2) = asyncio.run(both())
    assert len(builds) == 1, "concurrent first turns must build exactly once"
    assert a1 is a2 and s1 == s2 == "horde"
    assert {hit1, hit2} == {False, True}, "one caller builds, the other hits inside the lock"


def test_get_cached_agent_separate_fingerprints_build_separately(monkeypatch):
    builds = []

    async def fake_build(preferred=None):
        builds.append(preferred)
        return object(), preferred or "chain"

    monkeypatch.setattr(agent_mod, "create_bro_agent", fake_build)
    asyncio.run(get_cached_agent(preferred="horde"))
    asyncio.run(get_cached_agent(preferred="pollinations"))
    asyncio.run(get_cached_agent(preferred="horde"))
    assert builds == ["horde", "pollinations"], "distinct fingerprints build, same one reuses"


def test_invalidate_agent_cache_returns_count(monkeypatch):
    assert invalidate_agent_cache() == 0

    async def fake_build(preferred=None):
        return object(), "horde"

    monkeypatch.setattr(agent_mod, "create_bro_agent", fake_build)
    asyncio.run(get_cached_agent(preferred="horde"))
    asyncio.run(get_cached_agent(preferred="pollinations"))
    assert invalidate_agent_cache() == 2
    assert invalidate_agent_cache() == 0
