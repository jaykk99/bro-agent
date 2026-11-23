"""Subprocess entry-point tests: lines only a fresh interpreter can
execute — the `if __name__ == "__main__"` guards, the guard_policy
ImportError fallback (refusal.py deleted), and the tool_base
production path (uniharness installed). Each runs under coverage in
its own process and asserts the target lines executed.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = os.path.join(os.path.dirname(__file__), "..")


def _cov_lines(tmp_path, name, argv, env_extra=None, stdin_data=None):
    """Run argv under coverage in a subprocess; return the missing-line
    set for one app module, or raise if the process misbehaves."""
    cov_file = tmp_path / f".coverage-{name}"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.update(env_extra or {})
    proc = subprocess.Popen(
        [sys.executable, "-m", "coverage", "run", "--data-file", str(cov_file), *argv],
        cwd=REPO,
        env=env,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        out, _ = proc.communicate(input=stdin_data, timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise AssertionError(f"{name}: subprocess timed out")
    assert proc.returncode == 0, f"{name}: exit {proc.returncode}\n{out}"
    return cov_file, out


def _missing(cov_file, module_relpath):
    from coverage import Coverage

    cov = Coverage(data_file=str(cov_file))
    cov.load()
    _f, _s, _e, missing, _m = cov.analysis2(os.path.join(REPO, module_relpath))
    return set(missing)


def test_cli_main_guard_runs(tmp_path):
    # Unreachable server: main() prints "can't reach ..." and returns
    # before reading stdin, so this exits fast with no server.
    cov_file, out = _cov_lines(
        tmp_path,
        "cli",
        ["-m", "app.cli"],
        env_extra={"BRO_BASE_URL": "http://127.0.0.1:1"},
        stdin_data="/quit\n",
    )
    assert "can't reach" in out
    assert 202 not in _missing(cov_file, "app/cli.py")


def test_guard_policy_fallback_when_refusal_missing(tmp_path):
    # Block app.refusal at import time: guard_policy must define its
    # own identical BoundaryHit instead of crashing.
    script = tmp_path / "block_refusal.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "sys.modules['app.refusal'] = None\n"
        "import app.guard_policy as gp\n"
        "hit = gp.BoundaryHit(blocked=True, reason='r', alternative='a')\n"
        "assert hit.blocked and hit.reason == 'r'\n"
        "assert gp.check_owner_boundary('hello') is None\n"
    )
    cov_file, _out = _cov_lines(tmp_path, "guard-policy", [str(script)])
    missing = _missing(cov_file, "app/guard_policy.py")
    assert not ({85, 86, 87, 88, 89, 90, 91} & missing)


def test_tool_base_uses_real_uniharness_when_installed(tmp_path):
    # A minimal fake uniharness on PYTHONPATH: the production import
    # path must win over the fallback.
    fake = tmp_path / "fake-uniharness"
    (fake / "uniharness").mkdir(parents=True)
    (fake / "uniharness" / "__init__.py").write_text("")
    (fake / "uniharness" / "tools.py").write_text("class BaseAgentTool:\n    pass\n")
    (fake / "uniharness" / "types.py").write_text("class ToolResult:\n    pass\n")
    script = tmp_path / "check_tool_base.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "import app.tool_base as tb\n"
        "assert tb.USING_UNIHARNESS is True\n"
        "assert tb.BaseAgentTool is not None and tb.ToolResult is not None\n"
    )
    cov_file, _out = _cov_lines(
        tmp_path,
        "tool-base",
        [str(script)],
        env_extra={"PYTHONPATH": str(fake)},
    )
    missing = _missing(cov_file, "app/tool_base.py")
    assert not ({19, 20, 21, 22, 23} & missing)
