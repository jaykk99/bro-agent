"""Feature extraction: one git commit -> fixed-length numeric vector.

Every feature comes from the commit's real diff stats (git log
--numstat), its subject line's shape (length, word count, scoped
prefix), and its author timestamp. Nothing is invented.

LEAKAGE RULE: the label is "commit touches tests/" (see data.py),
so no feature may read whether a file is under tests/ or how many
test files a commit has: the frac_test_files and touches_tests
features are deliberately absent. Test files still count toward the
generic frac_py share (they are Python files); that correlation is
real, not leakage, and is disclosed in the README and viewer limits.
Subject shape is safe here: the label does not come from the subject.

Commit dict shape (built by data.build_dataset):
    sha, subject, author_ts, files: [(path, insertions, deletions)]
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

TOP_DIRS = ("app", "notebooks", "static")

CONFIG_EXTS = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
               ".lock", ".txt")
DOC_EXTS = (".md", ".rst")
CODE_EXTS = (".py", ".js", ".ts", ".jsx", ".tsx")
WEB_EXTS = (".html", ".css")


def _log1p_scaled(n: float, scale: float = 6.0) -> float:
    return min(1.0, math.log1p(max(0.0, n)) / scale)


def _frac(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def feature_names() -> List[str]:
    return [
        "log_files_changed",
        "log_insertions",
        "log_deletions",
        "log_total_churn",
        "insertion_share",
        "deletion_share",
        "net_insertions",
        "frac_py",
        "frac_js_ts",
        "frac_html_css",
        "frac_docs",
        "frac_config",
        "frac_other",
        "touches_app",
        "touches_notebooks",
        "touches_static",
        "touches_root",
        "subject_len",
        "subject_words",
        "has_scope_prefix",
        "hour_of_day",
        "is_weekend",
        "log_distinct_top_dirs",
    ]


def groups() -> Dict[str, Tuple[int, int]]:
    """Named index ranges, for permutation importance."""
    names = feature_names()
    idx = {n: i for i, n in enumerate(names)}

    def span(prefix: str) -> Tuple[int, int]:
        hits = [i for n, i in idx.items() if n.startswith(prefix)]
        return (min(hits), max(hits) + 1)

    return {
        "diff_size": (idx["log_files_changed"], idx["net_insertions"] + 1),
        "file_types": (idx["frac_py"], idx["frac_other"] + 1),
        "touched_areas": (idx["touches_app"], idx["touches_root"] + 1),
        "subject_shape": (idx["subject_len"], idx["has_scope_prefix"] + 1),
        "timestamp": (idx["hour_of_day"], idx["is_weekend"] + 1),
        "distinct_top_dirs": (idx["log_distinct_top_dirs"],
                              idx["log_distinct_top_dirs"] + 1),
    }


def _classify(path: str) -> str:
    # NOTE: files under tests/ are classified by extension like any
    # other file (test .py files count as "py"). Classifying them as
    # "test" would leak the label (label = touches tests/).
    p = path.lower()
    _, ext = os.path.splitext(p)
    base = os.path.basename(p)
    if ext == ".py":
        return "py"
    if ext in (".js", ".ts", ".jsx", ".tsx"):
        return "js"
    if ext in (".html", ".css"):
        return "web"
    if ext in DOC_EXTS:
        return "docs"
    if ext in CONFIG_EXTS or base in ("dockerfile", "makefile"):
        return "config"
    return "other"


def _top_dir(path: str) -> str:
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else ""


def extract(commit: Dict) -> List[float]:
    """One commit dict -> feature vector in feature_names() order."""
    files = commit.get("files") or []
    n_files = len(files)
    ins = sum(f[1] for f in files)
    dele = sum(f[2] for f in files)
    total = ins + dele

    kinds: Dict[str, int] = {}
    top_dirs = set()
    for path, _i, _d in files:
        kinds[_classify(path)] = kinds.get(_classify(path), 0) + 1
        d = _top_dir(path)
        if d:
            top_dirs.add(d)

    touches = {d: 0.0 for d in TOP_DIRS}
    touches_root = 0.0
    for path, _i, _d in files:
        d = _top_dir(path)
        if d in touches:
            touches[d] = 1.0
        elif not d:
            touches_root = 1.0

    subject = commit.get("subject") or ""
    words = subject.split()
    has_scope = 1.0 if "(" in subject.split(":")[0] else 0.0
    # Author timestamp -> hour of day + weekend flag (UTC).
    import datetime as dt
    ts = commit.get("author_ts") or 0
    try:
        when = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        hour = when.hour / 24.0
        weekend = 1.0 if when.weekday() >= 5 else 0.0
    except (OverflowError, OSError, ValueError):
        hour, weekend = 0.0, 0.0

    return [
        _log1p_scaled(n_files),
        _log1p_scaled(ins),
        _log1p_scaled(dele),
        _log1p_scaled(total),
        ins / (total + 1.0),
        dele / (total + 1.0),
        (ins - dele) / (total + 1.0) * 0.5 + 0.5,
        _frac(kinds.get("py", 0), n_files),
        _frac(kinds.get("js", 0), n_files),
        _frac(kinds.get("web", 0), n_files),
        _frac(kinds.get("docs", 0), n_files),
        _frac(kinds.get("config", 0), n_files),
        _frac(kinds.get("other", 0), n_files),
        touches["app"],
        touches["notebooks"],
        touches["static"],
        touches_root,
        min(1.0, len(subject) / 120.0),
        min(1.0, len(words) / 20.0),
        has_scope,
        hour,
        weekend,
        _log1p_scaled(len(top_dirs), scale=2.0),
    ]


def top_features(commit: Dict, net, k: int = 3) -> List[Dict]:
    """Top contributing features for one scored commit (leave-one-out)."""
    x = extract(commit)
    base = net.predict_proba(x)
    names = feature_names()
    deltas = []
    for i, name in enumerate(names):
        if x[i] == 0.0:
            continue
        x2 = list(x)
        x2[i] = 0.0
        d = base - net.predict_proba(x2)
        deltas.append((abs(d), d, name))
    deltas.sort(reverse=True)
    return [{"feature": name, "direction": "up" if d > 0 else "down",
             "delta": round(d, 4)} for _a, d, name in deltas[:k]]
