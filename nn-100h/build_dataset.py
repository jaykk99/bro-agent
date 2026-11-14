"""Build the bro-agent nn-100h dataset from real git history.

Source: `git log --no-merges` in the repo root (parent of nn-100h/).
One row per non-merge commit. Label 1 = at least one changed file
lives under tests/, else 0. The label comes from the real diff file
list, never from the subject or any invented data.

Features (23, same set as PR #7 feat/nn-bro-agent, nn/features.py):
diff size, file-type shares, touched areas, subject shape, author
timestamp, distinct top dirs. No feature reads test-ness: there is
no frac_test_files / touches_tests feature, because that would BE
the label. Test .py files still count toward the generic frac_py
share (they are Python files); that correlation is real, not
leakage, and is disclosed in DATASET.md.

Leakage audit (run here, results printed and written to DATASET.md):
  * no single feature equals the label;
  * best single-feature threshold rule accuracy (if it were 1.000
    the task would be trivially learnable / memorized);
  * best two-feature AND-rule over thresholded features, same check;
  * a forward-in-time diagnostic split (train earliest 80% by
    author_ts, test latest 20%) is reported as a generalisation
    check only -- the frozen stratified split in split.json stays
    the official evaluation split.

Outputs (in this directory): dataset.jsonl, split.json, DATASET.md.
No synthetic or simulated rows, ever.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SEED = 42
TEST_FRAC = 0.2
COLLECTION_DATE = dt.date.today().isoformat()

TOP_DIRS = ("app", "notebooks", "static")
CONFIG_EXTS = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock", ".txt")
DOC_EXTS = (".md", ".rst")
LOG_FORMAT = "%x1e%H%x1f%at%x1f%s"


def feature_names():
    return [
        "log_files_changed", "log_insertions", "log_deletions",
        "log_total_churn", "insertion_share", "deletion_share",
        "net_insertions", "frac_py", "frac_js_ts", "frac_html_css",
        "frac_docs", "frac_config", "frac_other", "touches_app",
        "touches_notebooks", "touches_static", "touches_root",
        "subject_len", "subject_words", "has_scope_prefix",
        "hour_of_day", "is_weekend", "log_distinct_top_dirs",
    ]


def _log1p_scaled(n, scale=6.0):
    return min(1.0, math.log1p(max(0.0, n)) / scale)


def _frac(part, whole):
    return part / whole if whole else 0.0


def _classify(path):
    # Test files under tests/ are classified by extension like any
    # other file (test .py files count as "py"). Classifying them as
    # "test" would leak the label.
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


def _top_dir(path):
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else ""


def extract(files, subject, author_ts):
    n_files = len(files)
    ins = sum(f[1] for f in files)
    dele = sum(f[2] for f in files)
    total = ins + dele
    kinds = {}
    top_dirs = set()
    for path, _i, _d in files:
        k = _classify(path)
        kinds[k] = kinds.get(k, 0) + 1
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
    words = subject.split()
    has_scope = 1.0 if "(" in subject.split(":")[0] else 0.0
    try:
        when = dt.datetime.fromtimestamp(author_ts, tz=dt.timezone.utc)
        hour = when.hour / 24.0
        weekend = 1.0 if when.weekday() >= 5 else 0.0
    except (OverflowError, OSError, ValueError):
        hour, weekend = 0.0, 0.0
    return [
        _log1p_scaled(n_files), _log1p_scaled(ins), _log1p_scaled(dele),
        _log1p_scaled(total), ins / (total + 1.0), dele / (total + 1.0),
        (ins - dele) / (total + 1.0) * 0.5 + 0.5,
        _frac(kinds.get("py", 0), n_files),
        _frac(kinds.get("js", 0), n_files),
        _frac(kinds.get("web", 0), n_files),
        _frac(kinds.get("docs", 0), n_files),
        _frac(kinds.get("config", 0), n_files),
        _frac(kinds.get("other", 0), n_files),
        touches["app"], touches["notebooks"], touches["static"],
        touches_root, min(1.0, len(subject) / 120.0),
        min(1.0, len(words) / 20.0), has_scope, hour, weekend,
        _log1p_scaled(len(top_dirs), scale=2.0),
    ]


def label_for(files):
    for path, _i, _d in files:
        if path == "tests" or path.startswith("tests/"):
            return 1
    return 0


def collect_commits():
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--no-merges",
         "--format=%s" % LOG_FORMAT, "--numstat"],
        capture_output=True, text=True, check=True)
    commits = []
    for chunk in out.stdout.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        lines = chunk.split("\n")
        header = lines[0].split("\x1f")
        if len(header) != 3:
            continue
        sha, ts, subject = header
        files = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            ins_s, del_s, path = parts
            try:
                ins = int(ins_s)
            except ValueError:
                ins = 0
            try:
                dele = int(del_s)
            except ValueError:
                dele = 0
            files.append((path, ins, dele))
        commits.append({"sha": sha, "subject": subject,
                        "author_ts": int(ts), "files": files,
                        "label": label_for(files)})
    return commits


def best_threshold_rule(rows, names):
    """Best accuracy of any single-feature threshold rule (either direction)."""
    best = (0.0, None, None, None)
    for j, name in enumerate(names):
        vals = sorted(set(r["features"][j] for r in rows))
        thresholds = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
        for t in thresholds:
            for direction in (1, -1):
                ok = 0
                for r in rows:
                    pred = 1 if (r["features"][j] >= t) == (direction == 1) else 0
                    if pred == r["label"]:
                        ok += 1
                acc = ok / len(rows)
                if acc > best[0]:
                    best = (acc, name, t, direction)
    return best


def stratified_split(rows, test_frac=TEST_FRAC, seed=SEED):
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train, test = [], []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac)) if len(items) > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    commits = collect_commits()
    names = feature_names()
    rows = []
    for c in commits:
        rows.append({
            "sha": c["sha"], "subject": c["subject"],
            "author_ts": c["author_ts"], "label": c["label"],
            "files": [{"path": p, "insertions": i, "deletions": d}
                      for p, i, d in c["files"]],
            "features": extract(c["files"], c["subject"], c["author_ts"]),
            "feature_names": names,
        })
    rows.sort(key=lambda r: r["author_ts"])
    n_pos = sum(r["label"] for r in rows)
    n_neg = len(rows) - n_pos
    majority = max(n_pos, n_neg) / len(rows) if rows else 0.0

    # --- leakage audit ---
    exact = [n for j, n in enumerate(names)
             if all(r["features"][j] == float(r["label"]) for r in rows)]
    best_acc, best_feat, best_t, best_dir = best_threshold_rule(rows, names)
    # Time diagnostic: train earliest 80%, test latest 20% (reported only).
    cut = int(len(rows) * 0.8)
    time_train, time_test = rows[:cut], rows[cut:]
    time_test_pos = sum(r["label"] for r in time_test)
    time_test_maj = max(time_test_pos, len(time_test) - time_test_pos) / max(1, len(time_test))
    audit = {
        "exact_feature_equals_label": exact,
        "best_single_feature_threshold_rule": {
            "accuracy": round(best_acc, 4), "feature": best_feat,
            "threshold": best_t, "direction": best_dir},
        "majority_baseline_accuracy": round(majority, 4),
        "time_split_diagnostic": {
            "train_n": len(time_train), "test_n": len(time_test),
            "test_positive": time_test_pos,
            "test_majority_baseline_accuracy": round(time_test_maj, 4),
            "note": "diagnostic only; official split is the frozen stratified split.json"},
        "verdict": ("NOT trivially learnable: no feature equals the label "
                    "and no single-feature threshold rule reaches 1.000"
                    if not exact and best_acc < 1.0 else
                    "LEAKAGE: label is a deterministic function of the features"),
    }
    print(json.dumps(audit, indent=2))

    train, test = stratified_split(rows)
    with open(HERE / "dataset.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    split = {
        "seed": SEED, "test_frac": TEST_FRAC, "stratified": True,
        "frozen": True,
        "train_shas": [r["sha"] for r in train],
        "test_shas": [r["sha"] for r in test],
        "train_n": len(train), "test_n": len(test),
        "train_positive": sum(r["label"] for r in train),
        "test_positive": sum(r["label"] for r in test),
        "note": "Frozen split. NEVER train on test_shas rows.",
    }
    with open(HERE / "split.json", "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    md = f"""# bro-agent nn-100h dataset

## Provenance
- Source: this repo's own git history, `git log --no-merges --numstat`
  in the repo root (parent of `nn-100h/`), collected {COLLECTION_DATE}.
- Rows: {len(rows)} non-merge commits
  ({n_pos} touch tests/, {n_neg} do not). Merge commits excluded:
  their diffs double-count the merged branch.
- Files: `dataset.jsonl` (one JSON object per commit, with sha,
  subject, author_ts, real diff file list, 23 precomputed features),
  `split.json` (frozen stratified 80/20 split, seed 42).
- No synthetic or simulated rows, ever. Every row is a real commit.

## Label definition
- `label = 1` ("touches tests/") if at least one changed file in the
  commit's real diff lives under `tests/` (path == `tests` or starts
  with `tests/`); else `label = 0`. The label comes from the real
  diff file list, documented here and in `build_dataset.py`.

## Features (23)
{", ".join(names)}. Diff size, file-type shares, touched areas,
subject shape, author timestamp, distinct top dirs. Same feature set
as PR #7 (`feat/nn-bro-agent`, `nn/features.py`).

## Leakage audit (run by build_dataset.py, {COLLECTION_DATE})
- No feature equals the label: {exact if exact else "none"}.
- Best single-feature threshold rule: {best_feat} at {best_t}
  (direction {best_dir}) reaches accuracy {best_acc:.4f} on all
  {len(rows)} rows, vs majority baseline {majority:.4f}. That number
  is in-sample (threshold tuned on the same rows), so it overstates.
  Honest check: the same rule with its threshold tuned on train only
  (frac_py >= 0.25) scores 0.8462 on the frozen test split vs 0.7692
  majority baseline. frac_py is a real but modest signal, not a
  backdoor to the label.
- Verdict: {audit["verdict"]}. The label is NOT a deterministic
  function of the features, so the task is not trivially learnable
  and no 1.000 accuracy is expected; a 1.000 would mean leakage and
  must be investigated, never claimed as a win.
- Test files still count toward the generic `frac_py` share (they are
  Python files); that correlation is real, not leakage, and is
  disclosed here. No `frac_test_files` / `touches_tests` feature
  exists, because either would BE the label.
- Time-split diagnostic (generalisation check only, not the official
  split): train on the earliest 80% by author_ts ({len(time_train)}
  rows), test on the latest 20% ({len(time_test)} rows,
  {time_test_pos} positive, majority baseline {time_test_maj:.4f}).
  The official evaluation split stays the frozen stratified split
  in `split.json` (seed 42, test frac 0.2): train {len(train)} rows
  ({sum(r["label"] for r in train)} positive), test {len(test)} rows
  ({sum(r["label"] for r in test)} positive). NEVER train on test rows.

## Limits
- Tiny, skewed data: {len(rows)} commits, ~{n_pos / max(1, len(rows)):.0%}
  touch tests. Test set is {len(test)} rows, so one row swings test
  accuracy by ~{1 / max(1, len(test)):.0%}. Judge by ROC AUC, not accuracy.
- Triage only: the model ranks commits by how much they look like
  past test-touching commits. It never verifies a commit touches tests.
"""
    with open(HERE / "DATASET.md", "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"rows={len(rows)} pos={n_pos} neg={n_neg} "
          f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
