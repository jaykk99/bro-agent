"""Training data: real labeled commits from this repo's own git history.

Source: `git log --no-merges` in the repo root (the checkout this
package lives in). Each commit is one row; features come from its
real diff stats (see features.py), the label from its file list.

Labels (triage only):
- 1 = touches tests: at least one changed file lives under tests/.
- 0 = touches no tests.

Why this label: bro-agent's main history is small (71 commits) and
only 5 of them are fix-prefixed, so a bugfix label would leave ~1
positive in the test split: untrainable. Test-touching is the only
binary label on this history with both classes represented in both
splits (49 touch / 14 do not, of 63 non-merge commits). It is a
thin dataset and the model is weak; the metrics say so plainly.

Merge commits are excluded: their diff stats double-count the merged
branch. Commits with no file changes are kept (all-zero diff
features, real label) rather than dropped.

No feature may read test-ness (features.py): the frac_test_files and
touches_tests features are deliberately absent, since they would be
the label itself.
"""
from __future__ import annotations

import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LABEL_TOUCHES_TESTS = 1
LABEL_NO_TESTS = 0

# git log --numstat record separator: \x1f between fields, \x1e between
# commits. The separator goes FIRST in the format: git prints the
# format string before the --numstat block, so a trailing separator
# would land between the header and its own numstat lines.
LOG_FORMAT = "%x1e%H%x1f%at%x1f%s"


def repo_root() -> Path:
    """Root of the repo this package lives in (nn/ sits at repo root)."""
    return Path(__file__).resolve().parent.parent


def label_for(commit: Dict) -> int:
    """1 if any changed file lives under tests/, else 0."""
    for path, _i, _d in commit.get("files") or []:
        if path == "tests" or path.startswith("tests/"):
            return LABEL_TOUCHES_TESTS
    return LABEL_NO_TESTS


def build_dataset(repo: Optional[str] = None,
                  limit: Optional[int] = None) -> Dict:
    """Build the labeled training set from git history."""
    root = Path(repo).expanduser() if repo else repo_root()
    out = subprocess.run(
        ["git", "-C", str(root), "log", "--no-merges",
         f"--format={LOG_FORMAT}", "--numstat"],
        capture_output=True, text=True, check=True)
    raw = out.stdout
    commits: List[Dict] = []
    for chunk in raw.split("\x1e"):
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
            # Binary files show "-\t-\t<path>": 0/0, never fatal.
            try:
                ins = int(ins_s)
            except ValueError:
                ins = 0
            try:
                dele = int(del_s)
            except ValueError:
                dele = 0
            files.append((path, ins, dele))
        commits.append({
            "sha": sha,
            "subject": subject,
            "author_ts": int(ts),
            "files": files,
            "label": label_for({"files": files}),
        })
        if limit and len(commits) >= limit:
            break

    stats = {
        "repo": str(root),
        "commits": len(commits),
        "train_rows": len(commits),
        "n_positive": sum(1 for c in commits if c["label"] == 1),
        "n_negative": sum(1 for c in commits if c["label"] == 0),
        "merges_excluded_note": "git log --no-merges; merge commits excluded",
    }
    return {"rows": commits, "stats": stats}


def stratified_split(rows: List[Dict], test_frac: float = 0.2,
                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/test split (keeps the class ratio in both)."""
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train: List[Dict] = []
    test: List[Dict] = []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac)) if len(items) > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
