"""Score commits with the trained triage network.

    python -m nn.score --sha <commit-sha> [--json]
    python -m nn.score --sha HEAD~5..HEAD [--json]  (scores each commit)

Scores a commit's real diff stats with the trained network and
prints its touches-tests probability plus the 3 features most
responsible for the score. Triage ranks only: a high score means
"looks like past test-touching commits", never "this commit touches
the tests".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .data import build_dataset, label_for, repo_root
from .features import extract, top_features
from .mlp import MLP


def load_model(weights_path: str | None = None):
    """Load (net) from weights.json, or None when untrained."""
    path = Path(weights_path) if weights_path else (
        Path(__file__).resolve().parent / "weights.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MLP.from_dict(data["mlp"])


def score_commit_dicts(commits, model=None, threshold: float = 0.5):
    """Score commit dicts (data.py row shape). Returns ranked rows."""
    if model is None:
        model = load_model()
    if model is None:
        raise RuntimeError("no trained weights found. Train first:\n"
                           "  python -m nn.train")
    out = []
    for c in commits:
        x = extract(c)
        score = model.predict_proba(x)
        out.append({
            "sha": c["sha"][:12],
            "subject": c.get("subject"),
            "label": c.get("label"),
            "nn_score": round(score, 4),
            "nn_label": "touches-tests" if score >= threshold else "no-tests",
            "nn_top_features": top_features(c, model),
        })
    out.sort(key=lambda r: r["nn_score"], reverse=True)
    return out


def _commits_for_shas(shas, repo: str) -> list:
    """Build commit dicts for the given shas (or a rev range)."""
    dataset = build_dataset(repo)
    by_sha = {c["sha"]: c for c in dataset["rows"]}
    found = []
    for sha in shas:
        # A range (HEAD~5..HEAD) resolves to many commits; a single
        # rev resolves to exactly one (rev-list on a lone sha would
        # walk its whole ancestry).
        cmd = (["git", "-C", repo, "rev-list", sha] if ".." in sha else
               ["git", "-C", repo, "rev-parse", sha + "^{commit}"])
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 check=True)
            resolved = out.stdout.split()
        except subprocess.CalledProcessError:
            resolved = []
        if not resolved:
            print(f"warning: commit not in dataset: {sha}",
                  file=sys.stderr)
            continue
        for full in resolved:
            if full in by_sha:
                found.append(by_sha[full])
            else:
                print(f"warning: commit not in dataset (merge commits "
                      f"are excluded): {full[:12]}", file=sys.stderr)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score bro-agent commits with the NN.")
    ap.add_argument("--sha", nargs="+", required=True,
                    help="commit sha(s) or a rev range like HEAD~5..HEAD")
    ap.add_argument("--repo", default=None,
                    help="repo root (default: this package's repo)")
    ap.add_argument("--weights", default=None,
                    help="weights.json path (default: nn/weights.json)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", action="store_true",
                    help="print the ranked rows as JSON")
    args = ap.parse_args(argv)
    repo = args.repo or str(repo_root())
    model = load_model(args.weights)
    if model is None:
        print("no trained weights found. Train first:\n"
              "  python -m nn.train", file=sys.stderr)
        return 2
    commits = _commits_for_shas(args.sha, repo)
    if not commits:
        print("no matching commits in the dataset", file=sys.stderr)
        return 2
    rows = score_commit_dicts(commits, model=model,
                              threshold=args.threshold)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'score':>6}  {'label':<8}  {'true':<8}  sha / subject")
    for r in rows:
        true = ("touches-tests" if r["label"] == 1 else "no-tests"
                if r["label"] is not None else "?")
        print(f"{r['nn_score']:>6.3f}  {r['nn_label']:<8}  {true:<8}  "
              f"{r['sha']} {r['subject']}")
        for tf in r["nn_top_features"]:
            print(f"         {tf['direction']} {tf['feature']} "
                  f"({tf['delta']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
