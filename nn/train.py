"""Train the commit triage network on the repo's real git history.

    python -m nn.train [--repo PATH] [--out DIR] [--epochs N]
                       [--hidden 16,8]

Trains an MLP to predict touches-tests-vs-not from diff stats
alone (no test-ness features: the label is "touches tests/", so
reading test files would leak it). Writes weights.json +
viewer-data.json into the out dir (default: this package dir) and
refreshes the embedded metrics block in viewer.html.

Honest limits, printed with the metrics and baked into the viewer:
- the data is tiny (63 non-merge commits) and skewed (~78% touch
  tests), so the majority baseline is ~0.78 and the test set holds
  only 13 rows: accuracy has to beat 0.78 to mean anything, and all
  metrics here are noisy. Judge it by ROC AUC, not accuracy.
- a commit's diff stats say how big it was and where it touched, not
  why. The model ranks commits by how much they look like past
  test-touching commits; it never verifies that a commit does (or
  does not) touch the tests.
- the bugfix label was untrainable here (only 5 fix-prefixed commits
  on main), so this network predicts test-touching instead. It says
  nothing about whether a commit is a fix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

from .data import build_dataset, stratified_split
from .features import extract, feature_names, groups
from .mlp import MLP, metrics_for, roc_auc

LIMITS = [
    "Triage only. The model ranks commits by how much their diff "
    "stats look like past test-touching commits. It does not verify "
    "that a commit does (or does not) touch the tests.",
    "Tiny, skewed data: 63 non-merge commits, ~78% touch tests, "
    "test set 13 rows. Metrics are noisy; accuracy near the 0.78 "
    "majority baseline means the model learned nothing.",
    # Index 2 is replaced at train time with the real label skew.
    "Label skew: see the dataset stats.",
    "Test-ness is never a feature (it would be the label itself). "
    "Test files still count toward the generic frac_py share, since "
    "they are Python files; that correlation is real, not leakage.",
]

VIEWER_START = "<!-- NN-DATA-START -->"
VIEWER_END = "<!-- NN-DATA-END -->"


def permutation_importance(net: MLP, X_test, y_test, seed: int = 42):
    """Grouped permutation importance: AUC drop when a feature group
    is shuffled in the test set. Positive drop = the group matters."""
    base_scores = [net.predict_proba(x) for x in X_test]
    base_auc = roc_auc(base_scores, y_test)
    rng = random.Random(seed)
    out = []
    for group, (a, b) in sorted(groups().items()):
        Xp = [list(x) for x in X_test]
        col_vals = [x[a:b] for x in Xp]
        rng.shuffle(col_vals)
        for row, vals in zip(Xp, col_vals):
            row[a:b] = vals
        auc = roc_auc([net.predict_proba(x) for x in Xp], y_test)
        out.append({"group": group, "auc_drop": round(base_auc - auc, 4)})
    out.sort(key=lambda d: d["auc_drop"], reverse=True)
    return out


def train(repo: str | None, out_dir: str,
          epochs: int = 40, hidden: str = "8,4", lr: float = 0.05,
          seed: int = 42, test_frac: float = 0.2,
          limit: int | None = None) -> dict:
    dataset = build_dataset(repo, limit=limit)
    rows = dataset["rows"]
    stats = dataset["stats"]
    if len(rows) < 20:
        raise ValueError(f"only {len(rows)} training rows; refusing")
    train_rows, test_rows = stratified_split(rows, test_frac, seed)
    X_train = [extract(r) for r in train_rows]
    y_train = [r["label"] for r in train_rows]
    X_test = [extract(r) for r in test_rows]
    y_test = [r["label"] for r in test_rows]

    n_pos = sum(y_train) + sum(y_test)
    n_neg = len(rows) - n_pos
    pos_pct = 100.0 * n_pos / max(1, len(rows))
    limits = list(LIMITS)
    limits[2] = (
        f"Label skew: the training set is {pos_pct:.0f}% touches-tests "
        f"/ {100.0 - pos_pct:.0f}% no-tests "
        f"({n_pos} touches-tests / {n_neg} no-tests of {len(rows)} "
        "commits). Accuracy near the majority baseline means the "
        "model learned nothing; judge it by ROC AUC and per-class "
        "precision/recall.")

    layers = [len(feature_names())] + [int(h) for h in hidden.split(",") if h] + [1]
    net = MLP(layers, seed=seed, lr=lr)
    history = net.fit(X_train, y_train, epochs=epochs, batch_size=32,
                      X_val=X_test, y_val=y_test, seed=seed)
    train_scores = [net.predict_proba(x) for x in X_train]
    test_scores = [net.predict_proba(x) for x in X_test]
    train_metrics = metrics_for(y_train, train_scores)
    test_metrics = metrics_for(y_test, test_scores)
    importance = permutation_importance(net, X_test, y_test, seed)

    result = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "layers": layers,
        "n_features": len(feature_names()),
        "feature_names": feature_names(),
        "dataset": stats,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "feature_importance": importance,
        "limits": limits,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = {
        "format": "bro-nn-weights/1",
        "trained_at": result["trained_at"],
        "mlp": net.to_dict(),
        "metrics": {"train": train_metrics, "test": test_metrics},
        "dataset": stats,
        "limits": limits,
    }
    (out / "weights.json").write_text(json.dumps(weights, indent=2),
                                      encoding="utf-8")
    (out / "viewer-data.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    _refresh_viewer(out / "viewer.html", result)
    return result


def _refresh_viewer(viewer_path: Path, result: dict) -> None:
    """Replace the embedded metrics JSON block in viewer.html so the
    file shows the real numbers even when opened as file://."""
    if not viewer_path.exists():
        return
    text = viewer_path.read_text(encoding="utf-8")
    if VIEWER_START not in text or VIEWER_END not in text:
        return
    block = (VIEWER_START + "\n<script type=\"application/json\" "
             "id=\"nn-data\">\n" + json.dumps(result, indent=2) +
             "\n</script>\n" + VIEWER_END)
    before, _sep, rest = text.partition(VIEWER_START)
    _mid, _sep2, after = rest.partition(VIEWER_END)
    viewer_path.write_text(before + block + after, encoding="utf-8")


def _print_report(result: dict) -> None:
    tm = result["test_metrics"]
    tr = result["train_metrics"]
    print(f"rows: train={result['train_rows']} test={result['test_rows']} "
          f"(touches-tests {result['dataset']['n_positive']} / "
          f"no-tests {result['dataset']['n_negative']})")
    print(f"features: {result['n_features']}  layers: {result['layers']}")
    print(f"train acc {tr['accuracy']:.3f}  test acc {tm['accuracy']:.3f}  "
          f"majority baseline {tm['majority_baseline_accuracy']:.3f}")
    print(f"test ROC AUC {tm['roc_auc']:.3f}")
    print(f"touches-tests: precision {tm['precision_touches_tests']:.3f} "
          f"recall {tm['recall_touches_tests']:.3f}")
    print(f"no-tests: precision {tm['precision_no_tests']:.3f} "
          f"recall {tm['recall_no_tests']:.3f}")
    cm = tm["confusion"]
    print(f"confusion (test): tp={cm['tp']} fp={cm['fp']} "
          f"fn={cm['fn']} tn={cm['tn']}")
    print("feature importance (AUC drop): " + ", ".join(
        f"{d['group']}={d['auc_drop']:.3f}"
        for d in result["feature_importance"][:6]))
    print("LIMITS: triage only; tiny skewed data (63 commits); never "
          "verifies a commit touches the tests.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the bro-agent commit triage network.")
    ap.add_argument("--repo", default=None,
                    help="repo root to read git history from "
                         "(default: this package's repo)")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", default="8,4")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    result = train(args.repo, args.out, epochs=args.epochs,
                   hidden=args.hidden, lr=args.lr, seed=args.seed,
                   test_frac=args.test_frac, limit=args.limit)
    _print_report(result)
    print(f"weights: {os.path.join(args.out, 'weights.json')}")
    print(f"viewer data: {os.path.join(args.out, 'viewer-data.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
