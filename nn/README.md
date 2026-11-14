# bro-agent NN — commit triage

A small pure-Python-stdlib neural network in the style of the
ReconAnything NN scorer (and the monico-agent commit triage network).
Same MLP: tanh hidden layers, sigmoid output, SGD, zero dependencies.

## What it predicts

Whether a commit **touches the tests** (`tests/`), from its diff
stats alone: files changed, insertions/deletions, which file types
and repo areas it touched, subject-line shape (length, word count,
scoped prefix), and author timestamp.

**No test-ness is a feature**: the label is "touches tests/", so a
`frac_test_files` or `touches_tests` feature would be the label
itself. Both are deliberately absent. Test files still count toward
the generic `frac_py` share (they are Python files); that
correlation is real, not leakage.

Why this label: bro-agent's main history is small (71 commits) and
only 5 are fix-prefixed, so a bugfix label would leave ~1 positive
in the test split: untrainable. Test-touching is the only binary
label on this history with both classes in both splits. It is a thin
dataset and the model is weak; the metrics say so plainly.

## Dataset and source

This repo's own `git log --no-merges`: 63 non-merge commits
(50 touches-tests / 13 no-tests). Merge commits excluded (their
diffs double-count the merged branch). Stratified 80/20 train/test
split (50 / 13), seed 42.

## Train / score / view

```bash
python -m nn.train            # writes nn/weights.json + nn/viewer-data.json
python -m nn.score --sha HEAD # score one commit (or a range: HEAD~5..HEAD)
open nn/viewer.html           # metrics + limits, works as file://
```

## Honest limits

- Triage only. The model ranks commits by how much they look like
  past test-touching commits. It never verifies that a commit does
  (or does not) touch the tests.
- Tiny, skewed data: 63 commits, ~79% touch tests, test set 13 rows.
  Metrics are noisy; accuracy near the 0.79 majority baseline means
  the model learned nothing. Judge it by ROC AUC, not accuracy.
- It says nothing about whether a commit is a fix.
- See `nn/viewer-data.json` for the real trained metrics (train/test
  accuracy vs majority baseline, ROC AUC, confusion, per-class
  precision/recall/F1, permutation importance).
