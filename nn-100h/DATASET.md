# bro-agent nn-100h dataset

## Provenance
- Source: this repo's own git history, `git log --no-merges --numstat`
  in the repo root (parent of `nn-100h/`), collected 2026-11-14.
- Rows: 64 non-merge commits
  (50 touch tests/, 14 do not). Merge commits excluded:
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
log_files_changed, log_insertions, log_deletions, log_total_churn, insertion_share, deletion_share, net_insertions, frac_py, frac_js_ts, frac_html_css, frac_docs, frac_config, frac_other, touches_app, touches_notebooks, touches_static, touches_root, subject_len, subject_words, has_scope_prefix, hour_of_day, is_weekend, log_distinct_top_dirs. Diff size, file-type shares, touched areas,
subject shape, author timestamp, distinct top dirs. Same feature set
as PR #7 (`feat/nn-bro-agent`, `nn/features.py`).

## Leakage audit (run by build_dataset.py, 2026-11-14)
- No feature equals the label: none.
- Best single-feature threshold rule: frac_py at 0.39285714285714285
  (direction 1) reaches accuracy 0.9375 on all
  64 rows, vs majority baseline 0.7812. That number is in-sample
  (threshold tuned on the same rows), so it overstates. Honest
  check: the same rule with its threshold tuned on train only
  (frac_py >= 0.25) scores 0.8462 on the frozen test split vs
  0.7692 majority baseline. frac_py is a real but modest signal,
  not a backdoor to the label.
- Verdict: NOT trivially learnable: no feature equals the label and no single-feature threshold rule reaches 1.000. The label is NOT a deterministic
  function of the features, so the task is not trivially learnable
  and no 1.000 accuracy is expected; a 1.000 would mean leakage and
  must be investigated, never claimed as a win.
- Test files still count toward the generic `frac_py` share (they are
  Python files); that correlation is real, not leakage, and is
  disclosed here. No `frac_test_files` / `touches_tests` feature
  exists, because either would BE the label.
- Time-split diagnostic (generalisation check only, not the official
  split): train on the earliest 80% by author_ts (51
  rows), test on the latest 20% (13 rows,
  10 positive, majority baseline 0.7692).
  The official evaluation split stays the frozen stratified split
  in `split.json` (seed 42, test frac 0.2): train 51 rows
  (40 positive), test 13 rows
  (10 positive). NEVER train on test rows.

## Limits
- Tiny, skewed data: 64 commits, ~78%
  touch tests. Test set is 13 rows, so one row swings test
  accuracy by ~8%. Judge by ROC AUC, not accuracy.
- Triage only: the model ranks commits by how much they look like
  past test-touching commits. It never verifies a commit touches tests.
