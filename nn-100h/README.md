# bro-agent nn-100h — 100-hour neural network trainer

Pure-stdlib MLP that ranks bro-agent commits by how likely they are
to touch `tests/`, from diff stats alone. Same model shape as PR #7
(`feat/nn-bro-agent`): tanh hidden layers, sigmoid output, binary
cross-entropy, plain SGD, zero dependencies, no imports from repo
code. This trainer runs the same checkout locally and in GitHub
Actions, accumulating 100 hours of grid + random search across
restarts.

## Quick start (local)

```bash
cd nn-100h
python3 build_dataset.py   # rebuild dataset.jsonl + split.json + DATASET.md from git log
python3 train.py --resume --max-run-hours 5.5   # one chunk; exit 0 on chunk end
python3 train.py --resume --max-run-hours 5.5   # next chunk continues the queue
```

Works fresh without `--resume` (in-progress experiment restarts at
epoch 0). With `--resume` it continues from the latest checkpoint.

## What it predicts

Whether a commit touches `tests/`, from its diff stats alone:
files changed, insertions/deletions, file types, repo areas,
subject-line shape, author timestamp. See `DATASET.md` for the
label definition, provenance and the leakage audit.

## Files in this directory

| file | what it is |
|---|---|
| `build_dataset.py` | builds `dataset.jsonl`, `split.json`, `DATASET.md` from real `git log --no-merges` data; runs the leakage audit |
| `train.py` | the trainer (see contract below) |
| `dataset.jsonl` | 64 real commits, 23 precomputed features each |
| `split.json` | frozen stratified 80/20 split, seed 42 — NEVER train on test rows |
| `DATASET.md` | provenance, label definition, leakage audit |
| `experiments.jsonl` | one line per finished experiment |
| `checkpoints/<exp_id>.json` | every 20 epochs + on chunk exit; deleted when the experiment finishes |
| `best.json` | best experiment so far (weights included) |
| `best-metrics.json` | full metrics of the best experiment |
| `status.json` | hours trained (cumulative), experiments done, best, status |
| `heartbeat.json` | pid, time, exp_id, epoch, experiments done — every <=60s |

## Trainer contract

- CLI: `python3 train.py [--resume] [--max-run-hours H]`, default 5.5.
- Frozen split: `split.json`, seed 42, stratified, test frac 0.2.
  Test rows are scored only, never trained on.
- Experiment loop: phase A is a fixed grid (hidden sizes
  [8],[16],[32],[64],[16,8],[32,16],[64,32] x lr
  0.01/0.03/0.05/0.1/0.2 x L2 0/1e-4/1e-3/1e-2 x epochs
  60/150/300/600 x seeds 42/1/2/3 = 2,240 configs). Phase B is a
  deterministic random search (seeded per experiment index,
  unbounded) so real work continues until 100h. Config for
  experiment N is a pure function of N, so restarts reproduce the
  same queue.
- Each experiment appends one line to `experiments.jsonl`:
  `{exp_id, config, test_acc, auc, confusion {tp,fp,tn,fn},
  per_class_recall, seconds}`. `best.json` updates only when
  honest test accuracy improves (AUC tiebreak).
- Checkpoints every 20 epochs to `checkpoints/<exp_id>.json`
  (`{exp_id, config, epoch, weights, biases, train_loss}`).
- Heartbeat every <=60s; `status.json` hours are updated on every
  heartbeat, so a VM reboot loses at most ~60s of accounting.
- Chunk exit (wall time >= `--max-run-hours`): checkpoint, add
  chunk hours, exit 0. At `hours_trained >= 100`: `status=done`,
  exit 0.

## Leakage audit (summary)

No feature equals the label and no single-feature threshold rule
reaches 1.000 (best is `frac_py`, 0.9375 in-sample; 0.8462 on the
frozen test split when the threshold is tuned on train only, vs
0.7692 majority baseline). The task is not trivially learnable.
A 1.000 accuracy would mean leakage and must be investigated,
never claimed as a win. Full audit in `DATASET.md`.

## Limits

- Tiny, skewed data: 64 commits, ~78% touch tests. Test set is 13
  rows, so one row swings test accuracy by ~8 points. Judge by
  ROC AUC, not accuracy.
- Triage only: the model ranks commits by how much they look like
  past test-touching commits. It never verifies a commit touches tests.

## GitHub Actions

`.github/workflows/nn-100h.yml` runs one chunk per workflow run
(`--resume --max-run-hours 5.5`), commits `nn-100h/` state
(experiments, status, best) back to `feat/nn-100h-bro-agent`, and
is safe to re-run: the queue, hours and best are all derived from
the committed files.
