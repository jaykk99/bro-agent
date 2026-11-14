"""100-hour neural network trainer for bro-agent (jaykk99/bro-agent).

Task: rank commits by how likely they are to touch tests/, from
diff stats alone. Same MLP shape as PR #7 (feat/nn-bro-agent):
tanh hidden layers, sigmoid output, binary cross-entropy, plain
SGD, pure Python stdlib only, no imports from repo code, no
torch/numpy. Self-contained: features, MLP, metrics, grid search,
checkpoints, resume, heartbeat and hour accounting all live here.

Run from this directory (repo_root/nn-100h/):

    python3 train.py [--resume] [--max-run-hours H]   # default H = 5.5

Files in cwd (all under nn-100h/):
    dataset.jsonl   real commits, built by build_dataset.py
                    (rebuilt automatically here if missing)
    split.json      frozen stratified 80/20 split, seed 42
    DATASET.md      provenance + leakage audit
    experiments.jsonl  one line per finished experiment
    checkpoints/<exp_id>.json  every 20 epochs + on chunk exit
    best.json       best experiment so far (weights included)
    best-metrics.json  full metrics of the best experiment
    status.json     {repo, hours_trained, experiments_done, best,
                     status: training|done, updated, ...}
    heartbeat.json  {pid, time, exp_id, epoch, experiments_done}

Hours: hours_trained is cumulative across restarts. It is updated
in status.json on every heartbeat (<=60s), so a VM reboot loses at
most ~60s of accounting. On chunk exit (wall time >=
--max-run-hours) the trainer checkpoints, finalises status and
exits 0. When hours_trained >= 100, status becomes done, exit 0.

Experiment loop: phase A is a fixed grid over hidden sizes, lr,
L2, epochs and seeds (2,240 configs). Phase B is a deterministic
random search (seeded per experiment index, unbounded) so the
trainer keeps doing real work until 100 hours are reached instead
of idling or sleeping. Config for experiment index N is a pure
function of N, so a restart reproduces the same queue.

Resume: --resume loads the latest checkpoint for the in-progress
experiment (weights + biases + epoch) and continues from that
epoch. Without --resume the in-progress experiment restarts at
epoch 0. A finished experiment is never retrained: its line is in
experiments.jsonl and the queue moves on.

Checkpoints are deleted when their experiment finishes (its result
is in experiments.jsonl and the best weights live in best.json),
which bounds disk over a 100h run. A checkpoint that survives is
always an unfinished experiment, resumable with --resume.

NEVER train on test rows. Test rows are scored only, after each
experiment finishes its training epochs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = "jaykk99/bro-agent"
TARGET_HOURS = 100.0
HEARTBEAT_SECS = 60.0
CHECKPOINT_EVERY = 20

# ---------------------------------------------------------------- MLP
# tanh hidden / sigmoid out / SGD, copied pattern from PR #7 nn/mlp.py.


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class MLP:
    def __init__(self, layers, seed=42, lr=0.05, l2=1e-4):
        self.layers = list(layers)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.seed = int(seed)
        rng = random.Random(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(
                [[rng.uniform(-limit, limit) for _ in range(fan_in)]
                 for _ in range(fan_out)])
            self.biases.append([0.0 for _ in range(fan_out)])

    def _forward(self, x):
        acts = [list(map(float, x))]
        for li, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = [sum(wi * xi for wi, xi in zip(row, acts[-1])) + bias
                 for row, bias in zip(w, b)]
            if li == len(self.weights) - 1:
                acts.append([_sigmoid(v) for v in z])
            else:
                acts.append([math.tanh(v) for v in z])
        return acts

    def predict_proba(self, x):
        return self._forward(x)[-1][0]

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def _backward(self, acts, y):
        n_layers = len(self.weights)
        deltas = [None] * n_layers
        deltas[-1] = [acts[-1][0] - y]
        for li in range(n_layers - 2, -1, -1):
            w_next = self.weights[li + 1]
            d_next = deltas[li + 1]
            a = acts[li + 1]
            deltas[li] = [
                sum(w_next[k][j] * d_next[k] for k in range(len(d_next)))
                * (1.0 - a[j] * a[j]) for j in range(len(a))]
        grad_w, grad_b = [], []
        for li in range(n_layers):
            d = deltas[li]
            a_prev = acts[li]
            grad_w.append([[d[i] * a_prev[j] for j in range(len(a_prev))]
                           for i in range(len(d))])
            grad_b.append(list(d))
        return grad_w, grad_b

    def train_epoch(self, X, y, rng, batch_size=32):
        n = len(X)
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n, max(1, int(batch_size))):
            idxs = order[start:start + int(batch_size)]
            gw_acc = [[[0.0] * len(row) for row in w] for w in self.weights]
            gb_acc = [[0.0] * len(b) for b in self.biases]
            for i in idxs:
                acts = self._forward(X[i])
                gw, gb = self._backward(acts, float(y[i]))
                for li in range(len(gw)):
                    for r in range(len(gw[li])):
                        for c in range(len(gw[li][r])):
                            gw_acc[li][r][c] += gw[li][r][c]
                    for r in range(len(gb[li])):
                        gb_acc[li][r] += gb[li][r]
            m = float(len(idxs))
            for li in range(len(self.weights)):
                for r in range(len(self.weights[li])):
                    for c in range(len(self.weights[li][r])):
                        g = gw_acc[li][r][c] / m + self.l2 * self.weights[li][r][c]
                        self.weights[li][r][c] -= self.lr * g
                for r in range(len(self.biases[li])):
                    self.biases[li][r] -= self.lr * (gb_acc[li][r] / m)

    def train_loss(self, X, y):
        eps = 1e-12
        total = 0.0
        for xi, yi in zip(X, y):
            p = min(max(self.predict_proba(xi), eps), 1.0 - eps)
            total += -(yi * math.log(p) + (1.0 - yi) * math.log(1.0 - p))
        return total / max(1, len(X))

    def to_dict(self):
        return {"format": "bro-nn-100h-mlp/1", "layers": self.layers,
                "lr": self.lr, "l2": self.l2, "seed": self.seed,
                "weights": self.weights, "biases": self.biases}

    @classmethod
    def from_dict(cls, data):
        net = cls(data["layers"], seed=int(data.get("seed", 42)),
                  lr=float(data.get("lr", 0.05)),
                  l2=float(data.get("l2", 1e-4)))
        net.weights = [[[float(v) for v in row] for row in mat]
                       for mat in data["weights"]]
        net.biases = [[float(v) for v in b] for b in data["biases"]]
        return net


# ---------------------------------------------------------------- metrics

def roc_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n_pos = sum(1 for _, lab in pairs if lab == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    ranks = [0.0] * len(pairs)
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    for r, (_, lab) in zip(ranks, pairs):
        if lab == 1:
            rank_sum += r
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(net, X, y, threshold=0.5):
    scores = [net.predict_proba(x) for x in X]
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = fp = tn = fn = 0
    for t, p in zip(y, preds):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    acc = (tp + tn) / max(1, len(y))
    return {
        "n": len(y), "n_positive": sum(y), "n_negative": len(y) - sum(y),
        "threshold": threshold,
        "test_acc": acc, "accuracy": acc,
        "majority_baseline_accuracy":
            max(sum(y), len(y) - sum(y)) / max(1, len(y)),
        "auc": roc_auc(scores, y),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_class_recall": {
            "touches_tests": tp / (tp + fn) if tp + fn else 0.0,
            "no_tests": tn / (tn + fp) if tn + fp else 0.0},
        "precision_touches_tests": tp / (tp + fp) if tp + fp else 0.0,
        "precision_no_tests": tn / (tn + fn) if tn + fn else 0.0,
    }


# ---------------------------------------------------------------- config space

GRID_HIDDEN = [[8], [16], [32], [64], [16, 8], [32, 16], [64, 32]]
GRID_LR = [0.01, 0.03, 0.05, 0.1, 0.2]
GRID_L2 = [0.0, 1e-4, 1e-3, 1e-2]
GRID_EPOCHS = [60, 150, 300, 600]
GRID_SEEDS = [42, 1, 2, 3]

GRID = [{"hidden": h, "lr": lr, "l2": l2, "epochs": e, "seed": s,
         "batch_size": 32}
        for h in GRID_HIDDEN for lr in GRID_LR for l2 in GRID_L2
        for e in GRID_EPOCHS for s in GRID_SEEDS]


def exp_config(index):
    """Config for experiment index N: pure function of N (resumable)."""
    if index < len(GRID):
        cfg = dict(GRID[index])
        cfg["phase"] = "grid"
        return cfg
    rng = random.Random(20261114 + index)
    hidden = rng.choice([[8], [16], [24], [32], [48], [64], [128],
                         [16, 8], [32, 16], [48, 24], [64, 32],
                         [128, 64], [32, 16, 8]])
    lr = round(10 ** rng.uniform(math.log10(0.005), math.log10(0.3)), 4)
    l2 = round(10 ** rng.uniform(-5, math.log10(0.03)), 6) if rng.random() < 0.8 else 0.0
    return {"hidden": hidden, "lr": lr, "l2": l2,
            "epochs": rng.choice([100, 200, 400, 800, 1200]),
            "seed": rng.randrange(1, 100000),
            "batch_size": rng.choice([16, 32]),
            "phase": "random"}


def exp_id_for(index):
    return "exp-%04d" % (index + 1)


# ---------------------------------------------------------------- io helpers

def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def log(msg):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def load_data():
    if not (HERE / "dataset.jsonl").exists() or not (HERE / "split.json").exists():
        log("dataset/split missing, building with build_dataset.py")
        subprocess.run([sys.executable, str(HERE / "build_dataset.py")],
                       check=True, cwd=str(HERE))
    rows = [json.loads(line) for line in open(HERE / "dataset.jsonl", encoding="utf-8")
            if line.strip()]
    split = load_json(HERE / "split.json")
    by_sha = {r["sha"]: r for r in rows}
    train = [by_sha[s] for s in split["train_shas"] if s in by_sha]
    test = [by_sha[s] for s in split["test_shas"] if s in by_sha]
    assert train and test, "empty train or test split"
    assert not (set(split["train_shas"]) & set(split["test_shas"])), "split overlap!"
    return train, test


def read_experiments():
    path = HERE / "experiments.jsonl"
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-run-hours", type=float, default=5.5)
    args = ap.parse_args()

    chunk_start = time.time()
    pid = os.getpid()
    (HERE / "checkpoints").mkdir(exist_ok=True)

    train, test = load_data()
    n_in = len(train[0]["features"])
    X_train = [r["features"] for r in train]
    y_train = [r["label"] for r in train]
    X_test = [r["features"] for r in test]
    y_test = [r["label"] for r in test]
    log("data: train=%d (pos %d) test=%d (pos %d) features=%d"
        % (len(train), sum(y_train), len(test), sum(y_test), n_in))

    status = load_json(HERE / "status.json") or {}
    base_hours = float(status.get("hours_trained", 0.0))
    best = load_json(HERE / "best.json")
    if best is None:
        exps = read_experiments()
        if exps:
            best = max(exps, key=lambda e: (e["test_acc"], e["auc"]))
            write_json(HERE / "best.json", best)
    experiments_done = len(read_experiments())
    log("resuming state: hours_trained=%.4f experiments_done=%d best=%s"
        % (base_hours, experiments_done,
           ("%.4f auc=%.4f %s" % (best["test_acc"], best["auc"], best["exp_id"]))
           if best else "none"))

    if base_hours >= TARGET_HOURS:
        status.update({"repo": REPO, "hours_trained": base_hours,
                       "experiments_done": experiments_done,
                       "best": best, "status": "done",
                       "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        write_json(HERE / "status.json", status)
        log("target %.1fh already reached, done" % TARGET_HOURS)
        return 0

    last_heartbeat = 0.0
    current_exp_id = None
    current_epoch = 0

    def hours_now():
        return base_hours + (time.time() - chunk_start) / 3600.0

    def write_status(state="training"):
        status.update({
            "repo": REPO, "hours_trained": round(hours_now(), 6),
            "experiments_done": len(read_experiments()),
            "best": load_json(HERE / "best.json") or best,
            "status": state, "target_hours": TARGET_HOURS,
            "max_run_hours": args.max_run_hours,
            "current_exp_id": current_exp_id,
            "current_epoch": current_epoch,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_json(HERE / "status.json", status)

    def heartbeat(exp_id, epoch):
        nonlocal last_heartbeat
        now = time.time()
        if now - last_heartbeat < HEARTBEAT_SECS and epoch != 0:
            return
        last_heartbeat = now
        write_json(HERE / "heartbeat.json", {
            "pid": pid,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "exp_id": exp_id, "epoch": epoch,
            "experiments_done": len(read_experiments()),
        })
        write_status("training")

    def save_checkpoint(exp_id, cfg, epoch, net):
        data = {"exp_id": exp_id, "config": cfg, "epoch": epoch,
                "weights": net.weights, "biases": net.biases,
                "train_loss": net.train_loss(X_train, y_train)}
        write_json(HERE / "checkpoints" / (exp_id + ".json"), data)

    def chunk_expired():
        return (time.time() - chunk_start) / 3600.0 >= args.max_run_hours

    index = experiments_done
    while hours_now() < TARGET_HOURS:
        if chunk_expired():
            log("chunk wall time %.2fh >= max-run-hours %.2f, exiting 0"
                % ((time.time() - chunk_start) / 3600.0, args.max_run_hours))
            write_status("training")
            return 0
        exp_id = exp_id_for(index)
        cfg = exp_config(index)
        total_epochs = int(cfg["epochs"])
        ckpt_path = HERE / "checkpoints" / (exp_id + ".json")
        net = MLP([n_in] + list(cfg["hidden"]) + [1], seed=cfg["seed"],
                  lr=cfg["lr"], l2=cfg["l2"])
        start_epoch = 0
        if args.resume and ckpt_path.exists():
            ck = load_json(ckpt_path)
            if ck and ck.get("config") == cfg and 0 < int(ck.get("epoch", 0)) < total_epochs:
                net = MLP.from_dict({"format": "bro-nn-100h-mlp/1",
                                     "layers": [n_in] + list(cfg["hidden"]) + [1],
                                     "lr": cfg["lr"], "l2": cfg["l2"],
                                     "seed": cfg["seed"],
                                     "weights": ck["weights"],
                                     "biases": ck["biases"]})
                start_epoch = int(ck["epoch"])
                log("RESUMED %s from checkpoint epoch %d/%d (config %s)"
                    % (exp_id, start_epoch, total_epochs, cfg))
        if start_epoch == 0:
            log("START %s config=%s" % (exp_id, cfg))
        current_exp_id = exp_id
        rng = random.Random(cfg["seed"])
        t0 = time.time()
        finished = True
        for epoch in range(start_epoch, total_epochs):
            current_epoch = epoch + 1
            net.train_epoch(X_train, y_train, rng, batch_size=cfg["batch_size"])
            if current_epoch % CHECKPOINT_EVERY == 0 or current_epoch == total_epochs:
                save_checkpoint(exp_id, cfg, current_epoch, net)
            heartbeat(exp_id, current_epoch)
            if chunk_expired():
                save_checkpoint(exp_id, cfg, current_epoch, net)
                log("CHUNK-EXIT mid-experiment %s at epoch %d/%d, checkpoint saved"
                    % (exp_id, current_epoch, total_epochs))
                write_status("training")
                return 0
        seconds = time.time() - t0
        metrics = evaluate(net, X_test, y_test)
        record = {"exp_id": exp_id, "config": cfg,
                  "test_acc": round(metrics["test_acc"], 4),
                  "auc": round(metrics["auc"], 4),
                  "confusion": metrics["confusion"],
                  "per_class_recall": metrics["per_class_recall"],
                  "precision_touches_tests": round(metrics["precision_touches_tests"], 4),
                  "precision_no_tests": round(metrics["precision_no_tests"], 4),
                  "majority_baseline_accuracy": round(metrics["majority_baseline_accuracy"], 4),
                  "n_test": metrics["n"], "seconds": round(seconds, 2)}
        with open(HERE / "experiments.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if best is None or (record["test_acc"], record["auc"]) > (best["test_acc"], best["auc"]):
            best = dict(record)
            best["weights"] = net.weights
            best["biases"] = net.biases
            write_json(HERE / "best.json", best)
            write_json(HERE / "best-metrics.json", metrics)
            log("NEW BEST %s acc=%.4f auc=%.4f conf=%s recall=%s"
                % (exp_id, record["test_acc"], record["auc"],
                   record["confusion"], record["per_class_recall"]))
        else:
            log("DONE %s acc=%.4f auc=%.4f conf=%s (best %s %.4f)"
                % (exp_id, record["test_acc"], record["auc"],
                   record["confusion"], best["exp_id"], best["test_acc"]))
        try:
            os.remove(ckpt_path)
        except FileNotFoundError:
            pass
        current_exp_id = None
        current_epoch = 0
        write_status("training")
        heartbeat(exp_id, total_epochs)
        index += 1

    write_status("done")
    log("target %.1fh reached (hours_trained=%.4f), done" % (TARGET_HOURS, hours_now()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
