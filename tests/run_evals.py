"""Benchmark the fast (Kev 0.8B) and smart (Qwen 7B) engines on the golden dataset.

Usage:
    make eval                              # both local engines (requires `make start`)
    python tests/run_evals.py --scores FILE NAME
        FILE is a JSON object {row_id: probability_true}; use this to score any
        other model (e.g. a cloud model's answers) with the same metrics.

Ports come from JEV_FAST_PORT (default 8080) and JEV_SMART_PORT (default 8081).
Each local engine first runs untimed warm-up passes so model loading and prior
computation are excluded from the reported latency.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATASET_PATH = ROOT / "tests" / "golden_dataset.json"
DEFAULT_THRESHOLD = 0.5
WARMUP_PASSES = 1
BOOTSTRAP_RESAMPLES = 1000
SATURATION_EPS = 1e-6
LEGACY_QUESTIONS = {
    "subscription_cancel": "Did the user cancel their subscription?",
    "delivery_status": "Was the package delivered?",
}


def question_for(row):
    return row.get("question_override") or row.get("question") or LEGACY_QUESTIONS[row["group_id"]]


def calculate_roc_auc_mann_whitney(y_true, scores):
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    r1 = sum(ranks[idx] for idx, label in enumerate(y_true) if label == 1)
    u1 = r1 - (n_pos * (n_pos + 1)) / 2.0
    return u1 / (n_pos * n_neg)


def bootstrap_auc_ci(y_true, scores, resamples=BOOTSTRAP_RESAMPLES, seed=0):
    """95% percentile interval for ROC AUC (seeded, so reruns are comparable)."""
    rng = random.Random(seed)
    n = len(y_true)
    aucs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [y_true[i] for i in idx]
        if 0 < sum(ys) < n:
            aucs.append(calculate_roc_auc_mann_whitney(ys, [scores[i] for i in idx]))
    aucs.sort()
    return aucs[int(0.025 * len(aucs))], aucs[int(0.975 * len(aucs)) - 1]


def compute_metrics(dataset, scores):
    y_true = [1 if row["expected"] else 0 for row in dataset]
    n = len(dataset)
    predicted = [s >= DEFAULT_THRESHOLD for s in scores]
    correct = [p == (y == 1) for p, y in zip(predicted, y_true)]
    by_kind = defaultdict(list)
    for row, ok in zip(dataset, correct):
        by_kind[row.get("kind", "legacy")].append(ok)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, scores)
    return {
        "n": n,
        "n_pos": sum(y_true),
        "auc": calculate_roc_auc_mann_whitney(y_true, scores),
        "auc_ci": (ci_lo, ci_hi),
        "accuracy": sum(correct) / n * 100,
        "saturated": sum(1 for s in scores if s < SATURATION_EPS or s > 1 - SATURATION_EPS) / n * 100,
        "exact_zero": sum(1 for s in scores if s == 0.0),
        "by_kind": {k: sum(v) / len(v) * 100 for k, v in sorted(by_kind.items())},
    }


def print_report(name, m, latency_per_sample=None):
    print(f"\n--- {name} ---")
    print(f"Samples: {m['n']} ({m['n_pos']} positive, {m['n'] - m['n_pos']} negative)")
    print(f"ROC AUC: {m['auc']:.4f}  (95% bootstrap CI {m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f})")
    print(f"Accuracy @ {DEFAULT_THRESHOLD}: {m['accuracy']:.1f}%")
    print(f"Saturated scores (<{SATURATION_EPS:g} or >1-{SATURATION_EPS:g}): {m['saturated']:.0f}%, exact 0.0: {m['exact_zero']}")
    if latency_per_sample is not None:
        print(f"Latency (warm): {latency_per_sample:.1f} ms per sample")
    print("Accuracy by kind: " + ", ".join(f"{k} {v:.0f}%" for k, v in m["by_kind"].items()))


def score_dataset(provider, dataset):
    from jev_mcp.provider import NoulQuestion

    scores = []
    for row in dataset:
        q = NoulQuestion(key="eval", prompt=question_for(row))
        res = provider.evaluate_batch(row["state"], [q])
        scores.append(res.get("eval", {}).get("probabilities", {}).get("true", 0.0))
    return scores


def run_benchmark(name, provider, dataset):
    print(f"\n=== Benchmarking {name} ===")
    print(f"Warm-up: {WARMUP_PASSES} untimed pass(es) (loads the model, fills caches)")
    for _ in range(WARMUP_PASSES):
        score_dataset(provider, dataset)

    t0 = time.perf_counter()
    scores = score_dataset(provider, dataset)
    latency = (time.perf_counter() - t0) * 1000 / len(dataset)

    if all(s == 0.0 for s in scores):
        print("WARNING: every score is 0.0; the daemon is probably not running (make start).")

    m = compute_metrics(dataset, scores)
    print_report(name, m, latency)
    return m, latency


def main(argv):
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    if len(argv) >= 3 and argv[0] == "--scores":
        with open(argv[1], "r") as f:
            by_id = json.load(f)
        missing = [r["id"] for r in dataset if r["id"] not in by_id]
        if missing:
            sys.exit(f"Scores file is missing {len(missing)} ids, e.g. {missing[:3]}")
        scores = [float(by_id[r["id"]]) for r in dataset]
        print_report(argv[2], compute_metrics(dataset, scores))
        return

    from jev_mcp.daemon_provider import DaemonProvider
    from jev_mcp.kev_provider import KevProvider

    fast_port = os.getenv("JEV_FAST_PORT", "8080")
    smart_port = os.getenv("JEV_SMART_PORT", "8081")

    run_benchmark("Kev-0.8B", KevProvider(f"http://127.0.0.1:{fast_port}/v1/systemone"), dataset)
    run_benchmark(
        "Qwen-2.5-7B", DaemonProvider(f"http://127.0.0.1:{smart_port}/v1/chat/completions"), dataset
    )


if __name__ == "__main__":
    main(sys.argv[1:])
