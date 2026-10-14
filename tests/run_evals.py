"""Benchmark the fast (Kev 0.8B) and smart (Qwen 7B) engines on the golden dataset.

Usage:
    make eval                              # both local engines (requires `make start`)
    python tests/run_evals.py --scores FILE NAME
        FILE is a JSON object {row_id: probability_true}; use this to score any
        other model (e.g. a cloud model's answers) with the same metrics.
    python tests/run_evals.py --scores-out scores.jsonl [--seed N]
        Also dump one JSON line per dataset row with both engines' scores.
    python tests/run_evals.py --simulate scores.jsonl [--gates START:STOP:STEP] [--json]
        Replay RoutingProvider's confidence gate over a dump, offline (no inference):
        routed accuracy and escalation rate at each gate.

Ports come from JEV_FAST_PORT (default 8080) and JEV_SMART_PORT (default 8081).
Each local engine first runs untimed warm-up passes so model loading and prior
computation are excluded from the reported latency.
"""

import argparse
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
# Mirrors RoutingProvider (src/jev_mcp/routing_provider.py): a state estimated above
# FAST_ROUTING_TOKEN_LIMIT tokens skips the fast engine; a fast answer whose top
# probability is below ROUTING_CONFIDENCE_GATE is escalated to the smart engine.
FAST_ROUTING_TOKEN_LIMIT = 4000
ROUTING_CONFIDENCE_GATE = 0.85
DEFAULT_GATE_RANGE = "0.5:1.0:0.05"
QUESTION_KEY = "eval"
LEGACY_QUESTIONS = {
    "subscription_cancel": "Did the user cancel their subscription?",
    "delivery_status": "Was the package delivered?",
}


def question_for(row):
    return (
        row.get("question_override")
        or row.get("question")
        or LEGACY_QUESTIONS[row["group_id"]]
    )


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
        "saturated": sum(
            1 for s in scores if s < SATURATION_EPS or s > 1 - SATURATION_EPS
        )
        / n
        * 100,
        "exact_zero": sum(1 for s in scores if s == 0.0),
        "by_kind": {k: sum(v) / len(v) * 100 for k, v in sorted(by_kind.items())},
    }


def print_report(name, m, latency_per_sample=None):
    print(f"\n--- {name} ---")
    print(f"Samples: {m['n']} ({m['n_pos']} positive, {m['n'] - m['n_pos']} negative)")
    print(
        f"ROC AUC: {m['auc']:.4f}  (95% bootstrap CI {m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f})"
    )
    print(f"Accuracy @ {DEFAULT_THRESHOLD}: {m['accuracy']:.1f}%")
    print(
        f"Saturated scores (<{SATURATION_EPS:g} or >1-{SATURATION_EPS:g}): {m['saturated']:.0f}%, exact 0.0: {m['exact_zero']}"
    )
    if latency_per_sample is not None:
        print(f"Latency (warm): {latency_per_sample:.1f} ms per sample")
    print(
        "Accuracy by kind: "
        + ", ".join(f"{k} {v:.0f}%" for k, v in m["by_kind"].items())
    )


def score_dataset_detailed(provider, dataset):
    """Per row: {"score": P(true), "gated": True when the engine gave no usable answer}.

    "gated" mirrors the two cases RoutingProvider escalates without looking at
    confidence: an empty probability map and an all-zero one.
    """
    from jev_mcp.provider import NoulQuestion

    detailed = []
    for row in dataset:
        q = NoulQuestion(key=QUESTION_KEY, prompt=question_for(row))
        res = provider.evaluate_batch(row["state"], [q])
        probs = res.get(QUESTION_KEY, {}).get("probabilities", {})
        detailed.append(
            {
                "score": probs.get("true", 0.0),
                "gated": not probs or all(v == 0.0 for v in probs.values()),
            }
        )
    return detailed


def score_dataset(provider, dataset):
    return [d["score"] for d in score_dataset_detailed(provider, dataset)]


def served_name(provider, role):
    """The model id the engine reported serving, or the role when it reported none."""
    model_id = getattr(provider, "current_model_id", "") or ""
    return role if model_id in ("", "unknown_model") else model_id


def run_benchmark(role, provider, dataset):
    """Warm up, then time one pass. Returns (metrics, latency_ms, detailed, name)."""
    print(f"\n=== Benchmarking {role} ===")
    print(f"Warm-up: {WARMUP_PASSES} untimed pass(es) (loads the model, fills caches)")
    for _ in range(WARMUP_PASSES):
        score_dataset(provider, dataset)

    t0 = time.perf_counter()
    detailed = score_dataset_detailed(provider, dataset)
    latency = (time.perf_counter() - t0) * 1000 / len(dataset)
    scores = [d["score"] for d in detailed]

    if all(s == 0.0 for s in scores):
        print(
            "WARNING: every score is 0.0; the daemon is probably not running (make start)."
        )

    name = served_name(provider, role)
    m = compute_metrics(dataset, scores)
    print_report(name, m, latency)
    return m, latency, detailed, name


def build_score_rows(dataset, fast, smart, fast_model, smart_model, state_tokens):
    """One JSON-ready dict per dataset row, combining both engines' answers."""
    return [
        {
            "index": i,
            "id": row.get("id"),
            "group_id": row.get("group_id"),
            "kind": row.get("kind", "legacy"),
            "question_key": QUESTION_KEY,
            "label": bool(row["expected"]),
            "fast_score": fast[i]["score"],
            "smart_score": smart[i]["score"],
            "fast_gated": fast[i]["gated"],
            "state_tokens": state_tokens[i],
            "fast_model": fast_model,
            "smart_model": smart_model,
        }
        for i, row in enumerate(dataset)
    ]


def write_scores_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_scores_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_escalated(row, gate):
    """RoutingProvider.evaluate_batch's decision for one row (lines 58-111).

    Order mirrors the source: oversized state bypasses the fast engine (L60-67);
    an empty or all-zero fast answer escalates (L76-86); otherwise the top
    probability, max(P(true), P(false)) = max(p, 1 - p), escalates when strictly
    below the gate (L89-94).
    """
    tokens = row.get("state_tokens")
    if tokens is not None and tokens > FAST_ROUTING_TOKEN_LIMIT:
        return True
    if row.get("fast_gated"):
        return True
    p = row["fast_score"]
    return max(p, 1.0 - p) < gate


def simulate_routing(rows, gate=ROUTING_CONFIDENCE_GATE):
    """Routed accuracy and escalation rate at one gate, with no inference."""
    n = len(rows)
    if n == 0:
        return {"routed_accuracy": 0.0, "escalation_rate": 0.0, "n": 0}
    escalated = 0
    correct = 0
    for row in rows:
        use_smart = _is_escalated(row, gate)
        escalated += use_smart
        score = row["smart_score"] if use_smart else row["fast_score"]
        correct += (score >= DEFAULT_THRESHOLD) == bool(row["label"])
    return {
        "routed_accuracy": correct / n,
        "escalation_rate": escalated / n,
        "n": n,
    }


def gate_sweep(rows, gates):
    return [{"gate": g, **simulate_routing(rows, g)} for g in gates]


def parse_gate_range(spec):
    """'START:STOP:STEP' -> inclusive list of gates (counted, so no float drift)."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected START:STOP:STEP, got {spec!r}")
    try:
        start, stop, step = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"expected numbers in START:STOP:STEP, got {spec!r}") from None
    if step <= 0 or start > stop:
        raise ValueError(f"need START <= STOP and STEP > 0, got {spec!r}")
    count = int(round((stop - start) / step + 1e-9))
    return [round(start + i * step, 10) for i in range(count + 1)]


def format_sweep(points):
    lines = [f"{'gate':>6}  {'routed accuracy':>16}  {'escalation rate':>16}  {'n':>5}"]
    for p in points:
        lines.append(
            f"{p['gate']:>6.2f}  {p['routed_accuracy'] * 100:>15.1f}%"
            f"  {p['escalation_rate'] * 100:>15.1f}%  {p['n']:>5}"
        )
    return "\n".join(lines)


def run_simulation(path, gate_spec, as_json):
    rows = read_scores_jsonl(path)
    if not rows:
        sys.exit(f"{path} has no rows")
    sweep = gate_sweep(rows, parse_gate_range(gate_spec))
    if as_json:
        print(json.dumps(sweep))
    else:
        models = {(r.get("fast_model"), r.get("smart_model")) for r in rows}
        for fast_model, smart_model in sorted(models, key=str):
            print(f"fast={fast_model}  smart={smart_model}")
        print(format_sweep(sweep))


def run_legacy_scores(dataset, path, name):
    with open(path, "r") as f:
        by_id = json.load(f)
    missing = [r["id"] for r in dataset if r["id"] not in by_id]
    if missing:
        sys.exit(f"Scores file is missing {len(missing)} ids, e.g. {missing[:3]}")
    scores = [float(by_id[r["id"]]) for r in dataset]
    print_report(name, compute_metrics(dataset, scores))


def run_live(dataset, scores_out):
    from jev_mcp.daemon_provider import DaemonProvider
    from jev_mcp.kev_provider import KevProvider

    fast_port = os.getenv("JEV_FAST_PORT", "8080")
    smart_port = os.getenv("JEV_SMART_PORT", "8081")

    fast = KevProvider(f"http://127.0.0.1:{fast_port}/v1/systemone")
    smart = DaemonProvider(f"http://127.0.0.1:{smart_port}/v1/chat/completions")
    _, _, fast_detail, fast_name = run_benchmark("fast", fast, dataset)
    _, _, smart_detail, smart_name = run_benchmark("smart", smart, dataset)

    if scores_out:
        # The router sizes states with the fast provider's estimate (L54, L60).
        tokens = [fast.check_token_limit(row["state"]) for row in dataset]
        write_scores_jsonl(
            scores_out,
            build_score_rows(
                dataset, fast_detail, smart_detail, fast_name, smart_name, tokens
            ),
        )
        print(f"\nWrote {len(dataset)} score rows to {scores_out}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scores",
        nargs=2,
        metavar=("FILE", "NAME"),
        help="score an external model from a {row_id: probability_true} JSON file",
    )
    parser.add_argument("--scores-out", metavar="PATH", help="write per-row JSONL dump")
    parser.add_argument(
        "--simulate", metavar="SCORES_JSONL", help="replay the routing gate offline"
    )
    parser.add_argument(
        "--gates",
        default=DEFAULT_GATE_RANGE,
        metavar="START:STOP:STEP",
        help=f"gates for --simulate (default {DEFAULT_GATE_RANGE})",
    )
    parser.add_argument("--json", action="store_true", help="--simulate: JSON output")
    parser.add_argument(
        "--seed",
        type=int,
        help="seed Python's global random (the 7B prompt salt) before scoring",
    )
    return parser


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.simulate:
        try:
            parse_gate_range(args.gates)
        except ValueError as exc:
            parser.error(str(exc))
        run_simulation(args.simulate, args.gates, args.json)
        return

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    if args.scores:
        run_legacy_scores(dataset, *args.scores)
        return

    if args.seed is not None:
        random.seed(args.seed)
    run_live(dataset, args.scores_out)


if __name__ == "__main__":
    main(sys.argv[1:])
