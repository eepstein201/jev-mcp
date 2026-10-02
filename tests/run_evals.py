import sys
import json
import time

sys.path.insert(0, "/Users/ericepstein/Projects/jev-mcp/src")
from jev_mcp.daemon_provider import DaemonProvider
from jev_mcp.kev_provider import KevProvider
from jev_mcp.provider import NoulQuestion

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

def run_benchmark(name, provider):
    print(f"\\n=== Benchmarking {name} ===")
    with open("/Users/ericepstein/Projects/jev-mcp/tests/golden_dataset.json", "r") as f:
        dataset = json.load(f)

    y_true = []
    scores = []

    t0 = time.time()
    for row in dataset:
        y_true.append(1 if row["expected"] else 0)
        q_str = row.get("question_override", "Did the user cancel their subscription?" if row["group_id"] == "subscription_cancel" else "Was the package delivered?")
        q = NoulQuestion(key="eval", prompt=q_str)

        res = provider.evaluate_batch(row["state"], [q])
        prob = res.get("eval", {}).get("probabilities", {}).get("true", 0.0)
        scores.append(prob)
        print(f"[{row['id']}] Expected: {row['expected']}, Prob(True): {prob:.4f}")

    latency = (time.time() - t0) * 1000
    auc = calculate_roc_auc_mann_whitney(y_true, scores)
    correct = sum([1 for y, s in zip(y_true, scores) if (s >= 0.5) == (y == 1)])
    
    print(f"\\nLatency: {latency:.2f} ms ({latency / len(dataset):.2f} ms per sample)")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {correct / len(dataset) * 100:.1f}%")
    return {"latency": latency / len(dataset), "auc": auc, "accuracy": correct / len(dataset) * 100}

if __name__ == "__main__":
    kev_stats = run_benchmark("Kev-0.8B", KevProvider("http://127.0.0.1:8080/v1/systemone"))
    qwen_stats = run_benchmark("Qwen-2.5-0.5B (Legacy)", DaemonProvider("http://127.0.0.1:8082/v1/chat/completions"))
    
    print("\\n=======================================================")
    print("                 BENCHMARK SUMMARY                     ")
    print("=======================================================")
    print(f"{'Metric':<20} | {'Kev 0.8B':<15} | {'Qwen 0.5B':<15}")
    print("-" * 55)
    print(f"{'ROC AUC':<20} | {kev_stats['auc']:<15.4f} | {qwen_stats['auc']:<15.4f}")
    print(f"{'Accuracy':<20} | {kev_stats['accuracy']:<13.1f}% | {qwen_stats['accuracy']:<13.1f}%")
    print(f"{'Latency / Sample':<20} | {kev_stats['latency']:<12.1f} ms | {qwen_stats['latency']:<12.1f} ms")
