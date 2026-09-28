import sys
import json
import time

sys.path.insert(0, "/Users/ericepstein/Projects/jev-mcp/src")
from jev_mcp.server import provider
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


def run():
    print("Loading Golden Dataset...")
    with open(
        "/Users/ericepstein/Projects/jev-mcp/tests/golden_dataset.json", "r"
    ) as f:
        dataset = json.load(f)

    y_true = []
    scores = []

    t0 = time.time()

    # Evaluate sequentially to simulate batching over distinct questions
    for row in dataset:
        y_true.append(1 if row["expected"] else 0)

        q_str = row.get(
            "question_override",
            "Did the user cancel their subscription?"
            if row["group_id"] == "subscription_cancel"
            else "Was the package delivered?",
        )
        q = NoulQuestion(key="eval", prompt=q_str)

        res = provider.evaluate_batch(row["state"], [q])
        prob = res.get("eval", {}).get("probabilities", {}).get("true", 0.0)
        scores.append(prob)
        print(f"[{row['id']}] Expected: {row['expected']}, Prob(True): {prob:.4f}")

    latency = (time.time() - t0) * 1000

    print("\n=== EVALUATION RESULTS ===")
    print(f"Latency: {latency:.2f} ms ({latency / len(dataset):.2f} ms per sample)")

    auc = calculate_roc_auc_mann_whitney(y_true, scores)
    print(f"ROC AUC (Mann-Whitney): {auc:.4f}")

    # Simple default 0.5 threshold accuracy
    correct = sum([1 for y, s in zip(y_true, scores) if (s >= 0.5) == (y == 1)])
    print(f"Accuracy (@ threshold=0.5): {correct / len(dataset) * 100:.1f}%")


if __name__ == "__main__":
    run()
