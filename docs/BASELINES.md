# Phase-0 engine baseline

Measured 2026-10-14 on the phase-0 engine pair, branch `feat/engine-hardening` @ 580e54d
(host: Mac14,10, Apple M2 Pro, 16 GiB RAM, Darwin 27.2.0; daemons up ~5 h, exclusive lane —
no tests or other workloads ran during measurement).

- fast: `llama-server -m models/Kev-0.8B-GGUF/Kev-0.8B-Q8_0.gguf --port 8080 --parallel 4 -c 65536` (pid 1596)
- smart: `llama-server -m models/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8081 --parallel 2 -c 32768` (pid 1604)

Phase-1 profile swaps are judged against these numbers.

## Memory (tests/evals/memory.py)

Idle = no request sent this session; warm = after one ~12k-token chat request per daemon
(fast 6.0 s, smart 53.0 s — both well inside the 600 s per-daemon timeout).

| signal | fast (Kev-0.8B Q8_0) | smart (Qwen2.5-7B Q4_K_M) |
|---|---|---|
| dirty footprint, idle | 1,055,246,472 B (1006.4 MiB) | 2,033,096,296 B (1938.9 MiB) |
| dirty footprint, warm | 1,168,853,200 B (1114.7 MiB) | 2,097,452,672 B (2000.3 MiB) |
| footprint delta (warm − idle) | +113,606,728 B (+108.3 MiB) | +64,356,376 B (+61.4 MiB) |
| peak footprint, idle / warm | 1,058,801,800 / 1,173,213,000 B | 2,036,373,096 / 2,097,927,808 B |
| GGUF on disk (weights upper bound) | 812,406,304 B (774.8 MiB) | 4,683,074,240 B (4466.1 MiB) |

**Footprint is NOT total model memory.** `footprint` counts dirty plus swapped/compressed
pages and misses clean file-backed pages, so mmap'd GGUF weights and the Metal buffers over
them are missing or only partly counted: the smart engine shows a ~1.9 GiB dirty footprint
against a 4.4 GiB GGUF, and its `mapped file` category row read 0 dirty / 0 clean at idle
(the weight pages are resident somewhere `footprint -f bytes` does not attribute to that
row on this build). Read footprints together with `gguf_bytes`, never alone.

System-wide counters (whole machine, not per-engine; before → after the warm requests):

| signal | idle | warm | delta |
|---|---|---|---|
| swap used (sysctl vm.swapusage) | 7650.19 / 9216 MiB | 11078.25 / 12288 MiB | +3428.06 MiB |
| wired (vm_stat) | 2,949,545,984 B (2812.9 MiB) | 11,787,157,504 B (11241.1 MiB) | +8,837,611,520 B |
| GPU "In use system memory" (ioreg) | 383,205,376 B (365.5 MiB) | 9,235,922,944 B (8808.1 MiB) | +8,852,717,568 B |
| pageouts / swapouts (vm_stat) | 969,704 / 3,780,969 | 970,051 / 4,025,589 | +347 / +244,620 |

The idle column already carries ~7.65 GiB of swap in use from load before this lane ran —
the deltas, not the absolutes, are the baseline signal.

The warm 12k-token prompt eval moved ~8.2 GiB into wired + GPU-attached system memory and
drove ~244k swapouts on this 16 GiB host. Startup buffer sizes (model/KV/compute MiB) were
absent: this llama-server build or verbosity prints no such lines in the daemon logs.

## Scoring and routing (tests/run_evals.py)

Eval set: `tests/golden_dataset.json` (the harness default), 88 rows, 43 positive /
45 negative, 12 groups × 4–8 rows, kinds: 8 legacy + 8 kinds × 10. One untimed warm-up
pass then one timed pass per engine; no `--seed`.

| metric | fast (Kev-0.8B-Q8_0.gguf) | smart (Qwen2.5-7B-Instruct-Q4_K_M.gguf) |
|---|---|---|
| ROC AUC (95% bootstrap CI) | 0.9282 (0.866–0.979) | 1.0000 (1.000–1.000) |
| accuracy @ 0.5 | 89.8% | 96.6% |
| saturated scores | 0% | 50% |
| exact 0.0 | 0 | 0 |
| latency (warm) | 32.9 ms per sample | 488.8 ms per sample |

Routing replay (`--simulate`) at the default gate 0.85 over the 88-row dump:
**routed accuracy 92.0%, escalation rate 51.1%, unanswered 0.** Full sweep for reference:

| gate | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | **0.85** | 0.90 | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| routed accuracy | 89.8% | 90.9% | 92.0% | 90.9% | 90.9% | 93.2% | 92.0% | **92.0%** | 94.3% | 96.6% | 96.6% |
| escalation rate | 0.0% | 4.5% | 10.2% | 17.0% | 19.3% | 31.8% | 42.0% | **51.1%** | 64.8% | 86.4% | 100.0% |

At gate 0.5 nothing escalates, so routed accuracy equals fast-only accuracy (89.8%); the
production gate's escalations buy +2.2 points (92.0%) at the cost of routing 51.1% of rows
to the ~15x slower engine (488.8 vs 32.9 ms per sample), and the pair still trails
smart-only accuracy (96.6%) by 4.6
points — the phase-1 question this baseline anchors.

## Reproducing

```
PYTHONPATH=src .venv/bin/python -m tests.evals.memory --json                # idle
PYTHONPATH=src .venv/bin/python -m tests.evals.memory --json --warm-request  # warm
PYTHONPATH=src .venv/bin/python -m tests.evals.memory --delta IDLE.json WARM.json
PYTHONPATH=src .venv/bin/python tests/run_evals.py --scores-out SCORES.jsonl
PYTHONPATH=src .venv/bin/python tests/run_evals.py --simulate SCORES.jsonl --json
```

Requires the daemons running (`make start`) and a quiet machine; system counters are
whole-host snapshots, so concurrent workloads contaminate them.
