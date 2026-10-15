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

# Phase-1 profile comparison

Measured 2026-10-14 to 2026-10-15 on branch `feat/engine-profiles` @ b0e50a8, same host and
exclusive-lane rules as phase 0. Candidates were started with `make start PROFILE=<name>`
and judged against the phase-0 default (Kev-0.8B + Qwen2.5-7B) above.

| profile | fast | smart |
|---|---|---|
| default | Kev-0.8B Q8_0 | Qwen2.5-7B Q4_K_M |
| lean | Kev-0.8B Q8_0 | Qwen3.5-4B Q4_K_M |
| balanced | Kev-4B Q4_K_M | Qwen3.5-4B Q4_K_M |

## Verdicts

A profile changes the shipped default only if (1) the paired-bootstrap 95% lower bound of
(profile − phase-0) routed accuracy exceeds −2.0 points, (2) synthetic-30 triage is within
1 row of the default's 24/30, and (3) dirty footprint is strictly lower. Verdict 2,
"promising but unproven", needs the point estimate ≥ −2.0 points plus (2) and (3), with
only the CI bound failing. Anything else is "not recommended".

Accuracy is routed accuracy on the 88-row golden set at a gate matching phase-0's 51.1%
escalation; the difference is against the phase-0 dump at gate 0.85. Values are r1 / r2 / r3.

| profile | routed accuracy (diff) | 95% CI of diff | McNemar p | triage /30 | dirty idle footprint | verdict |
|---|---|---|---|---|---|---|
| lean (gate 0.85) | 94.3 / 95.5 / 93.2% (+2.3 / +3.4 / +1.1) | [−2.3, +6.8] / [0.0, +8.0] / [−3.4, +5.7] | 0.63 / 0.25 / 1.0 | 22, 21 (fail) | 2.36 GB vs 3.09 GB (−24%) | **NOT RECOMMENDED** |
| balanced (gate 0.962) | 100.0 / 98.9 / 100.0% (+8.0 / +6.8 / +8.0) | [+3.4, +13.6] / [+1.1, +12.5] / [+3.4, +13.6] | 0.016 / 0.070 / 0.016 | 20, 22 (fail) | 3.83 GB vs 3.09 GB (+24%) | **NOT RECOMMENDED** |

No profile reaches verdict 1, so the default is unchanged. At n=88 one row is about 1.1
points and the CI half-width is several points (about 4.5 here), so verdict 1 needs a
larger labelled set (about 500 rows).

- **Lean** is not worse on accuracy (point estimate +1.1 to +3.4; the CI lower bound is
  below −2.0 on 2 of 3 repeats, so this is inconclusive at n=88) and uses less memory, but
  it fails triage (22 and 21 vs 24 ± 1), which keeps it out of verdict 2.
- **Balanced** is clearly more accurate (every repeat's CI lower bound is above 0), but
  fails triage (20 and 22) and uses more memory (+0.75 GB idle).

## Details

Routing replay (`--simulate`), 3 repeats per profile:

| | fast AUC | smart AUC | fast acc @ 0.5 | gate / escalation | routed accuracy |
|---|---|---|---|---|---|
| default | 0.9282 | 1.0000 | 89.8% | 0.85 / 51.1% | 92.0% |
| lean | 0.9282 | 1.0000 | 89.8% | 0.85 / 51.1% | 94.3 / 95.5 / 93.2% |
| balanced | 0.9948 | 1.0000 | 98.9% | 0.962 / 51.1% | 100.0 / 98.9 / 100.0% |

- Balanced has no 0.05-grid gate within ±2 points of 51.1% escalation (0.95 → 42.0%,
  1.00 → 100%); 0.962 was found with a 0.001 sweep. At the shipped gate 0.85 it escalates
  12.5% of rows at 98.9% routed accuracy.
- Both candidates run Qwen3.5-4B with reasoning off. Each profile's proof (empty
  `<think></think>` from `/apply-template`, true+false token mass median 0.988, min ≥ 0.955,
  0 of 88 rows under 0.05) is in the P1-M artifacts. Platt scaling was not triggered
  (>0.99 saturation 40.9% lean, 39.8% balanced), so no temperature was fitted or saved.
- Latency per sample: smart 0.5 s (Qwen2.5-7B) vs 1.6–1.7 s lean and 2.3–2.8 s balanced
  (Qwen3.5-4B, 3–5× slower); fast 33 ms (Kev-0.8B) vs 143 ms (Kev-4B).

Memory (dirty footprint, fresh daemons, ~20 s after health, same `-c` totals as phase 0):

| | fast idle | smart idle | idle total | warm total | swap delta (warm) |
|---|---|---|---|---|---|
| default | 1,055,246,472 B | 2,033,096,296 B | 3,088,342,768 B | 3,266,305,872 B | +3428 MiB (phase-0 12k prompt) |
| lean | 1,034,193,008 B | 1,322,356,600 B | 2,356,549,608 B | 2,630,670,480 B | −8 MiB |
| balanced | 2,509,509,456 B | 1,324,224,352 B | 3,833,733,808 B | 4,193,428,336 B | +47 MiB |

Triage (`make eval-triage`, branch code): synthetic-30 default 24, 24; lean 22, 21;
balanced 20, 22. Informational seeds (default not re-measured): lean 52/120 and 50/120,
balanced 41/120 and 53/120.

Browser agent (`make eval-browser`, 4 runs per case, informational, outside the decision
rule): default 31/32 and 30/32 (one run hit a harness CDP timeout), balanced 32/32,
lean 26/32 (T3 0/4). With n=4 per case none of this attributes a difference to a model.

## Caveats

- The three repeats re-score the same 88 rows, so they are not independent samples; the
  decision is made per repeat, and no pooled p-value is reported.
- The 24/30 triage anchor was labelled against the phase-0 engines, and the default's
  runs were measured with main's `jev_mcp` (the shared venv); `triage_eval` needs
  `PYTHONPATH=src` to import branch code. Lean's main-code runs (20, 21) fail the same
  rule. A 40-row human-labelled slice is pending and can separate model regression from
  anchor bias.
- Memory is footprint, not total model memory (see the phase-0 note above). Compare it
  alongside GGUF size: Kev-4B Q4_K_M is 3,033,489,824 B on disk.
- Candidates were measured at their profile-pinned gate; every profile ships with the 0.85
  gate, so balanced's matched-gate accuracy is not what `PROFILE=balanced` serves today.

## Reproducing

```
make start PROFILE=lean            # or balanced; restore with make start PROFILE=default
PYTHONPATH=src .venv/bin/python tests/run_evals.py --scores-out SCORES.jsonl
PYTHONPATH=src .venv/bin/python tests/run_evals.py --simulate SCORES.jsonl --json
PYTHONPATH=src .venv/bin/python -m tests.evals.triage_eval
PYTHONPATH=src .venv/bin/python -m tests.evals.memory --json
```
