# Choosing the `jev_compact_context` default threshold

The default `confidence_threshold` is 0.2 (it was 0.3). A chunk is kept when the fast engine's
relevance score is at or above the threshold. A dropped relevant chunk costs more than a kept
irrelevant one, so the default is chosen for recall first and compression second.

## Method

- Data: `tests/evals/compaction_goals.jsonl`, 36 labelled rows (319 chunks) across three goal
  types, split into tune (24 rows, 214 chunks) and holdout (12 rows, 105 chunks).
- Each chunk is scored once with the tool's generic question; thresholds 0.05 to 0.70 in steps of
  0.05 are swept offline (`python -m tests.evals.compaction_eval`, needs the fast daemon).
- Rule, applied to the tune split only with all goal types pooled: the threshold with the most
  compression subject to recall >= 0.95, ties going to the higher threshold. Holdout was looked
  at only after the value was frozen.

## Result (pooled)

| Threshold | Tune recall | Tune compression | Holdout recall | Holdout compression |
|-----------|-------------|------------------|----------------|---------------------|
| 0.30 (old) | 0.81 | 0.46 | 0.67 | 0.70 |
| 0.20 (new) | 0.95 | 0.27 | 0.94 | 0.43 |

## Caveats

- The holdout is 12 rows and 105 chunks (36 relevant), so a single chunk moves holdout recall by
  about 3 points. Tune recall at 0.20 sits exactly on the 0.95 limit; the neighbouring grid points
  differ by several points either way.
- The labels were written by Claude, not by the repository owner, and the contexts are
  hand-written rather than real session state. Re-measure on real content before treating 0.2 as
  more than a better-supported default than 0.3.
- Callers that pass `confidence_threshold` explicitly (for example email triage, which uses 0.3)
  are unchanged.
