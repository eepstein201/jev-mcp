# Triage accuracy metrics

`tests/evals/triage_eval.py` can score the triage `requires_action` score against binary
labels, so engine profiles and prompts can be compared with accuracy numbers instead of
routing agreement alone. The maths lives in `tests/evals/metrics.py` (pure Python, no numpy or
sklearn). The eval measures; it does not gate.

## Metrics

A prediction is positive when `score >= threshold`.

- **Precision**: of the rows predicted positive, the share that are truly positive. High
  precision means few false alarms.
- **Recall**: of the truly positive rows, the share predicted positive. High recall means few
  missed actions.
- **F1**: harmonic mean of precision and recall. With no predicted positives, or no actual
  positives, these score 0.0 rather than raising.
- **ROC-AUC**: the chance a random positive outscores a random negative (0.5 is chance, 1.0 is
  a perfect ranking). It ignores the threshold and judges only the ordering. Computed from
  average ranks, so a tied positive/negative pair counts one half. It is `undefined` (None in
  JSON) when all labels are one class.
- **95% CI for AUC**: a percentile bootstrap (1000 resamples of the rows, fixed seed 0, so
  reruns agree). Also `undefined` for a single class.
- **Average precision (PR-AUC)**: the area under the precision-recall step curve, one step per
  distinct score. It rewards putting positives at the top and is more informative than AUC
  when positives are rare. It is `undefined` with no positives.
- **Threshold sweep**: precision, recall and F1 at 0.05, 0.10 ... 0.95. `best_f1` is the row
  with the highest F1 (the lowest threshold wins a tie). Choosing a threshold on the same
  rows you report is optimistic; treat it as a pointer, not a result.
- **Routing**: precision, recall and F1 of the decision actually made, where the positive
  prediction is `suggested_routing == escalate_to_slack`. It reflects the real gate, not just
  the raw score.

## Running it on the 40-row slice

Both daemons must be running. The labels file holds `n`, `subject` and `label` (`act` or
`not_act`); the slice file holds the email text and an empty `expected_routing`. Join them on
`n` into a scratch file outside the repo, and give `expected_routing` any valid value, since
the loader requires one. Routing *agreement* from that file is meaningless; read the binary
block.

```
SLICE=~/Projects/jev-p1m-archive/engine-profiles
jq -c --slurpfile L $SLICE/human_slice_labels_final.jsonl \
  '(reduce $L[] as $r ({}; .[$r.n|tostring] = $r.label)) as $m
   | .label = $m[.n|tostring] | .expected_routing = "archive_and_label"' \
  $SLICE/human_slice_for_labelling.jsonl > "$TMPDIR/slice_joined.jsonl"

python -m tests.evals.triage_eval --dataset "$TMPDIR/slice_joined.jsonl" \
  --binary-label-key label --positive act          # text report
python -m tests.evals.triage_eval --dataset "$TMPDIR/slice_joined.jsonl" --json \
  --binary-label-key label --positive act          # adds a "binary" object with the sweep
```

Without the two flags the output is exactly what it was before. Pass both or neither.

Text output adds:

```
binary  n=40  positives=<k>
auc  0.xxx  95% CI [lo, hi]
average_precision  0.xxx
best_f1  threshold=0.xx  precision=...  recall=...  f1=...
routing  precision=...  recall=...  f1=...
```

## Reading the numbers at n=40

With 40 rows the estimates are noisy. A bootstrap interval for AUC typically spans 0.15 to
0.25, so two profiles whose AUCs differ by 0.05 are not distinguishable. One flipped row moves
precision or recall by several points when positives number about a dozen. Compare profiles
by whether the intervals overlap, and only treat a gap as real when it is clearly larger than
the interval width. The best-F1 threshold is picked on the same 40 rows and will look better
than it would on new mail. Use this slice to catch large regressions and to rank gross
differences, and collect more labels before tuning a threshold.
