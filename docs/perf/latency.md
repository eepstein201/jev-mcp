# Webhook latency measurement

`tests/evals/webhook_latency.py` times each `POST /api/v1/triage/email` request end to end
(client wall clock, `time.perf_counter`) so latency can be tracked over time.

## 1. Start the webhook on a spare port

Use a dedicated port (not the 8080/8081 engines) and keep the router config untouched:

```bash
export JEV_MCP_API_KEY=<your key>          # the harness reads the same variable
JEV_SKIP_MODEL_SYNC=1 \
JEV_MCP_RATE_LIMIT_PER_MINUTE=0 \
python -m jev_mcp.server serve-webhook --port 8097
```

- `JEV_SKIP_MODEL_SYNC=1` stops the startup model sync from rewriting `~/.jev/router_config.json`.
- `JEV_MCP_RATE_LIMIT_PER_MINUTE=0` disables the per-client rate limit; without it a bulk run
  gets 429s and the numbers are meaningless.
- The context id must have labels configured (`~/.jev-mcp/triage_configs.json`), otherwise the
  webhook answers 400 `needs_config`. Pass it with `--context-id`.

## 2. Run the harness

```bash
python -m tests.evals.webhook_latency --url http://127.0.0.1:8097 \
    --dataset PATH.jsonl --context-id <id> [--repeats 3] [--attempt-header script] [--json]
```

Dataset rows are JSON lines with `subject`, `sender`, `body`; other keys are ignored.
The key is sent only as `Authorization: Bearer ...` and is never printed or written.
Connection errors, timeouts and non-JSON bodies are recorded per request; the run never aborts.

`--attempt-header script` mimics `examples/gas_triage.js`: it sends `X-Triage-Attempt` 1..3 and
after each 503 sleeps 5 s (`RETRY_PAUSE_S`) before retrying, adding the pause to that message's
time. That is the delay a user sees for a degraded message. Default `none` sends no header, so a
degraded triage is answered 200 immediately.

### Smart-engine-down variant

Restart the webhook with the smart port pointing at an unused one, e.g.
`JEV_SMART_PORT=8099` (plus the variables above). Every escalated question then fails, so those
requests come back degraded: 503 with `--attempt-header script` on attempts 1-2, 200 on attempt 3.
Compare against the healthy run to see the cost of a smart-engine outage.

## 3. What the numbers mean

All times are seconds. Percentiles use linear interpolation between closest ranks
(`rank = q/100 * (n-1)`, numpy's default), so p50 of `[1,2,3,4]` is 2.5.

- overall: p50/p90/p95/max/mean over every request, including errors (a timeout is a measured wait).
- first request vs rest: the first request includes cold-start effects (model load, caches).
- by `suggested_routing`: the same statistics per routing outcome.
- fast vs escalated: the response does NOT say whether a request was escalated to the smart
  engine. Only a failed escalation is visible (`degraded: true`). The report therefore splits
  degraded vs not degraded; a clean response may be fast-path-only or a successful escalation.
- degraded share, HTTP 503 share (requests whose final answer was 503), and error share.
  With `script`, `503 responses in total` counts every 503 including retried ones.

## 4. Keep the engines idle

Latency depends on queueing in the two llama.cpp engines. Stop other traffic (MCP tool calls,
evals, browsers using the router) for the whole run, and run the healthy and smart-down variants
separately. A warm-up discarded by the harness does not exist on purpose: use the first-request
number to see the cold start, and `rest` for steady state.
