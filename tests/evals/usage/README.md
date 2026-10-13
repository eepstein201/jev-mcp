# Tool-usage evaluation

Ten read-only questions that a model can answer only by choosing and calling the right jev tools.
It measures whether the tool names, descriptions and parameters are good enough to be used
correctly. It is not part of `make check`: it calls the Anthropic API and needs the daemons running.

## Run it

```bash
# once: a venv outside the repo (the runner needs `anthropic` and an `mcp` 1.x client)
uv venv ~/.jev/usage-eval-venv
uv pip install --python ~/.jev/usage-eval-venv/bin/python 'anthropic>=0.39' 'mcp>=1.1,<2'

export ANTHROPIC_API_KEY=...        # read from the environment, never written anywhere
make eval-usage USAGE_EVAL_RUNNER=/path/to/mcp-builder/scripts/evaluation.py
```

`USAGE_EVAL_MODEL` picks the model (default `claude-sonnet-5-5`). A weaker model is a stricter
test of the descriptions.

## What it does

- `seed_home.py` builds `/tmp/jev-usage-eval-home`: a known router config (buckets `b1`-`b4`, one
  rule, temperature 1.7), a fresh model-catalog cache so nothing reaches the network, and a copy of
  `fixture/`. The server runs with that as `HOME`; your real `~/.jev` is never read or written.
- `run_eval.py` loads the mcp-builder runner and replaces its agent loop with `loop.py`, which
  1. answers **every** tool call in a turn (the runner's own loop answers only the first, so any
     parallel call is an API 400),
  2. sends MCP content blocks back as their text (the runner's JSON step cannot serialise them), and
  3. offers the model only the 9 read-only tools plus `jev_manage_router_config` and
     `jev_manage_temperature` (the questions need their read actions; against the seeded home they
     can only write there). A model is never offered a tool that launches a process, drives a
     browser or swaps model files. `tests/test_usage_eval_loop.py` keeps the list in step with the
     annotations table.
- The server command is the `jev-mcp` console script: the runner's `-a -m jev_mcp.server` is read by
  argparse as the runner's own `--model` flag.

## Questions that were dropped, and why

- Anything that depends on `jev_determine_best_model`: it answered `fixture-medium` and then
  `fixture-small` for the same trivial task, so its answer is not stable enough to check by string.
- "How many functions are defined across the inventory package?" It was answered correctly, but
  only by inference from a chunk count, after 145 tool calls and 8 minutes. No jev tool can list a
  directory, and the relevance gate withholds content for structural queries, so the question
  needed a capability that does not exist. See `docs/TODO.md`.

## Results

| Date | Model | Accuracy | Tool calls | Notes |
|---|---|---|---|---|
| 2026-10-13 | claude-sonnet-5-5 | 10/10 | 161 | all tools offered; one question took 145 calls |
| 2026-10-13 | claude-sonnet-5-5 | 10/10 | 19 | read-only tools only; descriptions say content can be withheld |
