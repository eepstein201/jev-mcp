# Engine availability sampler

`python -m jev_mcp.availability` measures how often the fast (8080) and smart
(8081) engines are really usable. It reuses the doctor's checks
(`doctor.check_engine`: the `/health` request, then the real scoring probe), so
"usable" means the engine answers a scoring request, not only that the port is open.

## Log schema

Default path `~/.jev/availability.jsonl` (`--log PATH` overrides). Every sample
appends one JSON line per engine:

```json
{"ts": "2026-01-01T12:00:00+00:00", "engine": "fast", "health_ok": true,
 "probe_ok": true, "latency_ms": 212.4, "error": null}
```

| Field | Meaning |
|-------|---------|
| `ts` | ISO-8601 UTC time the sample started |
| `engine` | `fast` or `smart` |
| `health_ok` | `/health` answered 200 (a 503 while loading is false) |
| `probe_ok` | the scoring probe returned a usable answer |
| `latency_ms` | time for the health check plus probe; `null` when the engine was not up or the check raised |
| `error` | short reason (at most 200 characters) when something failed, else `null` |

The sampler never raises on an engine error; a failure is recorded as a row. When the
file exceeds `MAX_LOG_BYTES` (4 MiB) it is renamed to `<path>.1` (one backup, the
previous backup is replaced) and a fresh file starts. `summary` reads the backup too.

## Commands

```
python -m jev_mcp.availability sample [--interval 60] [--log PATH] [--once]
python -m jev_mcp.availability summary [--log PATH] [--since 24h|ISO] [--json]
```

`sample` runs until SIGINT or SIGTERM, then exits 0. `--once` takes one sample.
`summary` prints, per engine over the window: samples, uptime % (`health_ok`), probe
success %, p50 and p95 latency (nearest rank), outage count and longest outage. An
outage is a run of consecutive samples where `health_ok` or `probe_ok` is false; it lasts
from its first failed sample to the next good one (or to its last failed sample if it is
still open). Corrupt or partial log lines are skipped. `--since` takes an ISO timestamp
(naive means UTC) or a duration such as `30m`, `24h`, `7d`.

Each probe is a real scoring request, so a short interval adds load to the engines; the
60 s default is deliberately light.

## Running in the background

Quick: `nohup python -m jev_mcp.availability sample >/dev/null 2>&1 &`.

Example launchd agent (text only; adjust paths, then install it yourself with
`launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jev.availability.plist`):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.jev.availability</string>
  <key>ProgramArguments</key>
  <array>
    <string>/path/to/jev-mcp/.venv/bin/python</string>
    <string>-m</string>
    <string>jev_mcp.availability</string>
    <string>sample</string>
    <string>--interval</string>
    <string>60</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONPATH</key><string>/path/to/jev-mcp/src</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardErrorPath</key><string>/Users/YOU/.jev/availability.err</string>
  <key>StandardOutPath</key><string>/dev/null</string>
</dict>
</plist>
```
