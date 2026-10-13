"""Build the fixed, seeded HOME the tool-usage evaluation runs against.

Creates /tmp/jev-usage-eval-home with a known router config, a fresh model-catalog cache (so the
run never reaches the network), and a copy of the fixture directory. Prints the HOME path.
Run the server with HOME set to that path; the user's real ~/.jev is never read or written.
"""

import json
import shutil
import time
from pathlib import Path

HOME = Path("/tmp/jev-usage-eval-home")
FIXTURE = Path(__file__).parent / "fixture"

ROUTER_CONFIG = {
    "buckets": {
        "b1": "fixture-small",
        "b2": "fixture-medium",
        "b3": "fixture-large",
        "b4": "fixture-xl",
    },
    "rules": [{"condition": "Touches the database schema", "target": "b4"}],
    "buckets_source": "user",
    "auto_update": False,
    "fitted_temperature": 1.7,
    "temperature_updated_at": "2026-01-01 00:00:00",
}


def seed(home: Path = HOME) -> Path:
    if home.exists():
        shutil.rmtree(home)
    (home / ".jev").mkdir(parents=True)
    (home / ".jev" / "router_config.json").write_text(
        json.dumps(ROUTER_CONFIG, indent=2)
    )
    (home / ".jev" / "model_catalog.json").write_text(
        json.dumps({"fetched_at": time.time(), "providers": {}})
    )
    shutil.copytree(FIXTURE, home / "fixture")
    return home


if __name__ == "__main__":  # pragma: no cover
    print(seed())
