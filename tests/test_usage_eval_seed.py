"""The seeded HOME the tool-usage evaluation runs against."""

import json
import time

from tests.evals.usage import seed_home


def test_the_seeded_home_holds_a_known_config_a_fresh_catalog_and_the_fixture(tmp_path):
    # Act
    home = seed_home.seed(tmp_path / "home")

    # Assert
    config = json.loads((home / ".jev" / "router_config.json").read_text())
    assert config["buckets"]["b3"] == "fixture-large"
    assert config["fitted_temperature"] == 1.7
    assert config["auto_update"] is False
    catalog = json.loads((home / ".jev" / "model_catalog.json").read_text())
    assert catalog["providers"] == {}
    assert (
        time.time() - catalog["fetched_at"] < 60
    )  # fresh, so the run never fetches it
    assert "Priya" in (home / "fixture" / "notes" / "registry.txt").read_text()


def test_seeding_replaces_whatever_was_in_the_home_before(tmp_path):
    # Arrange
    home = tmp_path / "home"
    home.mkdir()
    (home / "stale.txt").write_text("left over")

    # Act
    seed_home.seed(home)

    # Assert
    assert not (home / "stale.txt").exists()


def test_the_default_home_is_a_fixed_path_outside_the_users_home():
    assert str(seed_home.HOME) == "/tmp/jev-usage-eval-home"
