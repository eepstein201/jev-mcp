import io
import json
import os
import time
from unittest.mock import patch

import pytest

from jev_mcp import model_catalog


def _model(family, released, cost, **extra):
    return {
        "family": family,
        "release_date": released,
        "cost": {"input": cost},
        "tool_call": True,
        "modalities": {"output": ["text"]},
        **extra,
    }


RAW = {
    "anthropic": {
        "models": {
            "claude-haiku-4-5": _model("claude-haiku", "2025-10-15", 1),
            "claude-haiku-5-5": _model("claude-haiku", "2026-10-07", 0.1),
            "claude-sonnet-5": _model("claude-sonnet", "2026-06-29", 2),
            "claude-sonnet-5-5": _model("claude-sonnet", "2026-09-28", 2),
            "claude-opus-5-5": _model("claude-opus", "2026-09-22", 4),
            "claude-opus-5-5-20260922": _model("claude-opus", "2026-09-22", 4),
            "claude-fable-5-1": _model("claude-fable", "2026-09-01", 10),
        }
    },
    "google": {
        "models": {
            "gemini-3.8-flash": _model("gemini-flash", "2026-09-02", 0.75),
            "gemini-3.5-flash-lite": _model("gemini-flash-lite", "2026-07-21", 0.3),
            "gemini-3-pro": _model("gemini-pro", "2026-02-01", 2),
            "deep-research-max-preview-04-2026": _model("gemini-pro", "2026-04-21", 2),
            "gemini-embed": _model("gemini-embedding", "2026-09-30", 0.1, tool_call=False),
            "gemini-flash-voice": _model("gemini-flash", "2026-10-01", 0.75, modalities={"output": ["text", "audio"]}),
        }
    },
    "some-reseller": {"models": {"claude-haiku-9": _model("claude-haiku", "2027-01-01", 0.1)}},
}


@pytest.fixture
def catalog():
    return model_catalog.slim_catalog(RAW)


def _response(payload):
    return io.BytesIO(json.dumps(payload).encode())


# --- reducing the public catalog to what the router needs ---


def test_only_first_party_providers_are_kept(catalog):
    assert set(catalog) == {"anthropic", "google"}


def test_previews_dated_snapshots_and_non_tool_models_are_dropped(catalog):
    google_ids = {m["id"] for m in catalog["google"]}
    anthropic_ids = {m["id"] for m in catalog["anthropic"]}

    assert "deep-research-max-preview-04-2026" not in google_ids
    assert "gemini-embed" not in google_ids
    assert "gemini-flash-voice" not in google_ids  # speaks as well as writes: not a drop-in chat model
    assert "claude-opus-5-5-20260922" not in anthropic_ids
    assert "claude-opus-5-5" in anthropic_ids


# --- which family a configured model belongs to ---


def test_a_catalogued_model_reports_its_family_and_release_date(catalog):
    assert model_catalog.family_of(catalog, "claude-sonnet-5") == ("anthropic", "claude-sonnet", "2026-06-29")


def test_a_retired_model_is_matched_to_a_family_by_its_name(catalog):
    # Not in the catalog any more, but the name still says which tier it was.
    assert model_catalog.family_of(catalog, "claude-3-5-haiku") == ("anthropic", "claude-haiku", None)
    assert model_catalog.family_of(catalog, "claude-fable") == ("anthropic", "claude-fable", None)


def test_the_most_specific_family_wins_for_a_retired_model(catalog):
    assert model_catalog.family_of(catalog, "gemini-2.0-flash-lite")[1] == "gemini-flash-lite"


def test_an_unrecognised_model_has_no_family(catalog):
    assert model_catalog.family_of(catalog, "my-local-llama") is None


# --- suggesting newer models ---


def test_newer_models_are_suggested_in_the_same_family_only(catalog):
    buckets = {"b1": "claude-3-5-haiku", "b2": "claude-sonnet-5", "b3": "claude-opus-5-5", "b4": "my-local-llama"}

    updates = model_catalog.suggest_updates(buckets, catalog)

    assert updates == {
        "b1": {"current": "claude-3-5-haiku", "newer": "claude-haiku-5-5", "released": "2026-10-07"},
        "b2": {"current": "claude-sonnet-5", "newer": "claude-sonnet-5-5", "released": "2026-09-28"},
    }


def test_a_preview_is_never_suggested_over_a_released_model(catalog):
    assert model_catalog.suggest_updates({"b3": "gemini-3-pro"}, catalog) == {}


def test_nothing_is_suggested_without_a_catalog():
    assert model_catalog.suggest_updates({"b1": "claude-3-5-haiku"}, {}) == {}


# --- default buckets for a new install ---


def test_default_buckets_are_the_newest_model_of_each_tier_ordered_by_price(catalog):
    assert model_catalog.default_buckets(catalog) == {
        "b1": "claude-haiku-5-5",
        "b2": "claude-sonnet-5-5",
        "b3": "claude-opus-5-5",
        "b4": "claude-fable-5-1",
    }


def test_no_default_buckets_when_the_catalog_has_fewer_than_four_tiers():
    small = model_catalog.slim_catalog({"anthropic": {"models": {"claude-haiku-5-5": _model("claude-haiku", "2026-10-07", 0.1)}}})

    assert model_catalog.default_buckets(small) is None
    assert model_catalog.default_buckets({}) is None


# --- fetching and caching ---


def test_refresh_downloads_the_catalog_and_caches_the_reduced_form():
    with patch("urllib.request.urlopen", return_value=_response(RAW)) as urlopen:
        assert model_catalog.refresh() is True

    request = urlopen.call_args[0][0]
    assert request.full_url == model_catalog.CATALOG_URL
    # The catalog host answers 403 to urllib's default user agent.
    assert request.get_header("User-agent") == model_catalog.USER_AGENT
    cached = model_catalog.load_cached()
    assert set(cached) == {"anthropic", "google"}
    assert "some-reseller" not in open(model_catalog.CACHE_PATH).read()


def test_refresh_does_not_download_again_while_the_cache_is_fresh():
    with patch("urllib.request.urlopen", return_value=_response(RAW)):
        model_catalog.refresh()

    with patch("urllib.request.urlopen") as urlopen:
        assert model_catalog.refresh() is True

    urlopen.assert_not_called()


def test_refresh_downloads_again_once_the_cache_is_older_than_a_day():
    with patch("urllib.request.urlopen", return_value=_response(RAW)):
        model_catalog.refresh()
    tomorrow = time.time() + model_catalog.CACHE_TTL_SECONDS + 60

    with patch("time.time", return_value=tomorrow), patch("urllib.request.urlopen", return_value=_response(RAW)) as urlopen:
        model_catalog.refresh()

    urlopen.assert_called_once()


def test_a_failed_download_keeps_the_previous_cache():
    with patch("urllib.request.urlopen", return_value=_response(RAW)):
        model_catalog.refresh()

    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert model_catalog.refresh(force=True) is False

    assert "anthropic" in model_catalog.load_cached()


def test_a_missing_or_corrupt_cache_reads_as_empty():
    assert model_catalog.load_cached() == {}

    os.makedirs(os.path.dirname(model_catalog.CACHE_PATH), exist_ok=True)
    with open(model_catalog.CACHE_PATH, "w") as f:
        f.write("{not json")

    assert model_catalog.load_cached() == {}
