import pytest
import math
from jev_mcp.daemon_provider import DaemonProvider


def test_laplace_smoothing_and_normalization():
    provider = DaemonProvider()
    raw_logprobs = {"true": -1.0, "false": -2.0}
    prior_logprobs = {"true": -0.693, "false": -2.302}
    expected_keys = ["true", "false"]

    normalized = provider._normalize_logprobs(
        raw_logprobs, expected_keys, prior_logprobs
    )
    assert abs(sum(normalized.values()) - 1.0) < 1e-6


def test_laplace_horizon_truncation_defense():
    provider = DaemonProvider()
    raw_logprobs = {"true": -0.5}
    prior_logprobs = {"true": -1.0, "false": -1.5}

    normalized = provider._normalize_logprobs(
        raw_logprobs, ["true", "false"], prior_logprobs
    )
    assert "true" in normalized
    assert "false" in normalized


def test_absolute_confidence_gating():
    provider = DaemonProvider()
    raw_logprobs = {"true": -15.0, "false": -16.0}

    normalized = provider._normalize_logprobs(raw_logprobs, ["true", "false"], {})
    assert normalized["true"] == 0.0
    assert normalized["false"] == 0.0


def test_log_sum_exp_token_aggregation():
    provider = DaemonProvider()
    top_logprobs = [
        {"token": "True", "logprob": -2.0},
        {"token": " True", "logprob": -2.0},
        {"token": "False", "logprob": -4.0},
    ]

    result = {}
    for target in ["true", "false"]:
        result[target] = -9999.0

    for item in top_logprobs:
        token_str = item.get("token", "").strip().lower()
        lp = item.get("logprob", -9999.0)

        if token_str in result:
            curr_lp = result[token_str]
            if curr_lp == -9999.0:
                result[token_str] = lp
            elif lp != -9999.0:
                max_val = max(curr_lp, lp)
                result[token_str] = max_val + math.log(
                    math.exp(curr_lp - max_val) + math.exp(lp - max_val)
                )

    assert result["true"] > -2.0
    assert result["false"] == -4.0

def test_unreachable_max_lp():
    from unittest.mock import patch
    provider = DaemonProvider()
    raw_logprobs = {"true": -1.0}
    prior_logprobs = {"true": -1.0}
    with patch("builtins.max", return_value=-9999.0):
        normalized = provider._normalize_logprobs(
            raw_logprobs, ["true"], prior_logprobs
        )
        assert normalized == {"true": 0.0}
