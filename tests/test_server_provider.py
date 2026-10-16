import json
from unittest.mock import patch

import pytest

from jev_mcp import server
from jev_mcp.provider import NoulQuestion
from jev_mcp.routing_provider import RoutingProvider

# Calibration needs the smart engine to have reported a model.
pytestmark = pytest.mark.usefixtures("smart_model")


def test_module_level_provider_is_the_hybrid_router():
    # jev_evaluate_batch, jev_optimize_prompt, jev_calibrate_threshold and jev_compact_context all score through this object.
    assert isinstance(server.provider, RoutingProvider)


def test_router_token_limit_is_the_most_restrictive_engine_it_can_route_to():
    # Large states are routed to the smart engine, so its limit must gate QFE compression.
    provider = RoutingProvider()

    assert provider.max_tokens == min(
        provider.fast_provider.max_tokens, provider.smart_provider.max_tokens
    )
    assert provider.max_tokens == 8192


def test_routing_provider_evaluate_dataset_returns_expected_and_probabilities():
    # Arrange
    provider = RoutingProvider()
    question = NoulQuestion(key="k", prompt="p?")
    batch_result = {"k": {"probabilities": {"true": 0.7, "false": 0.3}}}

    # Act
    with patch.object(
        provider, "evaluate_batch", return_value=batch_result
    ) as mock_batch:
        results = provider.evaluate_dataset(
            [{"state": "s1", "expected": True}], question
        )

    # Assert
    assert results == [{"expected": True, "probabilities": {"true": 0.7, "false": 0.3}}]
    mock_batch.assert_called_once_with("s1", [question])


def test_compact_scores_each_top_level_key_of_a_dict_state_separately():
    # Arrange
    state = {
        "training": "Run mlx_lm.lora on the base model.",
        "kitchen": "Buy oat milk on Friday.",
    }
    scores = [{"chunk_0": {"noul": 0.9}}, {"chunk_1": {"noul": 0.1}}]

    # Act
    with patch.object(
        server.provider.fast_provider, "evaluate_batch", side_effect=scores
    ):
        result = json.loads(
            server.jev_compact_context(
                state=state, goal="train a model", confidence_threshold=0.5
            )
        )

    # Assert
    assert result["original_chunks"] == 2
    assert "mlx_lm.lora" in result["compacted_state"]
    assert "oat milk" not in result["compacted_state"]
    # Callers can see how close each dropped chunk was to the threshold.
    assert result["chunk_scores"] == [0.9, 0.1]


def test_compact_default_threshold_keeps_nearly_all_relevant_chunks_on_the_labelled_set():
    # On the labelled goal set (tests/evals/compaction_goals.jsonl) the tune split kept 95% of the
    # relevant chunks at 0.2 and 81% at the earlier 0.3; see docs/compaction-threshold.md.
    import inspect

    default = (
        inspect.signature(server.jev_compact_context)
        .parameters["confidence_threshold"]
        .default
    )

    assert default.default == 0.2


def test_compact_asks_whether_a_chunk_is_directly_relevant_to_the_goal():
    # Measured on 24 labelled pairs: "directly relevant" separated relevant from irrelevant
    # chunks cleanly (margin +0.06), while the earlier "strictly relevant" overlapped (-0.03).
    with patch.object(
        server.provider.fast_provider,
        "evaluate_batch",
        return_value={"chunk_0": {"noul": 0.9}},
    ) as mock_batch:
        server.jev_compact_context(
            state={"text": "some context"},
            goal="ship the release",
            confidence_threshold=0.3,
        )

    prompt = mock_batch.call_args[0][1][0].prompt
    assert "directly relevant" in prompt
    assert "ship the release" in prompt


def test_compact_keeps_a_chunk_and_records_no_score_when_scoring_fails():
    # Losing context because the engine errored would be worse than keeping it.
    with patch.object(
        server.provider.fast_provider,
        "evaluate_batch",
        side_effect=RuntimeError("engine down"),
    ):
        result = json.loads(
            server.jev_compact_context(
                state={"text": "important context"}, goal="g", confidence_threshold=0.5
            )
        )

    assert result["kept_chunks"] == 1
    assert result["compacted_state"] == "important context"
    assert result["chunk_scores"] == [None]


def test_evaluate_dataset_uses_a_rows_own_question_when_it_has_one():
    # Arrange
    provider = RoutingProvider()
    question = NoulQuestion(key="k", prompt="default?")
    dataset = [
        {"state": "s1", "expected": True, "question": "row specific?"},
        {"state": "s2", "expected": False},
    ]

    # Act
    with patch.object(
        provider, "evaluate_batch", return_value={"k": {"probabilities": {"true": 1.0}}}
    ) as mock_batch:
        provider.evaluate_dataset(dataset, question)

    # Assert
    first_question = mock_batch.call_args_list[0][0][1][0]
    second_question = mock_batch.call_args_list[1][0][1][0]
    assert (first_question.key, first_question.prompt) == ("k", "row specific?")
    assert second_question.prompt == "default?"
    assert question.prompt == "default?"  # the caller's question is not mutated


def _overconfident_results(n_pos, n_neg):
    sure_true = {"probabilities": {"true": 0.999, "false": 0.001}}
    sure_false = {"probabilities": {"true": 0.001, "false": 0.999}}
    wrong_true = {"probabilities": {"true": 0.97, "false": 0.03}}
    wrong_false = {"probabilities": {"true": 0.02, "false": 0.98}}
    dataset = [{"state": "x", "expected": True}] * n_pos + [
        {"state": "x", "expected": False}
    ] * n_neg
    results = [dict(sure_true) for _ in range(n_pos - 1)] + [dict(wrong_false)]
    results += [dict(sure_false) for _ in range(n_neg - 1)] + [dict(wrong_true)]
    return dataset, results


def _calibrate_always(dataset, results):
    question = NoulQuestion(key="k", prompt="p?")
    with patch.object(
        server.provider.smart_provider, "evaluate_dataset", return_value=results
    ):
        return json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset, question=question, apply_platt_scaling="always"
            )
        )["markdown_report"]


def test_calibrate_cross_validates_platt_scaling_when_each_class_has_enough_rows():
    dataset, results = _overconfident_results(n_pos=10, n_neg=10)

    report = _calibrate_always(dataset, results)

    assert "Platt Scaling Applied" in report
    assert "cross-validated" in report


def test_calibrate_says_when_platt_scaling_was_fitted_and_scored_on_the_same_rows():
    dataset, results = _overconfident_results(n_pos=2, n_neg=2)

    report = _calibrate_always(dataset, results)

    assert "Platt Scaling Applied" in report
    assert "same rows" in report


def _calibrate_json(tmp_path, monkeypatch, existing_config=None, **kwargs):
    """Run calibrate on overconfident scores against a throwaway router config."""
    from jev_mcp import router_config

    config_path = tmp_path / "router_config.json"
    if existing_config is not None:
        config_path.write_text(json.dumps(existing_config))
    monkeypatch.setattr(router_config, "ROUTER_CONFIG_PATH", str(config_path))
    dataset, results = _overconfident_results(n_pos=10, n_neg=10)
    kwargs.setdefault("apply_platt_scaling", "always")
    with patch.object(
        server.provider.smart_provider, "evaluate_dataset", return_value=results
    ):
        out = json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset, question=NoulQuestion(key="k", prompt="p?"), **kwargs
            )
        )
    return out, config_path


def test_calibrate_returns_a_fitted_temperature_above_one_for_overconfident_scores(
    tmp_path, monkeypatch
):
    out, _ = _calibrate_json(tmp_path, monkeypatch)

    assert out["fitted_temperature"] > 1.0
    assert out["temperature_saved"] is False


def test_calibrate_does_not_write_the_router_config_unless_asked(tmp_path, monkeypatch):
    # The temperature is global: fitting it on one question must not silently change every other one.
    _, config_path = _calibrate_json(tmp_path, monkeypatch)

    assert not config_path.exists()


def test_calibrate_saves_the_fitted_temperature_when_asked_and_keeps_other_settings(
    tmp_path, monkeypatch
):
    existing = {
        "buckets": {"b1": "some-model"},
        "rules": [{"condition": "c", "bucket": "b1"}],
    }

    out, config_path = _calibrate_json(
        tmp_path, monkeypatch, existing_config=existing, save_to_config=True
    )

    saved = json.loads(config_path.read_text())
    assert out["temperature_saved"] is True
    entry = saved["temperatures"][out["model"]]
    assert entry["temperature"] == out["fitted_temperature"]
    assert entry["updated_at"] == saved["temperature_updated_at"]
    assert saved["buckets"] == existing["buckets"]
    assert saved["rules"] == existing["rules"]


def test_calibrate_fitted_temperature_composes_with_the_one_already_applied(
    tmp_path, monkeypatch
):
    # Scores were produced under the current temperature, so the new one multiplies it.
    base, _ = _calibrate_json(tmp_path, monkeypatch)
    scaled, _ = _calibrate_json(
        tmp_path, monkeypatch, existing_config={"fitted_temperature": 2.0}
    )

    assert scaled["fitted_temperature"] == pytest.approx(
        2.0 * base["fitted_temperature"], rel=1e-3
    )


def test_calibrate_reports_no_temperature_and_saves_nothing_without_platt_scaling(
    tmp_path, monkeypatch
):
    out, config_path = _calibrate_json(
        tmp_path, monkeypatch, apply_platt_scaling="never", save_to_config=True
    )

    assert out["fitted_temperature"] is None
    assert out["temperature_saved"] is False
    assert not config_path.exists()


def test_calibrate_marks_threshold_unusable_when_no_positive_is_ever_predicted():
    # Arrange: every case is confidently auto-rejected, including the real positive.
    question = NoulQuestion(key="k", prompt="p?")
    dataset = [{"state": "a", "expected": True}, {"state": "b", "expected": False}]
    results = [{"probabilities": {"true": 0.0, "false": 0.0}} for _ in dataset]

    # Act
    with patch.object(
        server.provider.smart_provider, "evaluate_dataset", return_value=results
    ):
        report = json.loads(
            server.jev_calibrate_threshold(
                dataset=dataset, question=question, apply_platt_scaling="never"
            )
        )["markdown_report"]

    # Assert
    first_row = next(
        line for line in report.splitlines() if line.startswith("| > 0.50")
    )
    assert "Unusable" in first_row
    assert "Optimal" not in report
