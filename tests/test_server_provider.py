import json
from unittest.mock import patch

from jev_mcp import server
from jev_mcp.provider import NoulQuestion
from jev_mcp.routing_provider import RoutingProvider


def test_module_level_provider_is_the_hybrid_router():
    # evaluate, optimize-prompt, calibrate and compact all score through this object.
    assert isinstance(server.provider, RoutingProvider)


def test_router_token_limit_is_the_most_restrictive_engine_it_can_route_to():
    # Large states are routed to the smart engine, so its limit must gate QFE compression.
    provider = RoutingProvider()

    assert provider.max_tokens == min(provider.fast_provider.max_tokens, provider.smart_provider.max_tokens)
    assert provider.max_tokens == 8192


def test_routing_provider_evaluate_dataset_returns_expected_and_probabilities():
    # Arrange
    provider = RoutingProvider()
    question = NoulQuestion(key="k", prompt="p?")
    batch_result = {"k": {"probabilities": {"true": 0.7, "false": 0.3}}}

    # Act
    with patch.object(provider, "evaluate_batch", return_value=batch_result) as mock_batch:
        results = provider.evaluate_dataset([{"state": "s1", "expected": True}], question)

    # Assert
    assert results == [{"expected": True, "probabilities": {"true": 0.7, "false": 0.3}}]
    mock_batch.assert_called_once_with("s1", [question])


def test_compact_scores_each_top_level_key_of_a_dict_state_separately():
    # Arrange
    state = {"training": "Run mlx_lm.lora on the base model.", "kitchen": "Buy oat milk on Friday."}
    scores = [{"chunk_0": {"noul": 0.9}}, {"chunk_1": {"noul": 0.1}}]

    # Act
    with patch.object(server.provider.fast_provider, "evaluate_batch", side_effect=scores):
        result = json.loads(
            server.jev_compact_context(state=state, goal="train a model", confidence_threshold=0.5)
        )

    # Assert
    assert result["original_chunks"] == 2
    assert "mlx_lm.lora" in result["compacted_state"]
    assert "oat milk" not in result["compacted_state"]


def test_calibrate_marks_threshold_unusable_when_no_positive_is_ever_predicted():
    # Arrange: every case is confidently auto-rejected, including the real positive.
    question = NoulQuestion(key="k", prompt="p?")
    dataset = [{"state": "a", "expected": True}, {"state": "b", "expected": False}]
    results = [{"probabilities": {"true": 0.0, "false": 0.0}} for _ in dataset]

    # Act
    with patch.object(server.provider, "evaluate_dataset", return_value=results):
        report = json.loads(
            server.jev_calibrate_threshold(dataset=dataset, question=question, apply_platt_scaling="never")
        )["markdown_report"]

    # Assert
    first_row = next(line for line in report.splitlines() if line.startswith("| > 0.50"))
    assert "Unusable" in first_row
    assert "Optimal" not in report
