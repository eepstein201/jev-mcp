"""Error-branch coverage for jev_mcp.tools.files."""

import json
from unittest.mock import MagicMock, patch

from jev_mcp import server


def test_read_file_reports_missing_file():
    res = json.loads(server.jev_read_file("/nonexistent/file.py", "any task"))
    assert res["status"] == "ERROR"
    assert "not found" in res["message"]


@patch("jev_mcp.routing_provider.RoutingProvider.evaluate_batch")
def test_read_file_blocks_when_no_chunk_is_relevant(mock_eval, tmp_path):
    mock_eval.return_value = {"chunk_0": {"probabilities": {"true": 0.05}}}
    f = tmp_path / "mod.py"
    f.write_text("def relevant():\n    return 1\n")
    res = json.loads(
        server.jev_read_file(str(f), "find the thing", filter_by_chunk=True)
    )
    assert res["status"] == "BLOCKED"
    assert "0 relevant chunks" in res["message"]


def test_compact_rejects_empty_state():
    res = json.loads(server.jev_compact_context(state={"text": "   "}, goal="g"))
    assert res["status"] == "ERROR"
    assert "empty" in res["message"]


def _echo_scores(state, questions):
    return {q.key: {"noul": 0.9} for q in questions}


def test_compact_keeps_text_without_blank_lines_as_one_chunk():
    # No blank line, so the whole text is a single chunk (kept via the fast tier).
    with patch("jev_mcp.app.provider") as mock_provider:
        mock_provider.fast_provider.evaluate_batch.side_effect = _echo_scores
        res = json.loads(
            server.jev_compact_context(
                state={"text": "alpha\nbeta"}, goal="g", confidence_threshold=0.3
            )
        )
    assert res["status"] == "COMPACTION_COMPLETE"
    assert res["original_chunks"] == 1
    assert res["kept_chunks"] == 1


def test_compact_uses_plain_provider_when_no_fast_tier():
    with patch(
        "jev_mcp.app.provider", MagicMock(spec=["evaluate_batch"])
    ) as mock_provider:
        mock_provider.evaluate_batch.side_effect = _echo_scores
        res = json.loads(
            server.jev_compact_context(
                state={"text": "alpha\n\nbeta"}, goal="g", confidence_threshold=0.3
            )
        )
    assert res["status"] == "COMPACTION_COMPLETE"
    assert res["kept_chunks"] == 2


@patch("jev_mcp.routing_provider.RoutingProvider.evaluate_batch")
def test_scan_repo_falls_back_when_pointing_selects_nothing(mock_eval, tmp_path):
    mock_eval.return_value = {"chunk_0": {"probabilities": {"true": 0.9}}}
    for i in range(25):
        (tmp_path / f"mod{i}.py").write_text(f"def f{i}():\n    return {i}\n")
    with patch("jev_mcp.app._chat_completion", MagicMock(return_value="")):
        res = json.loads(server.jev_scan_repo(str(tmp_path), "find f0"))
    assert res["status"] == "SUCCESS"


@patch("jev_mcp.app._chat_completion", side_effect=RuntimeError("smart tier down"))
@patch("jev_mcp.routing_provider.RoutingProvider.evaluate_batch")
def test_scan_repo_falls_back_when_pointing_raises(mock_eval, mock_chat, tmp_path):
    mock_eval.return_value = {"chunk_0": {"probabilities": {"true": 0.9}}}
    for i in range(25):
        (tmp_path / f"mod{i}.py").write_text(f"def f{i}():\n    return {i}\n")
    res = json.loads(server.jev_scan_repo(str(tmp_path), "find f0"))
    assert res["status"] == "SUCCESS"


def _all_relevant(state, questions):
    return {q.key: {"probabilities": {"true": 0.9}} for q in questions}


def _engine_down(state, questions):
    from jev_mcp.provider import EngineUnavailableError

    raise EngineUnavailableError("No engine answered: chunk_0")


@patch(
    "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", side_effect=_engine_down
)
def test_read_file_reports_an_unavailable_engine_instead_of_blocking(
    mock_eval, tmp_path
):
    f = tmp_path / "mod.py"
    f.write_text("def relevant():\n    return 1\n")

    chunked = json.loads(
        server.jev_read_file(str(f), "find the thing", filter_by_chunk=True)
    )
    whole = json.loads(
        server.jev_read_file(str(f), "find the thing", filter_by_chunk=False)
    )

    assert chunked["status"] == "ERROR" and "No engine answered" in chunked["message"]
    assert whole["status"] == "ERROR" and "No engine answered" in whole["message"]


@patch(
    "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", side_effect=_engine_down
)
def test_scan_repo_reports_an_unavailable_engine_instead_of_blocking(
    mock_eval, tmp_path
):
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")

    res = json.loads(server.jev_scan_repo(str(tmp_path), "find f"))

    assert res["status"] == "ERROR"
    assert "No engine answered" in res["message"]


@patch(
    "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", side_effect=_all_relevant
)
def test_scan_repo_reads_the_files_the_smart_tier_lists_one_per_line(
    mock_eval, tmp_path
):
    for i in range(25):
        (tmp_path / f"mod{i}.py").write_text(f"def f{i}():\n    return {i}\n")

    with patch(
        "jev_mcp.app._chat_completion", MagicMock(return_value="mod3.py\n- mod7.py")
    ) as mock_chat:
        res = json.loads(server.jev_scan_repo(str(tmp_path), "find f3"))

    assert "scanned 2 files" in res["message"]
    assert "### FILE: mod3.py\ndef f3" in res["content"]
    assert "\\n" not in res["content"]
    assert "\\n" not in mock_chat.call_args[0][0]


def test_compact_says_so_when_no_engine_scored_any_chunk():
    with patch("jev_mcp.app.provider") as mock_provider:
        mock_provider.fast_provider.evaluate_batch.return_value = {}  # daemon down: no answers
        res = json.loads(
            server.jev_compact_context(state={"text": "alpha\n\nbeta"}, goal="g")
        )

    assert res["status"] == "ENGINE_UNAVAILABLE"
    assert res["compacted_state"] == "alpha\n\nbeta"
    assert res["chunk_scores"] == [None, None]


def test_compact_keeps_a_long_chunk_verbatim():
    text = "\n".join(f"        row_{i} = load(  {i}  )" for i in range(200))
    with patch("jev_mcp.app.provider") as mock_provider:
        mock_provider.fast_provider.evaluate_batch.side_effect = _echo_scores
        res = json.loads(
            server.jev_compact_context(
                state={"text": text}, goal="g", confidence_threshold=0.3
            )
        )
    assert res["status"] == "COMPACTION_COMPLETE"
    assert (
        "        row_1 = load(  1  )\n        row_2 = load(  2  )"
        in res["compacted_state"]
    )


def test_compact_accepts_a_text_field_that_is_not_a_string():
    with patch("jev_mcp.app.provider") as mock_provider:
        mock_provider.fast_provider.evaluate_batch.side_effect = _echo_scores
        res = json.loads(
            server.jev_compact_context(state={"text": 123, "note": "keep"}, goal="g")
        )
    assert res["status"] == "COMPACTION_COMPLETE"


@patch(
    "jev_mcp.routing_provider.RoutingProvider.evaluate_batch", side_effect=_all_relevant
)
def test_scan_repo_logs_a_file_it_could_not_read(mock_eval, tmp_path):
    (tmp_path / "good.py").write_text("def ok():\n    return 1\n")
    (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00broken")
    with patch("jev_mcp.tools.files.logger") as mock_logger:
        server.jev_scan_repo(str(tmp_path), "find ok")
    warnings = " ".join(str(c) for c in mock_logger.warning.call_args_list)
    assert "bad.py" in warnings
