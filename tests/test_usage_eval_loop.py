"""The agent loop the tool-usage evaluation runs: every tool call in a turn gets its result."""

import asyncio
import json
from types import SimpleNamespace as NS

from tests.evals.usage.loop import agent_loop


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.seen = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.seen.append([dict(m) for m in kwargs["messages"]])
        return self._responses.pop(0)


class FakeConnection:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return {"echo": name}


def _tool_use(call_id, name, arguments=None):
    return NS(type="tool_use", id=call_id, name=name, input=arguments or {})


def _final(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


def _run(client, connection):
    return asyncio.run(
        agent_loop(client, "m", "question", [], connection, system="rules")
    )


def test_every_parallel_tool_call_in_one_turn_gets_its_own_result():
    # Arrange
    turn = NS(
        stop_reason="tool_use",
        content=[_tool_use("a", "first"), _tool_use("b", "second")],
    )
    client = FakeClient([turn, _final("<response>ok</response>")])
    connection = FakeConnection()

    # Act
    text, metrics = _run(client, connection)

    # Assert
    results = client.seen[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert all(r["type"] == "tool_result" for r in results)
    assert results[0]["content"] == json.dumps({"echo": "first"})
    assert connection.calls == [("first", {}), ("second", {})]
    assert text == "<response>ok</response>"
    assert {name: m["count"] for name, m in metrics.items()} == {
        "first": 1,
        "second": 1,
    }


def test_a_failing_tool_call_is_reported_back_and_the_others_still_run():
    # Arrange
    class Flaky(FakeConnection):
        async def call_tool(self, name, arguments):
            if name == "bad":
                raise RuntimeError("boom")
            return await super().call_tool(name, arguments)

    turn = NS(
        stop_reason="tool_use", content=[_tool_use("a", "bad"), _tool_use("b", "good")]
    )
    client = FakeClient([turn, _final("done")])

    # Act
    _run(client, Flaky())

    # Assert
    first, second = client.seen[1][-1]["content"]
    assert "Error executing tool bad: boom" in first["content"]
    assert second["content"] == json.dumps({"echo": "good"})


def test_a_turn_with_no_tool_calls_returns_its_text_without_a_second_request():
    client = FakeClient([_final("<response>7</response>")])

    text, metrics = _run(client, FakeConnection())

    assert text == "<response>7</response>"
    assert metrics == {}
    assert len(client.seen) == 1


def test_mcp_content_blocks_are_sent_back_as_their_text():
    # The SDK client returns TextContent objects, which json.dumps cannot serialise.
    class BlockConnection(FakeConnection):
        async def call_tool(self, name, arguments):
            return [
                NS(type="text", text='{"status": "SUCCESS"}'),
                NS(type="text", text="more"),
            ]

    turn = NS(stop_reason="tool_use", content=[_tool_use("a", "t")])
    client = FakeClient([turn, _final("done")])

    _run(client, BlockConnection())

    assert client.seen[1][-1]["content"][0]["content"] == '{"status": "SUCCESS"}\nmore'


# The two config tools write only into the seeded temp home; the questions need their read actions.
CONFIG_TOOLS_FOR_READ_ACTIONS = {"jev_manage_router_config", "jev_manage_temperature"}


def test_the_model_is_offered_only_the_read_only_tools_and_the_two_config_tools():
    from tests.evals.usage.loop import EVAL_TOOLS
    from tests.test_tool_surface import ANNOTATIONS

    read_only = {
        name for name, (is_read_only, *_rest) in ANNOTATIONS.items() if is_read_only
    }

    assert EVAL_TOOLS == read_only | CONFIG_TOOLS_FOR_READ_ACTIONS


def test_no_offered_tool_reaches_beyond_the_machine():
    from tests.evals.usage.loop import EVAL_TOOLS
    from tests.test_tool_surface import ANNOTATIONS

    for name in EVAL_TOOLS:
        _read_only, _destructive, _idempotent, open_world = ANNOTATIONS[name]
        assert not open_world, name


def test_tools_outside_the_allow_list_are_not_sent_to_the_model():
    # Arrange
    tools = [
        {"name": "jev_read_file"},
        {"name": "jev_train_lora"},
        {"name": "jev_run_browser_agent"},
    ]
    seen = []

    class Client(FakeClient):
        def _create(self, **kwargs):
            seen.append([t["name"] for t in kwargs["tools"]])
            return super()._create(**kwargs)

    # Act
    asyncio.run(
        agent_loop(
            Client([_final("done")]), "m", "q", tools, FakeConnection(), system="s"
        )
    )

    # Assert
    assert seen == [["jev_read_file"]]
