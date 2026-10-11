import json

import pytest

from jev_mcp import browser_agent


class FakeBrowser:
    """Stands in for jev_ultrafast.Browser so no real Chrome is driven."""

    instances: list["FakeBrowser"] = []
    kind = "click"
    page_text = None  # set to put specific text on the fake page

    def __init__(self, url):
        self.url = url
        self.acted = []
        self.closed = False
        FakeBrowser.instances.append(self)

    def observe(self, screenshot=False):
        return {
            "url": self.url,
            "title": "Fake page",
            "text": self.page_text if self.page_text is not None else f"{len(self.acted)} actions so far",
            "actions": [{"id": "a1", "kind": self.kind, "label": "Submit", "node": 1, "role": "button"}],
            "fingerprint": f"fp-{len(self.acted)}",
        }

    def fresh(self, page, action=None):
        return True

    def act(self, action, page, text=None):
        self.acted.append((action["id"], text))

    def close(self):
        self.closed = True


class ScriptedRouter:
    """Answers the operation question from a script, and always picks target 1."""

    operations: list[str] = []
    prompts: list[str] = []

    def evaluate_batch(self, state, questions):
        results = {}
        for q in questions:
            ScriptedRouter.prompts.append(q.prompt)
            if q.key == "operation":
                step = len(state.get("recent_actions", []))
                pick = ScriptedRouter.operations[min(step, len(ScriptedRouter.operations) - 1)]
            else:
                pick = q.options[0]
            share = 0.0 if len(q.options) == 1 else 0.1 / (len(q.options) - 1)
            probabilities = {opt: (0.9 if len(q.options) > 1 else 1.0) if opt == pick else share for opt in q.options}
            results[q.key] = {"choice": pick, "confidence": probabilities[pick], "probabilities": probabilities}
        return results


@pytest.fixture
def fake_agent(monkeypatch):
    """Real jev-ultrafast agent loop, with the browser and the local engines replaced."""
    agent_module, _ = browser_agent.load_ultrafast()
    FakeBrowser.instances = []
    FakeBrowser.kind = "click"
    FakeBrowser.page_text = None
    ScriptedRouter.operations = ["CLICK", "DONE"]
    ScriptedRouter.prompts = []
    monkeypatch.setattr(agent_module, "Browser", FakeBrowser)
    monkeypatch.setattr("jev_mcp.ultrafast_adapter.RoutingProvider", ScriptedRouter)
    return agent_module


def test_done_is_forced_when_the_goal_marker_is_on_the_page(fake_agent):
    # The engine never says DONE here (script is all CLICK); the visible success
    # marker must end the run before any action is taken.
    FakeBrowser.page_text = "MISSION_COMPLETE"
    ScriptedRouter.operations = ["CLICK"]

    result = browser_agent.run_browser_agent(
        "https://example.test/form",
        "Click Submit. The page shows MISSION_COMPLETE only when you click the right one.")

    assert result["result"] == "done"
    assert result["steps"] == []
    assert FakeBrowser.instances[0].acted == []


def test_premature_done_is_demoted_when_the_marker_is_absent(fake_agent):
    # The engine says DONE at once, but the stated success marker is nowhere on
    # the page: the gate must fall back to the runner-up operation (CLICK).
    ScriptedRouter.operations = ["DONE"]

    result = browser_agent.run_browser_agent(
        "https://example.test/form",
        "Click Submit. The page shows MISSION_COMPLETE only when you click the right one.")

    assert result["steps"] and all(s["kind"] == "click" for s in result["steps"])
    assert FakeBrowser.instances[0].acted


def test_done_without_a_stated_marker_passes_through_unchanged(fake_agent):
    # No success marker in the goal: the engine's DONE is trusted, as before.
    ScriptedRouter.operations = ["DONE"]

    result = browser_agent.run_browser_agent("https://example.test/form", "Click the Submit button.")

    assert result["result"] == "done"
    assert result["steps"] == []


def test_recently_tried_elements_are_flagged_in_the_target_listing(fake_agent):
    ScriptedRouter.operations = ["CLICK"]

    browser_agent.run_browser_agent("https://example.test/form", "Click the Submit button.", max_steps=3)

    target_prompts = [p for p in ScriptedRouter.prompts if "Choose the target index" in p]
    assert target_prompts and any("(tried" in p for p in target_prompts)


def test_operation_prompt_states_the_completion_criterion(fake_agent):
    from jev_mcp import ultrafast_adapter

    operations = {"CLICK": "Click an element.", "DONE": "Every requirement is visibly satisfied."}
    targets = {"CLICK": {"1": {"label": "Submit", "element": "Submit"}}}
    ultrafast_adapter.local_choose(
        {"page": {"url": "u", "title": "t", "text": "x"}},
        "Click Submit. The page shows MISSION_COMPLETE only when you click the right one.",
        [], operations, targets, {})

    operation_prompts = [p for p in ScriptedRouter.prompts if "Choose the best operation" in p]
    assert operation_prompts and "only when the page shows MISSION_COMPLETE" in operation_prompts[-1]


def test_alternating_picks_add_a_warning_to_the_operation_prompt(fake_agent):
    from jev_mcp import ultrafast_adapter

    history = [
        {"action": "A", "kind": "click", "text": None, "page_changed": True},
        {"action": "B", "kind": "click", "text": None, "page_changed": True},
    ] * 2
    operations = {"CLICK": "Click an element.", "DONE": "Every requirement is visibly satisfied."}
    targets = {"CLICK": {"1": {"label": "A", "element": "A"}, "2": {"label": "B", "element": "B"}}}
    ultrafast_adapter.local_choose(
        {"page": {"url": "u", "title": "t", "text": "x"}}, "goal", history, operations, targets, {})

    operation_prompts = [p for p in ScriptedRouter.prompts if "Choose the best operation" in p]
    assert operation_prompts and "alternating" in operation_prompts[-1]


def _dense_targets(n=15):
    return {"CLICK": {str(i): {"label": f"Select — part {i}", "element": f"part {i}"} for i in range(1, n + 1)}}


def test_dense_target_listings_are_narrowed_to_a_shortlist(fake_agent):
    from jev_mcp import ultrafast_adapter

    ScriptedRouter.prompts = []
    operations = {"CLICK": "Click an element.", "DONE": "Every requirement is visibly satisfied."}
    out = ultrafast_adapter.local_choose(
        {"page": {"url": "u", "title": "t", "text": "x"}}, "goal", [], operations, _dense_targets(), {})

    shortlist_prompts = [p for p in ScriptedRouter.prompts if "Shortlist" in p]
    assert shortlist_prompts, "dense listing was never narrowed"
    assert shortlist_prompts[0].count("[") <= 5
    assert out["answers"]["click_target"]["choice"] in {"1", "2", "3", "4", "5"}


def test_small_target_listings_are_not_narrowed(fake_agent):
    from jev_mcp import ultrafast_adapter

    ScriptedRouter.prompts = []
    operations = {"CLICK": "Click an element.", "DONE": "Every requirement is visibly satisfied."}
    targets = {"CLICK": {"1": {"label": "A", "element": "A"}, "2": {"label": "B", "element": "B"},
               "3": {"label": "C", "element": "C"}, "4": {"label": "D", "element": "D"}}}
    ultrafast_adapter.local_choose(
        {"page": {"url": "u", "title": "t", "text": "x"}}, "goal", [], operations, targets, {})

    assert not [p for p in ScriptedRouter.prompts if "Shortlist" in p]


def test_consecutive_repeated_picks_get_a_break_the_loop_note(fake_agent):
    from jev_mcp import ultrafast_adapter

    ScriptedRouter.prompts = []
    history = [{"action": "Wait for the page to update", "kind": "wait", "text": None,
                "page_changed": False}] * 5
    operations = {"CLICK": "Click an element.", "WAIT": "Wait for the page to update.",
                  "DONE": "Every requirement is visibly satisfied."}
    ultrafast_adapter.local_choose(
        {"page": {"url": "u", "title": "t", "text": "x"}}, "goal", history, operations,
        {"CLICK": {"1": {"label": "A", "element": "A"}}}, {})

    operation_prompts = [p for p in ScriptedRouter.prompts if "Choose the best operation" in p]
    assert operation_prompts and "in a row" in operation_prompts[-1]


def test_a_just_filled_field_gets_a_progression_note(fake_agent):
    from jev_mcp import ultrafast_adapter

    ScriptedRouter.prompts = []
    history = [{"action": "Search", "kind": "fill", "text": "blue orchid 42", "page_changed": True}]
    operations = {"CLICK": "Click an element.", "TYPE_TEXT": "Enter text.", "WAIT": "Wait.",
                  "DONE": "Every requirement is visibly satisfied."}
    ultrafast_adapter.local_choose(
        {"page": {"url": "u", "title": "t", "text": "x"}}, "goal", history, operations,
        {"CLICK": {"1": {"label": "Submit", "element": "Submit"}}}, {})

    operation_prompts = [p for p in ScriptedRouter.prompts if "Choose the best operation" in p]
    assert operation_prompts and "waiting rarely completes a form" in operation_prompts[-1]


def test_wait_loop_detector():
    from jev_mcp.browser_agent import _is_stuck_in_waits

    waits = [{"kind": "wait", "page_changed": False}] * 3
    assert _is_stuck_in_waits(waits)
    assert not _is_stuck_in_waits(waits[:2])
    assert not _is_stuck_in_waits([{"kind": "fill", "page_changed": False}] + waits[:2])
    assert not _is_stuck_in_waits([{"kind": "wait", "page_changed": True}] + waits[:2])


def test_narrowed_answers_cover_the_full_candidate_set_again(fake_agent):
    # The submodule's validate_choice requires probabilities over exactly the
    # offered ids, so a shortlist answer must be expanded back to all candidates.
    _, model_module = browser_agent.load_ultrafast()

    class FakeModel:
        @staticmethod
        def action_space(actions):
            targets = {"CLICK": {str(i + 1): {"id": f"e{i + 1}", "label": f"part {i}"} for i in range(15)}}
            return [], targets, {}

        validate_choice = staticmethod(model_module.validate_choice)

    page = {"actions": [], "title": "t", "text": "x"}
    decision = browser_agent.choose_locally(FakeModel, page, "goal", [])

    assert decision["operation"] == "CLICK"
    assert decision["probabilities"]["e1"] > 0.5


def test_agent_clicks_the_chosen_element_then_reports_done(fake_agent):
    result = browser_agent.run_browser_agent("https://example.test/form", "Click the Submit button.")

    assert result["status"] == "SUCCESS"
    assert result["result"] == "done"
    assert [(s["kind"], s["action"]) for s in result["steps"]] == [("click", "Submit")]
    assert FakeBrowser.instances[0].acted == [("a1", None)]
    assert FakeBrowser.instances[0].closed


def test_the_ultrafast_module_is_left_unpatched_after_a_run(fake_agent):
    choose_before, field_text_before = fake_agent.choose, fake_agent.field_text

    browser_agent.run_browser_agent("https://example.test/form", "Click the Submit button.")

    assert fake_agent.choose is choose_before
    assert fake_agent.field_text is field_text_before


def test_text_for_a_field_comes_from_the_local_smart_engine(fake_agent, monkeypatch):
    FakeBrowser.kind = "fill"
    ScriptedRouter.operations = ["TYPE_TEXT", "DONE"]
    calls = []

    def fake_post_json(url, payload, timeout=60):
        calls.append(url)
        return {"choices": [{"message": {"content": json.dumps({"text": "hello world"})}}]}

    monkeypatch.setattr(browser_agent, "post_json", fake_post_json)
    monkeypatch.setenv("JEV_SMART_PORT", "8081")

    result = browser_agent.run_browser_agent("https://example.test/form", "Type hello world into the field.")

    assert FakeBrowser.instances[0].acted == [("a1", "hello world")]
    assert result["steps"][0]["text"] == "hello world"
    assert calls == ["http://127.0.0.1:8081/v1/chat/completions"]


def test_nothing_is_typed_when_the_local_engine_returns_no_usable_text(fake_agent, monkeypatch):
    FakeBrowser.kind = "fill"
    ScriptedRouter.operations = ["TYPE_TEXT", "DONE"]
    no_text = {"choices": [{"message": {"content": json.dumps({"text": None})}}]}
    monkeypatch.setattr(browser_agent, "post_json", lambda url, payload, timeout=60: no_text)

    result = browser_agent.run_browser_agent("https://example.test/form", "Type my password into the field.")

    assert result["status"] == "ERROR"
    assert "nothing typed" in result["message"]
    assert FakeBrowser.instances[0].acted == []


def test_run_stops_at_the_step_limit(fake_agent):
    ScriptedRouter.operations = ["CLICK"]

    result = browser_agent.run_browser_agent("https://example.test/form", "Keep clicking.", max_steps=2)

    assert result["result"] == "step_limit"
    assert len(result["steps"]) == 2
    assert FakeBrowser.instances[0].closed


@pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "chrome://settings", "example.test", ""])
def test_only_http_and_https_urls_are_opened(fake_agent, url):
    result = browser_agent.run_browser_agent(url, "anything")

    assert result["status"] == "ERROR"
    assert "http" in result["message"]
    assert FakeBrowser.instances == []


def test_an_empty_goal_is_rejected_before_a_browser_is_opened(fake_agent):
    result = browser_agent.run_browser_agent("https://example.test/", "   ")

    assert result["status"] == "ERROR"
    assert FakeBrowser.instances == []


def test_missing_submodule_is_reported_with_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("JEV_ULTRAFAST_PATH", str(tmp_path / "nowhere"))

    result = browser_agent.run_browser_agent("https://example.test/", "anything")

    assert result["status"] == "ERROR"
    assert "git submodule update --init" in result["message"]


def test_submodule_is_found_from_the_package_location_not_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("JEV_ULTRAFAST_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    assert (browser_agent.ultrafast_path() / "jev_ultrafast" / "agent.py").exists()


def test_a_failure_inside_the_agent_is_returned_not_raised_and_the_browser_is_closed(fake_agent, monkeypatch):
    def broken_act(self, action, page, text=None):
        raise RuntimeError("tab crashed")

    monkeypatch.setattr(FakeBrowser, "act", broken_act)

    result = browser_agent.run_browser_agent("https://example.test/form", "Click the Submit button.")

    assert result["status"] == "ERROR"
    assert "tab crashed" in result["message"]
    assert FakeBrowser.instances[0].closed


def test_target_question_shows_the_element_labels_to_the_model(fake_agent):
    # Options are bare indices; without the labels the model cannot tell elements apart.
    browser_agent.run_browser_agent("https://example.test/form", "Click the Submit button.")

    target_prompts = [p for p in ScriptedRouter.prompts if "target" in p.lower()]
    assert target_prompts and all("[1] Submit" in p for p in target_prompts)


def test_tool_returns_json_for_a_run(fake_agent):
    from jev_mcp.server import jev_run_browser_agent

    out = json.loads(jev_run_browser_agent(url="https://example.test/form", goal="Click the Submit button."))

    assert out["status"] == "SUCCESS"
    assert out["result"] == "done"


def test_tool_returns_json_for_a_bad_url():
    from jev_mcp.server import jev_run_browser_agent

    out = json.loads(jev_run_browser_agent(url="file:///etc/passwd", goal="read it"))

    assert out["status"] == "ERROR"
