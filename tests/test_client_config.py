import copy
import json
import os

import pytest

from jev_mcp.client_config import (
    link_opencode,
    main,
    opencode_entry,
    unlink_opencode,
)

LEGACY_OTHERS = {
    "headroom": {"type": "local", "command": ["a", "b", "c"], "enabled": True},
    "serena": {"type": "local", "command": ["x"], "enabled": True},
}


def test_entry_is_a_local_server_with_command_as_one_array():
    assert opencode_entry("/repo/.venv/bin/python") == {
        "type": "local",
        "command": ["/repo/.venv/bin/python", "-m", "jev_mcp.server"],
    }


def test_link_writes_the_servers_shape_with_command_as_one_array():
    config = link_opencode({}, "/repo/.venv/bin/python")
    assert config["mcp"]["servers"]["jev-mcp"] == {
        "type": "local",
        "command": ["/repo/.venv/bin/python", "-m", "jev_mcp.server"],
    }
    assert config["$schema"] == "https://opencode.ai/config.json"


def test_link_leaves_other_servers_and_settings_exactly_as_they_were():
    before = {"$schema": "custom", "theme": "dark", "mcp": dict(LEGACY_OTHERS)}
    after = link_opencode(before, "/p")
    assert after["mcp"]["headroom"] == LEGACY_OTHERS["headroom"]
    assert after["mcp"]["serena"] == LEGACY_OTHERS["serena"]
    assert after["theme"] == "dark" and after["$schema"] == "custom"


def test_link_keeps_servers_already_registered_in_the_new_shape():
    before = {"mcp": {"servers": {"other": {"type": "local"}}}}
    after = link_opencode(before, "/p")
    assert after["mcp"]["servers"]["other"] == {"type": "local"}
    assert "jev-mcp" in after["mcp"]["servers"]


def test_link_replaces_a_legacy_top_level_entry_instead_of_registering_twice():
    legacy = {
        "mcp": {"jev-mcp": {"command": "python", "args": ["-m", "jev_mcp.server"]}}
    }
    after = link_opencode(legacy, "/p")
    assert "jev-mcp" not in after["mcp"] and "jev-mcp" in after["mcp"]["servers"]


def test_link_is_idempotent_and_does_not_mutate_its_input():
    before = {"mcp": dict(LEGACY_OTHERS)}
    snapshot = copy.deepcopy(before)
    once = link_opencode(before, "/p")
    assert link_opencode(once, "/p") == once and before == snapshot


def test_unlink_removes_both_shapes_and_keeps_everything_else():
    config = {
        "mcp": {
            **LEGACY_OTHERS,
            "jev-mcp": {},
            "servers": {"jev-mcp": {}, "other": {"type": "local"}},
        }
    }
    snapshot = copy.deepcopy(config)
    after = unlink_opencode(config)
    assert "jev-mcp" not in after["mcp"] and "jev-mcp" not in after["mcp"]["servers"]
    assert after["mcp"]["servers"]["other"] == {"type": "local"}
    assert after["mcp"]["headroom"] == LEGACY_OTHERS["headroom"]
    assert config == snapshot


def test_unlink_of_a_config_that_never_had_the_server_changes_nothing():
    config = {"mcp": dict(LEGACY_OTHERS)}
    assert unlink_opencode(config) == config
    assert unlink_opencode({"theme": "dark"}) == {"theme": "dark"}


@pytest.mark.parametrize(
    "config",
    [{"mcp": ["not", "an", "object"]}, {"mcp": {"servers": "not an object"}}],
)
def test_a_config_whose_mcp_section_has_an_unexpected_shape_is_refused(config):
    with pytest.raises(ValueError):
        link_opencode(config, "/p")
    with pytest.raises(ValueError):
        unlink_opencode(config)


def test_a_config_file_that_is_not_valid_json_is_left_untouched(tmp_path, capsys):
    path = tmp_path / "opencode.json"
    path.write_text('{"mcp": {"headroom": ')
    assert main(["link-opencode", "--path", str(path), "--python", "/p"]) == 1
    assert path.read_text() == '{"mcp": {"headroom": '
    assert str(path) in capsys.readouterr().err
    assert main(["unlink-opencode", "--path", str(path)]) == 1
    assert path.read_text() == '{"mcp": {"headroom": '


@pytest.mark.parametrize(
    "content",
    ['["a", "list"]', '{"mcp": "a string"}', '{"mcp": {"servers": 3}}'],
)
def test_a_config_file_with_an_unexpected_shape_is_left_untouched(
    tmp_path, capsys, content
):
    path = tmp_path / "opencode.json"
    path.write_text(content)
    assert main(["link-opencode", "--path", str(path), "--python", "/p"]) == 1
    assert path.read_text() == content
    assert str(path) in capsys.readouterr().err


def test_a_missing_config_file_is_not_created(tmp_path):
    path = tmp_path / "opencode.json"
    assert main(["link-opencode", "--path", str(path), "--python", "/p"]) == 0
    assert main(["unlink-opencode", "--path", str(path)]) == 0
    assert not path.exists()


def test_link_then_unlink_round_trips_through_the_file(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"mcp": LEGACY_OTHERS}))
    main(["link-opencode", "--path", str(path), "--python", "/p"])
    assert (
        json.loads(path.read_text())["mcp"]["servers"]["jev-mcp"]["command"][0] == "/p"
    )
    main(["unlink-opencode", "--path", str(path)])
    assert json.loads(path.read_text()) == {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {**LEGACY_OTHERS, "servers": {}},
    }


def test_a_file_that_needs_no_change_is_not_rewritten(tmp_path):
    path = tmp_path / "opencode.json"
    original = '{"mcp":{"headroom":{"type":"local"}}}\n'
    path.write_text(original)
    assert main(["unlink-opencode", "--path", str(path)]) == 0
    assert path.read_text() == original


def test_a_symlinked_config_stays_a_symlink_and_its_target_is_updated(tmp_path):
    target = tmp_path / "dotfiles" / "opencode.json"
    target.parent.mkdir()
    target.write_text("{}")
    path = tmp_path / "opencode.json"
    path.symlink_to(target)
    assert main(["link-opencode", "--path", str(path), "--python", "/p"]) == 0
    assert path.is_symlink()
    assert (
        json.loads(target.read_text())["mcp"]["servers"]["jev-mcp"]["command"][0]
        == "/p"
    )


def test_python_defaults_to_the_repo_virtualenv(tmp_path):
    path = tmp_path / "opencode.json"
    path.write_text("{}")
    assert main(["link-opencode", "--path", str(path)]) == 0
    python = json.loads(path.read_text())["mcp"]["servers"]["jev-mcp"]["command"][0]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert python == os.path.join(repo, ".venv", "bin", "python")
