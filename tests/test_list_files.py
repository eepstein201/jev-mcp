"""jev_list_files: a read-only, root-confined, symlink-safe directory listing."""

import json
import os
import stat

import pytest

from jev_mcp import security, server
from jev_mcp.listing import MAX_DEPTH_CAP, MAX_ENTRIES_CAP


@pytest.fixture(autouse=True)
def _isolated_roots(monkeypatch, tmp_path):
    # tmp_path is under the temp root, so it is allowed; no real home or config is touched.
    monkeypatch.delenv("JEV_ALLOWED_ROOTS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _list(path, **kwargs):
    return json.loads(server.jev_list_files(str(path), **kwargs))


def _names(res):
    return [e["path"] for e in res["entries"]]


def _tree(root):
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("def a():\n    pass\n")
    (root / "pkg" / "b.py").write_text("x")
    (root / "pkg" / "sub" / "c.py").write_text("y")
    return root / "pkg"


def test_a_normal_listing_is_sorted_and_typed(tmp_path):
    pkg = _tree(tmp_path)
    res = _list(pkg)
    assert res["status"] == "SUCCESS"
    assert res["path"] == str(pkg)
    assert _names(res) == ["a.py", "b.py", "sub", "sub/c.py"]
    by_path = {e["path"]: e for e in res["entries"]}
    assert by_path["a.py"] == {"path": "a.py", "type": "file", "size": 18}
    assert by_path["sub"] == {"path": "sub", "type": "dir", "size": None}
    assert res["truncated"] is False
    assert res["skipped"] == 0


def test_the_listing_is_deterministic(tmp_path):
    pkg = _tree(tmp_path)
    assert server.jev_list_files(str(pkg)) == server.jev_list_files(str(pkg))


def test_max_depth_limits_how_far_it_descends(tmp_path):
    pkg = _tree(tmp_path)
    assert _names(_list(pkg, max_depth=1)) == ["a.py", "b.py", "sub"]
    assert "sub/c.py" in _names(_list(pkg, max_depth=2))


def test_max_depth_is_clamped_to_the_hard_cap(tmp_path):
    deep = tmp_path
    for i in range(MAX_DEPTH_CAP + 2):
        deep = deep / f"d{i}"
    deep.mkdir(parents=True)
    res = _list(tmp_path, max_depth=99)
    assert max(p.count("/") + 1 for p in _names(res)) == MAX_DEPTH_CAP


def test_max_entries_truncates_and_says_so(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("x")
    res = _list(tmp_path, max_entries=4)
    assert len(res["entries"]) == 4
    assert res["truncated"] is True
    assert _names(res) == ["f0.txt", "f1.txt", "f2.txt", "f3.txt"]


def test_exactly_max_entries_is_not_truncated(tmp_path):
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("x")
    res = _list(tmp_path, max_entries=3)
    assert len(res["entries"]) == 3
    assert res["truncated"] is False


def test_max_entries_is_clamped_to_the_hard_cap(tmp_path):
    for i in range(MAX_ENTRIES_CAP + 5):
        (tmp_path / f"f{i:04}.txt").write_text("")
    res = _list(tmp_path, max_entries=10**6)
    assert len(res["entries"]) == MAX_ENTRIES_CAP
    assert res["truncated"] is True


def test_non_positive_limits_are_raised_to_one(tmp_path):
    (tmp_path / "a").write_text("")
    (tmp_path / "b").write_text("")
    res = _list(tmp_path, max_depth=0, max_entries=-5)
    assert len(res["entries"]) == 1
    assert res["truncated"] is True


@pytest.mark.parametrize("bad", ["3", 2.5, None, True])
def test_non_integer_limits_are_an_error(tmp_path, bad):
    assert _list(tmp_path, max_depth=bad)["status"] == "ERROR"
    assert _list(tmp_path, max_entries=bad)["status"] == "ERROR"


def test_a_path_outside_every_root_is_refused(tmp_path, monkeypatch):
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    (other / "x.txt").write_text("x")
    monkeypatch.chdir(project)
    monkeypatch.setattr(security, "_TEMP_ROOT", str(project))
    monkeypatch.setattr(security, "_home", lambda: str(tmp_path / "home"))
    res = _list(other)
    assert res["status"] == "BLOCKED"
    assert "entries" not in res
    assert "x.txt" not in json.dumps(res)


def test_dotdot_traversal_out_of_the_root_is_refused(tmp_path, monkeypatch):
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(security, "_TEMP_ROOT", str(project))
    assert _list(f"{project}/../other")["status"] == "BLOCKED"
    assert _list("../other")["status"] == "BLOCKED"


@pytest.mark.parametrize("path", ["/etc", "/usr/lib", "/private/etc", "/"])
def test_absolute_system_paths_are_refused(path):
    assert _list(path)["status"] == "BLOCKED"


def test_a_symlink_to_outside_is_listed_as_a_link_and_not_followed(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "secret-name.txt").write_text("top secret")
    (project / "real.txt").write_text("r")
    os.symlink(outside, project / "escape")
    os.symlink(outside / "secret-name.txt", project / "filelink")
    monkeypatch.chdir(project)
    monkeypatch.setattr(security, "_TEMP_ROOT", str(project))
    res = _list(project, max_depth=5)
    by_path = {e["path"]: e for e in res["entries"]}
    assert by_path["escape"] == {"path": "escape", "type": "link", "size": None}
    assert by_path["filelink"] == {"path": "filelink", "type": "link", "size": None}
    assert sorted(by_path) == ["escape", "filelink", "real.txt"]
    dumped = json.dumps(res)
    assert "secret-name" not in dumped
    assert str(outside) not in dumped


def test_asking_for_a_symlink_that_points_outside_is_refused(tmp_path, monkeypatch):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "x.txt").write_text("x")
    os.symlink(outside, project / "escape")
    monkeypatch.chdir(project)
    monkeypatch.setattr(security, "_TEMP_ROOT", str(project))
    res = _list(project / "escape")
    assert res["status"] == "BLOCKED"
    assert "x.txt" not in json.dumps(res)


def test_a_symlink_loop_inside_the_root_is_not_followed(tmp_path):
    (tmp_path / "d").mkdir()
    os.symlink(tmp_path / "d", tmp_path / "d" / "loop")
    res = _list(tmp_path, max_depth=5)
    assert _names(res) == ["d", "d/loop"]
    assert res["entries"][1]["type"] == "link"


@pytest.mark.parametrize(
    "relative",
    [".aws/credentials", ".ssh/id_rsa", ".env", ".kube/config", ".netrc"],
)
def test_a_sensitive_path_is_refused_when_asked_for(tmp_path, relative):
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret")
    asks = (target, target.parent) if target.parent != tmp_path else (target,)
    for ask in asks:
        res = _list(ask)
        assert res["status"] == "BLOCKED"
        assert "entries" not in res


def test_sensitive_entries_are_absent_from_listings_and_counted(tmp_path):
    (tmp_path / "ok.txt").write_text("fine")
    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws" / "credentials").write_text("k")
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".env").write_text("TOKEN=1")
    (tmp_path / ".env.production").write_text("TOKEN=2")
    (tmp_path / ".env.example").write_text("TOKEN=")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "id_rsa").write_text("k")
    os.symlink(tmp_path / "ok.txt", tmp_path / "id_ed25519")
    res = _list(tmp_path, max_depth=3)
    assert sorted(_names(res)) == [".env.example", "ok.txt", "sub"]
    assert res["skipped"] == 6
    dumped = json.dumps(res).casefold()
    for secret in (".aws", ".ssh", "credentials", "id_rsa", "id_ed25519", "production"):
        assert secret not in dumped


def test_sensitive_names_match_case_insensitively(tmp_path):
    (tmp_path / ".SSH").mkdir()
    (tmp_path / "ID_RSA").write_text("k")
    res = _list(tmp_path)
    assert res["entries"] == []
    assert res["skipped"] == 2


def test_library_credential_directories_are_skipped(tmp_path):
    lib = tmp_path / "Library"
    (lib / "Keychains").mkdir(parents=True)
    (lib / "Developer").mkdir()
    res = _list(tmp_path, max_depth=3)
    assert _names(res) == ["Library", "Library/Developer"]
    assert res["skipped"] == 1


def test_a_file_path_is_an_error(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    res = _list(f)
    assert res["status"] == "ERROR"
    assert "not a directory" in res["message"].lower()


def test_a_missing_path_is_an_error(tmp_path):
    assert _list(tmp_path / "nope")["status"] == "ERROR"


@pytest.mark.parametrize("bad", ["", "   ", "a\x00b"])
def test_empty_or_malformed_paths_are_refused_without_raising(bad):
    res = json.loads(server.jev_list_files(bad))
    assert res["status"] in ("ERROR", "BLOCKED")


def test_an_unreadable_directory_is_an_error_not_a_raise(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "x").write_text("x")
    locked.chmod(0)
    try:
        res = _list(locked)
    finally:
        locked.chmod(stat.S_IRWXU)
    if os.geteuid() == 0:
        pytest.skip("root can read a mode-000 directory")
    assert res["status"] == "ERROR"
    assert "x" not in res.get("entries", [])


def test_an_unreadable_subdirectory_is_counted_not_hidden(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        res = _list(tmp_path, max_depth=3)
    finally:
        locked.chmod(stat.S_IRWXU)
    if os.geteuid() == 0:
        pytest.skip("root can read a mode-000 directory")
    assert res["status"] == "SUCCESS"
    assert _names(res) == ["locked", "ok.txt"]
    assert res["unreadable"] == 1


def test_special_files_are_typed_other(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    res = _list(tmp_path)
    assert res["entries"] == [{"path": "pipe", "type": "other", "size": None}]


def test_an_entry_that_vanishes_mid_listing_does_not_raise(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("x")
    real_scandir = os.scandir

    class Gone:
        name = "gone.txt"

        def is_symlink(self):
            raise FileNotFoundError

    class Scan:
        def __init__(self, fd):
            self.inner = real_scandir(fd)

        def __enter__(self):
            return [*self.inner, Gone()]

        def __exit__(self, *exc):
            self.inner.close()

    monkeypatch.setattr(os, "scandir", Scan)
    res = _list(tmp_path)
    assert res["status"] == "SUCCESS"
    assert _names(res) == ["a.txt"]


def test_the_tool_never_raises_even_if_the_walk_does(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("jev_mcp.listing.list_directory", boom)
    res = json.loads(server.jev_list_files(str(tmp_path)))
    assert res["status"] == "ERROR"


def test_it_never_reads_file_contents(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello")

    def no_open(*args, **kwargs):
        raise AssertionError("opened a file")

    monkeypatch.setattr("builtins.open", no_open)
    assert _list(tmp_path)["status"] == "SUCCESS"


def test_one_call_returns_what_is_needed_to_count_a_packages_files(tmp_path):
    # The usage-eval problem: counting a three-file package took 145 calls without a listing.
    pkg = tmp_path / "inventory"
    pkg.mkdir()
    for name in ("__init__.py", "stock.py", "orders.py"):
        (pkg / name).write_text("def f():\n    pass\n")
    res = _list(pkg)
    python_files = [
        e["path"]
        for e in res["entries"]
        if e["type"] == "file" and e["path"].endswith(".py")
    ]
    assert python_files == ["__init__.py", "orders.py", "stock.py"]
    assert res["truncated"] is False and res["skipped"] == 0


def test_truncation_inside_a_subdirectory_stops_the_whole_walk(tmp_path):
    (tmp_path / "a").mkdir()
    for i in range(3):
        (tmp_path / "a" / f"f{i}").write_text("")
    (tmp_path / "z.txt").write_text("")
    res = _list(tmp_path, max_entries=2)
    assert _names(res) == ["a", "a/f0"]
    assert res["truncated"] is True


def test_a_file_whose_size_cannot_be_read_is_listed_without_one(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("x")

    class Entry:
        name = "a.txt"

        def is_symlink(self):
            return False

        def is_dir(self, follow_symlinks=True):
            return False

        def is_file(self, follow_symlinks=True):
            return True

        def stat(self, follow_symlinks=True):
            raise PermissionError

    class Scan:
        def __init__(self, fd):
            pass

        def __enter__(self):
            return [Entry()]

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(os, "scandir", Scan)
    res = _list(tmp_path)
    assert res["entries"] == [{"path": "a.txt", "type": "file", "size": None}]


def test_a_directory_that_cannot_be_scanned_is_counted_unreadable(
    tmp_path, monkeypatch
):
    (tmp_path / "sub").mkdir()
    real_scandir = os.scandir
    calls = []

    def flaky(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise PermissionError
        return real_scandir(fd)

    monkeypatch.setattr(os, "scandir", flaky)
    res = _list(tmp_path)
    assert _names(res) == ["sub"]
    assert res["unreadable"] == 1
