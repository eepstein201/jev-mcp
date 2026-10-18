"""Read-only directory listing behind jev_list_files.

Only names, types and sizes are reported: no file is opened, no symlink is followed
(a link is listed as a link, without its target) and entries that match the
credential rules in security.py are counted but never named.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from jev_mcp.security import is_sensitive_entry

DEFAULT_MAX_DEPTH = 2
MAX_DEPTH_CAP = 5
DEFAULT_MAX_ENTRIES = 200
MAX_ENTRIES_CAP = 1000

# O_NOFOLLOW makes a directory swapped for a symlink after the scan fail instead of being entered.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@dataclass
class _Walk:
    max_depth: int
    max_entries: int
    entries: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    unreadable: int = 0
    truncated: bool = False


def clamp_limit(value: Any, name: str, cap: int) -> int:
    """`value` raised to at least 1 and lowered to `cap`; ValueError when it is not an int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return max(1, min(value, cap))


def _scan(fd: int) -> list[Any]:
    with os.scandir(fd) as iterator:
        return sorted(iterator, key=lambda entry: entry.name)


def _describe(entry: Any) -> tuple[str, int | None]:
    """(type, size) from lstat-style checks only; raises OSError if the entry vanished."""
    if entry.is_symlink():
        return "link", None
    if entry.is_dir(follow_symlinks=False):
        return "dir", None
    if entry.is_file(follow_symlinks=False):
        try:
            return "file", entry.stat(follow_symlinks=False).st_size
        except OSError:
            return "file", None
    return "other", None


def _walk_dir(fd: int, parent_name: str, prefix: str, depth: int, walk: _Walk) -> None:
    try:
        children = _scan(fd)
    except OSError:
        walk.unreadable += 1
        return
    for entry in children:
        if walk.truncated:
            return
        if is_sensitive_entry(parent_name, entry.name):
            walk.skipped += 1
            continue
        try:
            kind, size = _describe(entry)
        except OSError:
            walk.unreadable += 1
            continue
        if len(walk.entries) >= walk.max_entries:
            walk.truncated = True
            return
        relative = f"{prefix}{entry.name}"
        walk.entries.append({"path": relative, "type": kind, "size": size})
        if kind == "dir" and depth < walk.max_depth:
            _descend(fd, entry.name, relative, depth + 1, walk)


def _descend(parent_fd: int, name: str, relative: str, depth: int, walk: _Walk) -> None:
    try:
        child_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError:
        walk.unreadable += 1
        return
    try:
        _walk_dir(child_fd, name, f"{relative}/", depth, walk)
    finally:
        os.close(child_fd)


def list_directory(path: str, max_depth: int, max_entries: int) -> dict[str, Any]:
    """List `path`, which the caller has already checked with is_safe_path.

    Raises OSError when the top directory itself cannot be opened.
    """
    real = os.path.realpath(path)
    walk = _Walk(max_depth=max_depth, max_entries=max_entries)
    root_fd = os.open(real, _DIR_FLAGS)
    try:
        _walk_dir(root_fd, os.path.basename(real), "", 1, walk)
    finally:
        os.close(root_fd)
    return {
        "status": "SUCCESS",
        "path": path,
        "entries": walk.entries,
        "truncated": walk.truncated,
        "skipped": walk.skipped,
        "unreadable": walk.unreadable,
    }
