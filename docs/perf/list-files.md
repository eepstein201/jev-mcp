# jev_list_files

## Why

The tool-usage evaluation (`tests/evals/usage/`) needed 145 tool calls and 8 minutes to count the functions in a three-file package, because no jev tool could list a directory (`docs/TODO.md`). `jev_list_files` is the read-only listing the owner chose; it does not add an unfiltered file read and does not widen what any other tool can read.

## Design

- `tools/files.py` registers the tool (read-only, non-destructive, idempotent, not open-world, pinned in `tests/test_tool_surface.py`) and does the checks; `listing.py` does the walk.
- Input: `path` (required), `max_depth` (default 2, cap 5; 1 lists the directory's own entries), `max_entries` (default 200, cap 1000). Values below 1 become 1, values over the cap are lowered to it, non-integers are an `ERROR`.
- Output: `{"status": "SUCCESS", "path", "entries": [{"path": relative, "type": "file"|"dir"|"link"|"other", "size": bytes or null}], "truncated", "skipped", "unreadable"}`. Entries come in depth-first order, sorted by name within each directory, so the same tree always gives the same answer. `truncated` is true only when another entry existed beyond `max_entries`. Directories deeper than `max_depth` are not entered and not flagged.
- Problems answer `{"status": "ERROR"|"BLOCKED", "message"}`; the tool never raises (`json_errors` is a backstop).

## Security rules

1. The requested path goes through `security.is_safe_path` (the function `jev_read_file` and `jev_scan_repo` use) and `denial_message`. It resolves `..`, `~` and symlinks first, so a path or a link that resolves outside the allowed roots, or onto a credential location or system prefix, is `BLOCKED`.
2. Entries whose names match the credential rules are never named. `security.is_sensitive_entry` applies the same name rules as `_classify` (`FORBIDDEN_NAMES`, `.env` files other than `.example`/`.sample`/`.template`, the `Library` credential directories, case-folded). They are counted in `skipped`, and a skipped directory is not entered.
3. No symlink is followed. A link is listed as `link` with no target and a size of `null`. Directories are opened with `O_DIRECTORY | O_NOFOLLOW` relative to their parent's descriptor, so a directory swapped for a link during the walk fails to open instead of being entered.
4. No file is opened or read. Sizes come from `lstat`.
5. A file path is an `ERROR`. A directory that cannot be opened is an `ERROR`; a subdirectory that cannot be read or an entry that vanished is counted in `unreadable` rather than dropped silently.

## Limits and known gaps

- The top directory is resolved by `realpath` between the safety check and the open, so a swap in that window is possible in theory; the walk below it is protected as described.
- Names inside allowed directories are shown as they are; only the credential-name rules hide anything.
- Depth and entry caps bound the work of one call; a huge directory stops at `max_entries` and says `truncated`.

Tests: `tests/test_list_files.py`.
