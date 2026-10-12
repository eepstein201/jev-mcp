import re
import unicodedata
import json
from typing import Any


def sanitize_payload(text: Any) -> str:
    """
    Sanitizes LLM input by enforcing Stringify-Then-Sanitize ordering,
    NFKC normalization (Homoglyphs), and exact-match space substitution
    to prevent structural context breakout attacks and recursive WAF evasion.
    """
    if not isinstance(text, str):
        text = json.dumps(text)

    normalized_text = unicodedata.normalize("NFKC", text)

    # Exact-match regex to strip known control tokens without wildcards
    # Replaced with space to prevent recursive collapse (e.g. [IN[INST]ST])
    pattern = (
        r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|start_header_id\|>|"
        r"<\|end_header_id\|>|<\|eot_id\|>|\[/?INST\]|<<SYS>>|<</SYS>>|"
        r"<\|user\|>|<\|assistant\|>|<\|system\|>|<\|end\|>|<start_of_turn>|<end_of_turn>)"
    )

    # Ignore case and replace with space
    return re.sub(pattern, " ", normalized_text, flags=re.IGNORECASE)


import os
import tempfile
from collections.abc import Iterable

# Per-user scratch space stays readable: on macOS the system temp root lives
# under /private/var, which the forbidden prefixes below would otherwise block.
_TEMP_ROOT = os.path.realpath(tempfile.gettempdir())

# Every comparison below is on the case-folded path: macOS filesystems are
# case-insensitive, so ~/.SSH opens ~/.ssh.
FORBIDDEN_NAMES = frozenset(
    {
        ".ssh",
        ".aws",
        ".gnupg",
        ".kube",
        ".npmrc",
        ".bash_history",
        ".zsh_history",
        ".config",
        ".netrc",
        ".git-credentials",
        ".docker",
        ".azure",
        ".pgpass",
        ".pypirc",
        ".jev-mcp",
        ".credentials.json",
        "client_secrets.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
    }
)
# Directories under a Library folder that hold keychains, cookies and browser profiles.
FORBIDDEN_LIBRARY_DIRS = frozenset({"keychains", "cookies", "application support"})
ENV_FILE_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")
FORBIDDEN_PREFIXES = (
    "/etc",
    "/var",
    "/dev",
    "/usr",
    "/bin",
    "/sbin",
    "/opt",
    "/system",
    "/private",
)


def _is_env_file(name: str) -> bool:
    if name == ".env":
        return True
    return name.startswith(".env.") and not name.endswith(ENV_FILE_TEMPLATE_SUFFIXES)


ALLOWED_ROOTS_ENV = "JEV_ALLOWED_ROOTS"
JEV_HOME = "~/.jev"

SENSITIVE_MESSAGE = (
    "Access denied: '{path}' resolves to a sensitive system or credential location."
)
OUTSIDE_ROOTS_MESSAGE = (
    "Access denied: '{path}' is outside the allowed roots (the server's working "
    f"directory, {JEV_HOME}, the temp directory, and anything listed in "
    f"{ALLOWED_ROOTS_ENV}, separated by '{os.pathsep}')."
)


def _resolve_roots(raw_roots: Iterable[str]) -> list[str]:
    roots = []
    for raw in raw_roots:
        real = os.path.realpath(os.path.expanduser(raw))
        # The filesystem root would allow everything, so it never counts as a root.
        if real != os.sep:
            roots.append(real.casefold())
    return roots


def _within(resolved: str, root: str) -> bool:
    return resolved == root or resolved.startswith(root.rstrip("/") + "/")


def _explicit_roots() -> list[str]:
    raw = os.environ.get(ALLOWED_ROOTS_ENV, "")
    return _resolve_roots(r for r in raw.split(os.pathsep) if r.strip())


def _default_roots() -> list[str]:
    return _resolve_roots([os.getcwd(), JEV_HOME])


def _classify(requested_path: str) -> str | None:
    """None when the path is allowed, else "sensitive" or "outside_roots"."""
    real = os.path.realpath(os.path.expanduser(requested_path))
    resolved = real.casefold()
    parts = resolved.split("/")

    # Block sensitive credential/history names wherever they live, including inside
    # an allowed root or the temp root (scratch copies of keys are a real pattern).
    if any(part in FORBIDDEN_NAMES or _is_env_file(part) for part in parts):
        return "sensitive"
    if any(
        parent == "library" and child in FORBIDDEN_LIBRARY_DIRS
        for parent, child in zip(parts, parts[1:])
    ):
        return "sensitive"

    # Allow the per-user temp root (macOS $TMPDIR lives under /private/var)
    if real == _TEMP_ROOT or real.startswith(_TEMP_ROOT + os.sep):
        return None

    # A root the user listed is a deliberate choice, so it may sit under a system prefix.
    if any(_within(resolved, root) for root in _explicit_roots()):
        return None

    # Block known dangerous absolute paths (OS level)
    if any(resolved == p or resolved.startswith(p + "/") for p in FORBIDDEN_PREFIXES):
        return "sensitive"

    if not any(_within(resolved, root) for root in _default_roots()):
        return "outside_roots"
    return None


def is_safe_path(requested_path: str) -> bool:
    """
    Checks if the requested file path is safe to access.
    Only paths under the working directory, ~/.jev, the temp directory or a root
    listed in JEV_ALLOWED_ROOTS are allowed, and credential locations stay blocked
    inside all of them.
    """
    try:
        return _classify(requested_path) is None
    except Exception:
        return False


def denial_message(requested_path: str) -> str:
    """The user-facing reason is_safe_path refused the path."""
    try:
        reason = _classify(requested_path)
    except Exception:
        reason = "sensitive"
    template = OUTSIDE_ROOTS_MESSAGE if reason == "outside_roots" else SENSITIVE_MESSAGE
    return template.format(path=requested_path)
