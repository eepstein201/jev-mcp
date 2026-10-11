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

# Per-user scratch space stays readable: on macOS the system temp root lives
# under /private/var, which the forbidden prefixes below would otherwise block.
_TEMP_ROOT = os.path.realpath(tempfile.gettempdir())

# Every comparison below is on the case-folded path: macOS filesystems are
# case-insensitive, so ~/.SSH opens ~/.ssh.
FORBIDDEN_NAMES = frozenset({
    ".ssh", ".aws", ".gnupg", ".kube", ".npmrc", ".bash_history", ".zsh_history", ".config",
    ".netrc", ".git-credentials", ".docker", ".azure", ".pgpass", ".pypirc", ".jev-mcp",
    ".credentials.json", "client_secrets.json", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
})
# Directories under a Library folder that hold keychains, cookies and browser profiles.
FORBIDDEN_LIBRARY_DIRS = frozenset({"keychains", "cookies", "application support"})
ENV_FILE_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")
FORBIDDEN_PREFIXES = ("/etc", "/var", "/dev", "/usr", "/bin", "/sbin", "/opt", "/system", "/private")


def _is_env_file(name: str) -> bool:
    if name == ".env":
        return True
    return name.startswith(".env.") and not name.endswith(ENV_FILE_TEMPLATE_SUFFIXES)


def is_safe_path(requested_path: str) -> bool:
    """
    Checks if the requested file path is safe to access.
    Prevents directory traversal into sensitive OS or user credential directories.
    """
    try:
        real = os.path.realpath(os.path.expanduser(requested_path))
        resolved = real.casefold()
        parts = resolved.split("/")

        # Block sensitive credential/history names wherever they live —
        # including inside the temp root (scratch copies of keys are a real pattern).
        if any(part in FORBIDDEN_NAMES or _is_env_file(part) for part in parts):
            return False
        if any(parent == "library" and child in FORBIDDEN_LIBRARY_DIRS for parent, child in zip(parts, parts[1:])):
            return False

        # Allow the per-user temp root (macOS $TMPDIR lives under /private/var)
        if real == _TEMP_ROOT or real.startswith(_TEMP_ROOT + os.sep):
            return True

        # Block known dangerous absolute paths (OS level)
        return not any(resolved == p or resolved.startswith(p + "/") for p in FORBIDDEN_PREFIXES)
    except Exception:
        return False
