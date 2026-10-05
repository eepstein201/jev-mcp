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

def is_safe_path(requested_path: str) -> bool:
    """
    Checks if the requested file path is safe to access.
    Prevents directory traversal into sensitive OS or user credential directories.
    """
    try:
        resolved = os.path.realpath(requested_path)
        
        # Block known dangerous absolute paths (OS level)
        forbidden_prefixes = ["/etc", "/var", "/dev", "/usr", "/bin", "/sbin", "/opt", "/System", "/private"]
        for p in forbidden_prefixes:
            if resolved.startswith(p + "/") or resolved == p:
                return False
        
        # Block sensitive user credential/history directories
        forbidden_parts = [".ssh", ".aws", ".gnupg", ".kube", ".npmrc", ".bash_history", ".zsh_history", ".config"]
        for part in forbidden_parts:
            if f"/{part}/" in resolved or resolved.endswith(f"/{part}"):
                return False
                
        return True
    except Exception:
        return False
