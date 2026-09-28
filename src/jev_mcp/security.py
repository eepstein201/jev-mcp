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
