import pytest
from jev_mcp.security import sanitize_payload

def test_sanitize_payload_string():
    assert sanitize_payload("hello") == "hello"

def test_sanitize_payload_dict():
    assert sanitize_payload({"key": "value"}) == '{"key": "value"}'

def test_sanitize_payload_homoglyphs():
    # '𝒶' (U+1D4B6) normalizes to 'a'
    assert sanitize_payload("𝒶") == "a"

def test_sanitize_payload_control_tokens():
    text = "<|im_start|>user\nHello<|im_end|>"
    assert sanitize_payload(text) == " user\nHello "

def test_sanitize_payload_recursive_collapse():
    text = "[IN[INST]ST]"
    assert sanitize_payload(text) == "[IN ST]"

def test_sanitize_payload_case_insensitive():
    text = "<|IM_START|>user"
    assert sanitize_payload(text) == " user"
