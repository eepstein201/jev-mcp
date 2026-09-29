import pytest
from jev_mcp.linter import DecisionPreflightLinter, LintFinding

@pytest.fixture
def linter():
    return DecisionPreflightLinter()

def test_linter_generative_intent(linter):
    prompt = "Summarize the customer email."
    options = [{"id": "a", "description": "Good"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "GENERATIVE_INTENT_DETECTED" for f in report.findings)

def test_linter_arithmetic_intent(linter):
    prompt = "What is 5 + 5?"
    options = [{"id": "a", "description": "10"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "ARITHMETIC_INTENT_DETECTED" for f in report.findings)

def test_linter_compound_intent(linter):
    prompt = "Is it big and red?"
    options = [{"id": "a", "description": "Yes"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "COMPOUND_QUESTION" for f in report.findings)

def test_linter_chronological_intent(linter):
    prompt = "Did event A happened before event B?"
    options = [{"id": "a", "description": "Yes"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "CHRONOLOGICAL_INTENT_DETECTED" for f in report.findings)

def test_linter_boilerplate_intent(linter):
    prompt = "You are an expert. Is this red?"
    options = [{"id": "a", "description": "Yes"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "META_PROMPTING_BOILERPLATE" for f in report.findings)

def test_linter_no_fallback_option(linter):
    prompt = "Is it red or blue?"
    options = [{"id": "a", "description": "Red"}, {"id": "b", "description": "Blue"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "MISSING_FALLBACK_OPTION" for f in report.findings)

def test_linter_has_fallback_option(linter):
    prompt = "Is it red or blue?"
    options = [{"id": "a", "description": "Red"}, {"id": "b", "description": "Blue"}, {"id": "c", "description": "None"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert not any(f.code == "MISSING_FALLBACK_OPTION" for f in report.findings)

def test_linter_overlapping_options(linter):
    prompt = "Color?"
    # To trigger OVERLAPPING_OPTIONS, we need descriptions > 10 chars, and one subsumes another.
    options = [{"id": "a", "description": "Light Blue Color"}, {"id": "b", "description": "Light Blue Color Extra"}, {"id": "c", "description": "Unknown"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "OVERLAPPING_OPTIONS" for f in report.findings)

def test_linter_imbalanced_length(linter):
    prompt = "Color?"
    options = [{"id": "a", "description": "Blue"}, {"id": "b", "description": "A very very very very long sentence that explains blue color completely and thoroughly in absolute detail"}, {"id": "c", "description": "Unknown"}]
    report = linter.lint({}, prompt, options, is_score=False)
    assert any(f.code == "UNBALANCED_OPTION_LENGTH" for f in report.findings)

def test_linter_score_scale(linter):
    prompt = "Rate it"
    options = [{"id": "a", "description": "Good"}, {"id": "b", "description": "Bad"}]
    report = linter.lint({}, prompt, options, is_score=True)
    assert any(f.code == "SUBJECTIVE_ADJECTIVE_SCALE" for f in report.findings)

def test_linter_perfect_score(linter):
    prompt = "Rate it"
    options = [{"id": "1", "description": "Low"}, {"id": "2", "description": "Med"}, {"id": "3", "description": "Unknown"}]
    report = linter.lint({}, prompt, options, is_score=True)
    assert len(report.findings) == 0

def test_linter_multi_token_labels(linter):
    prompt = "Rate it"
    options = [{"id": "abc", "description": "Low"}, {"id": "def", "description": "Med"}, {"id": "ghi", "description": "Unknown"}]
    report = linter.lint({}, prompt, options, is_score=True)
    assert any(f.code == "MULTI_TOKEN_LABELS" for f in report.findings)

def test_linter_excessive_scale_points(linter):
    prompt = "Rate it"
    options = [{"id": str(i), "description": f"Level {i}"} for i in range(6)]
    report = linter.lint({}, prompt, options, is_score=True)
    assert any(f.code == "EXCESSIVE_SCALE_POINTS" for f in report.findings)

def test_linter_reversed_polarity(linter):
    prompt = "Rate it"
    options = [{"id": "1", "description": "Excellent"}, {"id": "2", "description": "Med"}, {"id": "3", "description": "Unknown"}]
    report = linter.lint({}, prompt, options, is_score=True)
    assert any(f.code == "REVERSED_POLARITY" for f in report.findings)

