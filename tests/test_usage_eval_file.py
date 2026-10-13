"""The tool-usage evaluation file: ten read-only, well-formed questions with short answers."""

import xml.etree.ElementTree as ET
from pathlib import Path

from tests.test_tool_surface import ANNOTATIONS, EXPECTED_TOOLS

EVAL = Path(__file__).parent / "evals" / "usage" / "evaluation.xml"
WRITE_TOOLS = {
    name for name, (read_only, *_rest) in ANNOTATIONS.items() if not read_only
}
ANSWER_MAX_CHARS = 60


def _pairs():
    return ET.parse(EVAL).getroot().findall("qa_pair")


def test_the_usage_evaluation_has_ten_well_formed_read_only_questions():
    # Arrange
    pairs = _pairs()

    # Assert
    assert len(pairs) == 10
    for pair in pairs:
        question, answer = pair.findtext("question"), pair.findtext("answer")
        assert question and answer and len(answer.strip()) <= ANSWER_MAX_CHARS
        assert not any(tool in question for tool in WRITE_TOOLS)
    assert len({p.findtext("question") for p in pairs}) == 10


def test_no_question_names_any_tool():
    # The model must pick the tool from its description, so a question never names one.
    for pair in _pairs():
        question = pair.findtext("question")
        assert not any(tool in question for tool in EXPECTED_TOOLS)


def test_every_question_points_at_the_seeded_fixture_home():
    # Questions that name a path must use the fixed seeded home, not a checkout path.
    for pair in _pairs():
        question = pair.findtext("question")
        assert "/Users/" not in question
