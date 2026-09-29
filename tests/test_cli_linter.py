import json
import pytest
from unittest.mock import patch
import sys
import io
from jev_mcp.cli_linter import main

def test_cli_linter_invalid_json(tmp_path):
    p = tmp_path / "invalid.json"
    p.write_text("{invalid json")
    
    with patch.object(sys, 'argv', ['cli_linter.py', str(p)]):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 1
            assert "ERROR: FAILED_TO_PARSE" in fake_out.getvalue()

def test_cli_linter_no_questions(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"other": "data"}))
    
    with patch.object(sys, 'argv', ['cli_linter.py', str(p)]):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 0
            assert "WARNING: NO_QUESTIONS" in fake_out.getvalue()

def test_cli_linter_noul_question(tmp_path):
    p = tmp_path / "noul.json"
    p.write_text(json.dumps([{"type": "noul", "prompt": "Is this working?"}]))
    
    with patch.object(sys, 'argv', ['cli_linter.py', str(p)]):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 0

def test_cli_linter_choice_question_with_errors(tmp_path):
    p = tmp_path / "choice.json"
    # Choice with only 1 option will trigger INVALID_OPTION_COUNT ERROR
    p.write_text(json.dumps([{"type": "choice", "prompt": "What color?", "options": ["Red"]}]))
    
    with patch.object(sys, 'argv', ['cli_linter.py', str(p)]):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 1
            assert "ERROR: INVALID_OPTION_COUNT" in fake_out.getvalue()

def test_cli_linter_score_question_with_warnings(tmp_path):
    p = tmp_path / "score.json"
    # Score without issues should return 0 exit code
    p.write_text(json.dumps([{"type": "score", "prompt": "Rate it", "labels": ["1", "2", "3"]}]))
    
    with patch.object(sys, 'argv', ['cli_linter.py', str(p)]):
        with patch('sys.stdout', new=io.StringIO()) as fake_out:
            with pytest.raises(SystemExit) as e:
                main()
            assert e.value.code == 0
