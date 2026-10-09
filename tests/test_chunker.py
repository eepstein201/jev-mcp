import pytest
from jev_mcp.chunker import SemanticChunker

def test_chunker_python():
    chunker = SemanticChunker()
    code = """
import os

class MyClass:
    def method1(self):
        pass

def standalone_func():
    return True
"""
    chunks = chunker.get_semantic_chunks("test.py", code)
    assert len(chunks) in (2, 3)
    assert any("class MyClass:" in c for c in chunks)
    assert any("def standalone_func():" in c for c in chunks)

def test_chunker_javascript():
    chunker = SemanticChunker()
    code = """
function doSomething() {
    return false;
}
const x = 5;
"""
    chunks = chunker.get_semantic_chunks("app.js", code)
    assert len(chunks) == 1
    assert "function doSomething()" in chunks[0]
    
def test_chunker_typescript():
    chunker = SemanticChunker()
    code = """
interface MyInterface {
    id: number;
}
class TSClass {
    constructor() {}
}
"""
    chunks = chunker.get_semantic_chunks("app.ts", code)
    assert len(chunks) == 1
    assert "class TSClass {" in chunks[0]

def test_chunker_unsupported_language():
    chunker = SemanticChunker()
    code = "this is just a huge block of text " * 100
    chunks = chunker.get_semantic_chunks("notes.txt", code)
    # The text wrapper splits at 2000 chars, so there will be multiple
    assert len(chunks) > 0
    assert "this is just a huge block of text" in chunks[0]
    
def test_chunker_fallback_on_parse_error(monkeypatch):
    chunker = SemanticChunker()
    
    # Force a parse error by mocking tree_sitter
    def mock_parse(*args, **kwargs):
        raise Exception("Mock parse error")
    
    # We can just pass invalid bytes or mock the parser
    # But an easier way is to just feed it completely invalid python that causes no nodes? 
    # Tree-sitter is very fault-tolerant. Let's just monkeypatch the parser.
    import tree_sitter
    monkeypatch.setattr(tree_sitter.Parser, "parse", mock_parse)
    
    code = "def a(): pass"
    chunks = chunker.get_semantic_chunks("test.py", code)
    # It should fallback to text wrap
    assert len(chunks) == 1
    assert "def a(): pass" in chunks[0]
