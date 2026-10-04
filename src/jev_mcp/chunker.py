import logging
import textwrap

logger = logging.getLogger(__name__)

def fallback_chunker(code: str, chunk_size: int = 1500) -> list[str]:
    """Fallback simple recursive text chunker."""
    raw_chunks = [c.strip() for c in code.split("\n\n") if c.strip()]
    if not raw_chunks:
        raw_chunks = [c.strip() for c in code.split("\n") if c.strip()]
        
    chunks = []
    for c in raw_chunks:
        if len(c) > chunk_size:
            chunks.extend(textwrap.wrap(c, chunk_size))
        else:
            chunks.append(c)
    return chunks

class SemanticChunker:
    """Uses Tree-sitter to break code into semantic blocks (functions, classes)."""
    
    def __init__(self):
        self.languages = {}
        try:
            import tree_sitter_python
            import tree_sitter
            self.languages[".py"] = tree_sitter.Language(tree_sitter_python.language())
        except ImportError:  # pragma: no cover
            logger.warning("tree-sitter-python not installed. Falling back to text chunking for Python.")
            
        try:
            import tree_sitter_javascript
            import tree_sitter
            self.languages[".js"] = tree_sitter.Language(tree_sitter_javascript.language())
            self.languages[".jsx"] = tree_sitter.Language(tree_sitter_javascript.language())
        except ImportError:  # pragma: no cover
            pass

        try:
            import tree_sitter_typescript
            import tree_sitter
            self.languages[".ts"] = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            self.languages[".tsx"] = tree_sitter.Language(tree_sitter_typescript.language_tsx())
        except ImportError:  # pragma: no cover
            pass

    def get_semantic_chunks(self, file_path: str, code: str) -> list[str]:
        import os
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.languages:
            return fallback_chunker(code)
            
        try:
            import tree_sitter
            parser = tree_sitter.Parser(self.languages[ext])
            tree = parser.parse(bytes(code, "utf8"))
            
            # Simple traversal to find top-level function/class declarations
            chunks = []
            
            # Target nodes we want to extract
            target_types = {
                "function_definition", "class_definition", 
                "function_declaration", "method_definition", "class_declaration"
            }
            
            def traverse(node):
                if node.type in target_types:
                    chunks.append(node.text.decode("utf8"))
                else:
                    for child in node.children:
                        traverse(child)
            
            traverse(tree.root_node)
            
            if not chunks:
                return fallback_chunker(code)
                
            return chunks
        except Exception as e:
            logger.warning(f"Tree-sitter failed for {file_path}: {e}. Using fallback.")
            return fallback_chunker(code)

