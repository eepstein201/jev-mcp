import os
import logging

logger = logging.getLogger(__name__)

IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "build", "dist"}
IGNORE_EXTS = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".tar", ".gz", ".bin", ".exe", ".dll"}

def walk_repository(root_path: str, max_depth: int = 10) -> list[str]:
    """Walks the directory and returns a list of valid code file paths."""
    valid_files = []
    
    root_path = os.path.abspath(root_path)
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Calculate current depth
        rel_path = os.path.relpath(dirpath, root_path)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        if depth > max_depth:
            dirnames[:] = []
            continue
            
        # Filter directories in place
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IGNORE_EXTS or f.startswith("."):
                continue
                
            full_path = os.path.join(dirpath, f)
            # Basic sanity check to avoid massive files (e.g. minified JS)
            try:
                if os.path.getsize(full_path) < 1024 * 500: # 500 KB limit
                    valid_files.append(full_path)
            except OSError:
                pass
                
    return valid_files

def generate_repo_map(file_paths: list[str], root_path: str) -> str:
    """Generates a simple tree representation of the repository."""
    tree_lines = []
    for fp in file_paths:
        rel = os.path.relpath(fp, root_path)
        tree_lines.append(f"- {rel}")
    return "\n".join(tree_lines)
