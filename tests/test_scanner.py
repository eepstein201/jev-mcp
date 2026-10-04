import os
import pytest
from jev_mcp.scanner import walk_repository, generate_repo_map

def test_scanner_basics(tmp_path):
    # tmp_path is a built-in pytest fixture providing a temporary directory
    repo_dir = tmp_path / "my_repo"
    repo_dir.mkdir()
    
    (repo_dir / "main.py").write_text("print('hello')")
    (repo_dir / "utils.js").write_text("console.log('hi')")
    
    # Create ignored folders
    node_modules = repo_dir / "node_modules"
    node_modules.mkdir()
    (node_modules / "lib.js").write_text("ignore me")
    
    git_dir = repo_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("ignore me")
    
    # Create invalid extension
    (repo_dir / "image.png").write_text("binary")
    
    files = walk_repository(str(repo_dir))
    
    assert len(files) == 2
    assert any("main.py" in f for f in files)
    assert any("utils.js" in f for f in files)
    assert not any("node_modules" in f for f in files)
    assert not any(".git" in f for f in files)
    assert not any("image.png" in f for f in files)

def test_generate_repo_map(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    (repo_dir / "a.py").write_text("")
    
    subdir = repo_dir / "src"
    subdir.mkdir()
    (subdir / "b.py").write_text("")
    
    files = walk_repository(str(repo_dir))
    tree_map = generate_repo_map(files, str(repo_dir))
    
    assert "a.py" in tree_map
    assert "src/" in tree_map
    assert "b.py" in tree_map


def test_scanner_max_depth(tmp_path):
    repo_dir = tmp_path / "deep_repo"
    repo_dir.mkdir()
    
    current = repo_dir
    for i in range(15):
        current = current / f"dir_{i}"
        current.mkdir()
        (current / f"file_{i}.py").write_text("code")
        
    files = walk_repository(str(repo_dir), max_depth=5)
    
    # It should only find files up to depth 5
    # Depth 0: root. Depth 1: dir_0. Depth 2: dir_1. ...
    # So we should only have 5 files.
    assert len(files) == 5
    assert not any("dir_10" in f for f in files)
