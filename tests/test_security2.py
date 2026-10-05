from jev_mcp.security import is_safe_path

def test_is_safe_path():
    # Safe paths
    assert is_safe_path("/Users/test/workspace/file.py") == True
    assert is_safe_path("local_file.txt") == True
    
    # Dangerous absolute paths
    assert is_safe_path("/etc/passwd") == False
    assert is_safe_path("/var/log/system.log") == False
    assert is_safe_path("/dev/null") == False
    assert is_safe_path("/usr/bin/local") == False
    assert is_safe_path("/bin/bash") == False
    assert is_safe_path("/sbin/ifconfig") == False
    assert is_safe_path("/opt/homebrew") == False
    assert is_safe_path("/System/Library") == False
    
    # Exactly dangerous paths
    assert is_safe_path("/etc") == False
    
    # Sensitive credential/history directories
    assert is_safe_path("/Users/test/.ssh/id_rsa") == False
    assert is_safe_path("/Users/test/.aws/credentials") == False
    assert is_safe_path("/Users/test/.gnupg/secring.gpg") == False
    assert is_safe_path("/Users/test/.kube/config") == False
    assert is_safe_path("/Users/test/.npmrc") == False
    assert is_safe_path("/Users/test/.bash_history") == False
    assert is_safe_path("/Users/test/.zsh_history") == False
    assert is_safe_path("/Users/test/.config/opencode.json") == False
    
    # Exception testing
    assert is_safe_path(None) == False
