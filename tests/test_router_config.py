import os
import json
from unittest.mock import patch, mock_open
from jev_mcp.router_config import load_router_config, save_router_config, ROUTER_CONFIG_PATH

def test_load_router_config_default():
    with patch("os.path.exists", return_value=False):
        config = load_router_config()
        assert "buckets" in config
        assert "rules" in config
        assert config["buckets"]["b1"] == "claude-3-5-haiku"

def test_load_router_config_existing():
    mock_config = {"buckets": {"b1": "test-model"}, "rules": []}
    mock_json = json.dumps(mock_config)
    
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_json)):
            config = load_router_config()
            assert config["buckets"]["b1"] == "test-model"

def test_save_router_config():
    mock_config = {"buckets": {"b1": "test-model"}, "rules": []}
    
    with patch("os.makedirs") as mock_makedirs:
        with patch("builtins.open", mock_open()) as mock_file:
            save_router_config(mock_config)
            
            mock_makedirs.assert_called_once_with(os.path.dirname(ROUTER_CONFIG_PATH), exist_ok=True)
            mock_file.assert_called_once_with(ROUTER_CONFIG_PATH, "w")
            
            # json.dump makes multiple write calls, so we just check if it was called
            # and the file was opened correctly
            handle = mock_file()
            assert handle.write.called
